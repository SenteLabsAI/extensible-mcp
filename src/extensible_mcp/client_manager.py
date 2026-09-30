# Copyright (c) 2026 Matthew Fuchs
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as mcp_types

from ._mcp_compat import (
    HTTPStatusError,
    http_streams,
    new_http_client,
    tool_input_schema,
)
from .types import ServerConfig, ToolRecord

logger = logging.getLogger(__name__)


def _read_tokens_file(path: Path) -> dict[str, str]:
    """Read server_name=token pairs from a tokens property file."""
    if not path.exists():
        return {}
    tokens: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        tokens[key.strip()] = value
    return tokens


class TokenExpiredError(Exception):
    """Raised when an HTTP call fails with 401/403, suggesting the token has expired."""

    def __init__(self, server_name: str, token_age_minutes: int) -> None:
        self.server_name = server_name
        self.token_age_minutes = token_age_minutes
        super().__init__(
            f"Authentication failed for server '{server_name}' "
            f"(token unchanged for {token_age_minutes} minutes)"
        )


class _AuthWatch:
    """Records whether a server answered 401/403 during one URL session.

    mcp 2.x turns an HTTP error status into a generic JSON-RPC error that no
    longer carries the status, so the exception alone cannot say it was auth.
    One per call, so concurrent calls to the same server don't share it.
    """

    def __init__(self) -> None:
        self.rejected = False

    async def on_response(self, response: Any) -> None:
        if response.status_code in (401, 403):
            self.rejected = True


class _Connection:
    """Manages a single downstream MCP server connection (stdio or URL)."""

    def __init__(self, config: ServerConfig, tokens_file: Path | None = None) -> None:
        self.config = config
        self.session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._tokens_file = tokens_file
        self._last_token_value: str | None = None
        self._token_set_at: float | None = None

    @property
    def is_url(self) -> bool:
        return self.config.url is not None

    @property
    def token_age_minutes(self) -> int:
        if self._token_set_at is None:
            return 0
        return int((time.monotonic() - self._token_set_at) / 60)

    def _resolve_token(self) -> str | None:
        """Read the current token for this server from the tokens file.

        Tracks when the token value last changed for age reporting.
        """
        if not self._tokens_file:
            return None
        tokens = _read_tokens_file(self._tokens_file)
        value = tokens.get(self.config.name)
        if value is None:
            self._last_token_value = None
            self._token_set_at = None
            return None
        if value != self._last_token_value:
            self._last_token_value = value
            self._token_set_at = time.monotonic()
        return value

    def _make_http_client(self, auth: _AuthWatch | None = None) -> Any:
        """Build an HTTP client with auth headers if a token is available."""
        token = self._resolve_token()
        if not token:
            return None
        return new_http_client(
            {"Authorization": f"Bearer {token}"},
            on_response=auth.on_response if auth else None,
        )

    async def _open_url_session(
        self, stack: AsyncExitStack, auth: _AuthWatch | None = None
    ) -> ClientSession:
        """Open an initialized session to this URL server on ``stack``."""
        http_client = self._make_http_client(auth)
        if http_client:
            await stack.enter_async_context(http_client)
        read, write = await stack.enter_async_context(
            http_streams(self.config.url, http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def _check_auth_error(self, exc: BaseException, *, auth_rejected: bool = False) -> None:
        """Raise TokenExpiredError if the exception looks like a 401/403."""
        if auth_rejected:
            raise TokenExpiredError(self.config.name, self.token_age_minutes) from exc
        if isinstance(exc, HTTPStatusError) and exc.response.status_code in (401, 403):
            raise TokenExpiredError(self.config.name, self.token_age_minutes) from exc
        msg = str(exc).lower()
        if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
            raise TokenExpiredError(self.config.name, self.token_age_minutes) from exc
        # anyio TaskGroups wrap downstream errors in an ExceptionGroup; recurse.
        if isinstance(exc, BaseExceptionGroup):
            for sub in exc.exceptions:
                self._check_auth_error(sub)

    async def connect(self) -> None:
        """Open a persistent connection. Used for stdio servers."""
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if self.config.url:
                session = await self._open_url_session(stack)
            else:
                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._stack = stack
        self.session = session

    async def list_tools_ephemeral(self) -> list[mcp_types.Tool]:
        """Connect, list tools, and disconnect. Used for URL servers to avoid
        background tasks interfering with the stdio transport."""
        async with AsyncExitStack() as stack:
            session = await self._open_url_session(stack)
            result = await session.list_tools()
            return result.tools

    async def call_tool_ephemeral(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        """Connect, call a tool, and disconnect. Used for URL servers."""
        auth = _AuthWatch()
        try:
            async with AsyncExitStack() as stack:
                session = await self._open_url_session(stack, auth)
                return await session.call_tool(tool_name, arguments)
        except TokenExpiredError:
            raise
        except Exception as exc:
            if self._last_token_value:
                self._check_auth_error(exc, auth_rejected=auth.rejected)
            raise

    async def close(self) -> None:
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
        self.session = None


class ClientManager:
    def __init__(self, tokens_file: Path | None = None) -> None:
        self._connections: dict[str, _Connection] = {}
        self._tool_to_server: dict[str, str] = {}  # qualified_name -> server_name
        self._tool_original_name: dict[str, str] = {}  # qualified_name -> original name
        self._tokens_file = tokens_file

    async def connect_all(self, configs: list[ServerConfig]) -> list[ToolRecord]:
        all_tools: list[ToolRecord] = []
        for config in configs:
            try:
                conn = _Connection(config, tokens_file=self._tokens_file)
                if conn.is_url:
                    tools_raw = await conn.list_tools_ephemeral()
                    self._connections[config.name] = conn
                    tools = self._index_tools(config.name, tools_raw)
                else:
                    await conn.connect()
                    self._connections[config.name] = conn
                    result = await conn.session.list_tools()
                    tools = self._index_tools(config.name, result.tools)
                all_tools.extend(tools)
                logger.info(
                    "Connected to '%s': %d tools", config.name, len(tools)
                )
            except Exception:
                logger.warning(
                    "Failed to connect to '%s', skipping", config.name, exc_info=True
                )
        if not self._connections:
            raise RuntimeError("Could not connect to any downstream MCP server")
        return all_tools

    def _index_tools(
        self, server_name: str, tools: list[mcp_types.Tool]
    ) -> list[ToolRecord]:
        records: list[ToolRecord] = []
        for tool in tools:
            qualified = f"{server_name}__{tool.name}"
            self._tool_to_server[qualified] = server_name
            self._tool_original_name[qualified] = tool.name
            records.append(
                ToolRecord(
                    name=tool.name,
                    qualified_name=qualified,
                    description=tool.description or "",
                    input_schema=tool_input_schema(tool),
                    server_name=server_name,
                )
            )
        return records

    async def call_tool(
        self, qualified_name: str, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        server_name = self._tool_to_server.get(qualified_name)
        if not server_name:
            raise ValueError(f"Unknown tool: {qualified_name}")

        conn = self._connections.get(server_name)
        if not conn:
            raise RuntimeError(f"Server '{server_name}' is not connected")

        original_name = self._tool_original_name[qualified_name]

        if conn.is_url:
            return await conn.call_tool_ephemeral(original_name, arguments)

        if not conn.session:
            raise RuntimeError(f"Server '{server_name}' is not connected")

        try:
            return await conn.session.call_tool(original_name, arguments)
        except Exception:
            # Reconnect once and retry
            logger.warning(
                "Call to '%s' failed, attempting reconnect", qualified_name, exc_info=True
            )
            try:
                await conn.close()
                await conn.connect()
                result = await conn.session.list_tools()
                self._index_tools(server_name, result.tools)
                return await conn.session.call_tool(original_name, arguments)
            except Exception:
                logger.error(
                    "Reconnect to '%s' failed", server_name, exc_info=True
                )
                raise

    async def connect_url(self, name: str, url: str) -> list[ToolRecord]:
        """Connect to a remote MCP server by URL and index its tools."""
        if name in self._connections:
            raise ValueError(f"Server '{name}' is already connected")
        config = ServerConfig(name=name, url=url)
        conn = _Connection(config, tokens_file=self._tokens_file)
        tools_raw = await conn.list_tools_ephemeral()
        self._connections[name] = conn
        tools = self._index_tools(name, tools_raw)
        logger.info("Connected to '%s' (%s): %d tools", name, url, len(tools))
        return tools

    def get_qualified_names(self) -> set[str]:
        return set(self._tool_to_server.keys())

    async def close_all(self) -> None:
        for conn in self._connections.values():
            await conn.close()
        self._connections.clear()
