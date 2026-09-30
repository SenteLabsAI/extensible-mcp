# Copyright (c) 2026 Matthew Fuchs
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for URL-server bearer-token auth.

Spins up a real FastMCP HTTP server in-process with a bearer-check ASGI
middleware, then exercises the proxy's tokens-file machinery against it.
Covers the connect path, the rotate-token path, and the 401-mid-session
path that should surface as TokenExpiredError.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
import uvicorn
from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from extensible_mcp._mcp_compat import result_is_error
from extensible_mcp.client_manager import ClientManager, TokenExpiredError
from extensible_mcp.types import ServerConfig


class TokenStore:
    """Mutable set of accepted bearers. Tests mutate `valid` to simulate rotation."""

    def __init__(self, initial: set[str]) -> None:
        self.valid: set[str] = set(initial)


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject any request whose Authorization bearer isn't in the shared store."""

    def __init__(self, app, store: TokenStore) -> None:
        super().__init__(app)
        self.store = store

    async def dispatch(self, request, call_next):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return Response("Missing bearer", status_code=401)
        if auth[7:] not in self.store.valid:
            return Response("Bad bearer", status_code=401)
        return await call_next(request)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture
async def mock_mcp_server():
    """Start a FastMCP HTTP server with bearer auth on a random port."""
    store = TokenStore({"correct-token"})

    server = FastMCP(name="mock-bearer-server")

    @server.tool()
    def echo(message: str) -> str:
        """Return the input unchanged."""
        return message

    middleware = [Middleware(BearerAuthMiddleware, store=store)]
    app = server.http_app(middleware=middleware, transport="streamable-http")

    port = _free_port()
    config = uvicorn.Config(
        app=app, host="127.0.0.1", port=port, log_level="error", lifespan="on"
    )
    uv_server = uvicorn.Server(config)
    task = asyncio.create_task(uv_server.serve())

    # Wait up to 5s for the server to come up.
    for _ in range(100):
        if uv_server.started:
            break
        await asyncio.sleep(0.05)
    else:
        uv_server.should_exit = True
        await task
        raise RuntimeError("Mock MCP server failed to start within 5s")

    url = f"http://127.0.0.1:{port}/mcp"
    try:
        yield url, store
    finally:
        uv_server.should_exit = True
        await task


@pytest.mark.asyncio
async def test_valid_token_connects_and_indexes_tools(tmp_path, mock_mcp_server):
    url, _ = mock_mcp_server
    tokens_file = tmp_path / "tokens"
    tokens_file.write_text("mockauth=correct-token\n")

    mgr = ClientManager(tokens_file=tokens_file)
    try:
        tools = await mgr.connect_all(
            [ServerConfig(name="mockauth", url=url)]
        )
        assert any(t.name == "echo" for t in tools)
    finally:
        await mgr.close_all()


@pytest.mark.asyncio
async def test_invalid_token_fails_to_connect(tmp_path, mock_mcp_server):
    url, _ = mock_mcp_server
    tokens_file = tmp_path / "tokens"
    tokens_file.write_text("mockauth=wrong-token\n")

    mgr = ClientManager(tokens_file=tokens_file)
    with pytest.raises(RuntimeError, match="Could not connect"):
        await mgr.connect_all([ServerConfig(name="mockauth", url=url)])


@pytest.mark.asyncio
async def test_missing_token_fails_to_connect(tmp_path, mock_mcp_server):
    """No token entry for the server name → no Authorization header → 401."""
    url, _ = mock_mcp_server
    tokens_file = tmp_path / "tokens"
    tokens_file.write_text("# no entry for mockauth\n")

    mgr = ClientManager(tokens_file=tokens_file)
    with pytest.raises(RuntimeError, match="Could not connect"):
        await mgr.connect_all([ServerConfig(name="mockauth", url=url)])


@pytest.mark.asyncio
async def test_call_after_server_revokes_token_translates_to_token_expired(
    tmp_path, mock_mcp_server
):
    """Connect successfully, then have the server stop accepting the token."""
    url, store = mock_mcp_server
    tokens_file = tmp_path / "tokens"
    tokens_file.write_text("mockauth=correct-token\n")

    mgr = ClientManager(tokens_file=tokens_file)
    try:
        await mgr.connect_all([ServerConfig(name="mockauth", url=url)])

        # Server-side: token is no longer accepted (simulates expiry/revocation).
        store.valid = {"different-token-now"}

        with pytest.raises(TokenExpiredError) as info:
            await mgr.call_tool("mockauth__echo", {"message": "hi"})
        assert info.value.server_name == "mockauth"
    finally:
        await mgr.close_all()


@pytest.mark.asyncio
async def test_token_rotation_via_file_picks_up_new_value(tmp_path, mock_mcp_server):
    """Update the tokens file mid-session; next call should use the new value."""
    url, store = mock_mcp_server
    tokens_file = tmp_path / "tokens"
    tokens_file.write_text("mockauth=correct-token\n")

    mgr = ClientManager(tokens_file=tokens_file)
    try:
        await mgr.connect_all([ServerConfig(name="mockauth", url=url)])

        # Rotate: server now accepts a different token, file updated to match.
        store.valid = {"new-token"}
        tokens_file.write_text("mockauth=new-token\n")

        result = await mgr.call_tool("mockauth__echo", {"message": "ping"})
        assert not result_is_error(result)
        assert "ping" in result.content[0].text
    finally:
        await mgr.close_all()
