# Copyright (c) 2026 Matthew Fuchs
# SPDX-License-Identifier: Apache-2.0

"""Bridges the MCP SDK's 1.x -> 2.x break so the proxy runs on either.

Three things changed under us in mcp 2.0:

* ``streamable_http_client`` yields ``(read, write)``; 1.x yields
  ``(read, write, get_session_id)``.
* Its ``http_client`` must be an ``httpx2.AsyncClient``; 1.x takes ``httpx``.
* Model fields went snake_case: ``Tool.inputSchema`` -> ``input_schema``,
  ``CallToolResult.isError`` -> ``is_error``. fastmcp 4 shims the old names
  with a deprecation warning, but that is fastmcp's shim, not the SDK's.

Everything that touches those goes through here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Any

import mcp.types as mcp_types
from mcp.client.streamable_http import streamable_http_client

MCP_MAJOR = int(version("mcp").split(".", 1)[0])

if MCP_MAJOR >= 2:
    import httpx2 as http_lib
else:
    import httpx as http_lib  # type: ignore[no-redef]

HTTPStatusError = http_lib.HTTPStatusError


def new_http_client(
    headers: dict[str, str],
    on_response: Callable[[Any], Awaitable[None]] | None = None,
) -> Any:
    """An async HTTP client of the kind this SDK's transport accepts."""
    hooks = {"response": [on_response]} if on_response else {}
    return http_lib.AsyncClient(headers=headers, event_hooks=hooks)


@asynccontextmanager
async def http_streams(url: str, http_client: Any = None) -> AsyncIterator[tuple[Any, Any]]:
    """``streamable_http_client`` narrowed to ``(read, write)`` on any SDK version."""
    async with streamable_http_client(url, http_client=http_client) as streams:
        yield streams[0], streams[1]


def tool_input_schema(tool: mcp_types.Tool) -> dict[str, Any]:
    if "input_schema" in type(tool).model_fields:
        return tool.input_schema  # type: ignore[attr-defined,no-any-return]
    return tool.inputSchema  # type: ignore[attr-defined,no-any-return]


def result_is_error(result: mcp_types.CallToolResult) -> bool:
    if "is_error" in type(result).model_fields:
        return bool(result.is_error)  # type: ignore[attr-defined]
    return bool(result.isError)  # type: ignore[attr-defined]
