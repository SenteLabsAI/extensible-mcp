"""A URL server reached with a bearer token gets the MCP SDK's own timeouts,
as one without a token does, not httpx's 5 s default."""

from __future__ import annotations

from pathlib import Path

from extensible_mcp._mcp_compat import HTTP_READ_TIMEOUT_S, HTTP_TIMEOUT_S, http_lib
from extensible_mcp.client_manager import _Connection
from extensible_mcp.types import ServerConfig


def test_the_timeouts_are_the_sdks_own() -> None:
    from mcp.shared._httpx_utils import (
        MCP_DEFAULT_SSE_READ_TIMEOUT,
        MCP_DEFAULT_TIMEOUT,
    )

    assert HTTP_TIMEOUT_S == MCP_DEFAULT_TIMEOUT
    assert HTTP_READ_TIMEOUT_S == MCP_DEFAULT_SSE_READ_TIMEOUT


async def test_a_bearer_token_client_has_the_sdk_timeouts(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens"
    tokens.write_text("remote=test-token\n")
    conn = _Connection(ServerConfig(name="remote", url="http://localhost:9/mcp"), tokens_file=tokens)
    client = conn._make_http_client()
    assert client is not None
    async with client:
        assert client.timeout == http_lib.Timeout(30.0, read=300.0)
        assert client.headers["Authorization"] == "Bearer test-token"
