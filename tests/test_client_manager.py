# Copyright (c) 2026 Matthew Fuchs
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from extensible_mcp._mcp_compat import http_lib, result_is_error
from extensible_mcp.client_manager import (
    ClientManager,
    TokenExpiredError,
    _AuthWatch,
    _Connection,
    _read_tokens_file,
)
from extensible_mcp.types import ServerConfig


MOCK_SERVER_PATH = str(Path(__file__).parent / "mock_server.py")


class TestClientManager:
    @pytest.mark.asyncio
    async def test_connect_and_index(self):
        """Connect to the mock server and verify tools are indexed."""
        mgr = ClientManager()
        configs = [
            ServerConfig(
                name="mock",
                command=sys.executable,
                args=[MOCK_SERVER_PATH],
            )
        ]
        try:
            tools = await mgr.connect_all(configs)
            assert len(tools) == 4
            names = {t.qualified_name for t in tools}
            assert "mock__add_numbers" in names
            assert "mock__send_email" in names
            assert "mock__search_files" in names
            assert "mock__delete_files" in names
        finally:
            await mgr.close_all()

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Call a tool on the mock server and verify the result."""
        mgr = ClientManager()
        configs = [
            ServerConfig(
                name="mock",
                command=sys.executable,
                args=[MOCK_SERVER_PATH],
            )
        ]
        try:
            await mgr.connect_all(configs)
            result = await mgr.call_tool("mock__add_numbers", {"a": 2, "b": 3})
            assert not result_is_error(result)
            text = result.content[0].text
            assert "5" in text
        finally:
            await mgr.close_all()

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        """Calling an unknown tool raises ValueError."""
        mgr = ClientManager()
        with pytest.raises(ValueError, match="Unknown tool"):
            await mgr.call_tool("nonexistent__tool", {})

    @pytest.mark.asyncio
    async def test_partial_startup(self):
        """One failing server doesn't prevent others from connecting."""
        mgr = ClientManager()
        configs = [
            ServerConfig(name="bad", command="/nonexistent/binary", args=[]),
            ServerConfig(
                name="mock",
                command=sys.executable,
                args=[MOCK_SERVER_PATH],
            ),
        ]
        try:
            tools = await mgr.connect_all(configs)
            assert len(tools) == 4  # only mock server's tools
            assert "mock__add_numbers" in mgr.get_qualified_names()
        finally:
            await mgr.close_all()

    @pytest.mark.asyncio
    async def test_all_servers_fail(self):
        """If all servers fail, connect_all raises RuntimeError."""
        mgr = ClientManager()
        configs = [
            ServerConfig(name="bad1", command="/nonexistent1", args=[]),
            ServerConfig(name="bad2", command="/nonexistent2", args=[]),
        ]
        with pytest.raises(RuntimeError, match="Could not connect"):
            await mgr.connect_all(configs)


class TestReadTokensFile:
    def test_missing_file_returns_empty(self, tmp_path):
        assert _read_tokens_file(tmp_path / "nope") == {}

    def test_parses_simple_pairs(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("github=ghp_xxx\nnotion=secret_yyy\n")
        assert _read_tokens_file(path) == {"github": "ghp_xxx", "notion": "secret_yyy"}

    def test_strips_whitespace_around_key_and_value(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("  github  =  ghp_xxx  \n")
        assert _read_tokens_file(path) == {"github": "ghp_xxx"}

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("\n\ngithub=ghp_xxx\n\n")
        assert _read_tokens_file(path) == {"github": "ghp_xxx"}

    def test_skips_comments(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("# this is a comment\ngithub=ghp_xxx\n# trailing comment\n")
        assert _read_tokens_file(path) == {"github": "ghp_xxx"}

    def test_skips_lines_without_equals(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("garbage line\ngithub=ghp_xxx\n")
        assert _read_tokens_file(path) == {"github": "ghp_xxx"}

    def test_strips_matching_double_quotes(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text('github="ghp_xxx"\n')
        assert _read_tokens_file(path) == {"github": "ghp_xxx"}

    def test_strips_matching_single_quotes(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("github='ghp_xxx'\n")
        assert _read_tokens_file(path) == {"github": "ghp_xxx"}

    def test_keeps_mismatched_quotes(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("github=\"ghp_xxx'\n")
        assert _read_tokens_file(path) == {"github": '"ghp_xxx\''}

    def test_value_with_internal_equals(self, tmp_path):
        # `partition` splits on first `=`, so JWT-style values with `=` survive.
        path = tmp_path / "tokens"
        path.write_text("api=eyJhbGc=signature\n")
        assert _read_tokens_file(path) == {"api": "eyJhbGc=signature"}

    def test_later_entry_overrides_earlier(self, tmp_path):
        path = tmp_path / "tokens"
        path.write_text("github=old\ngithub=new\n")
        assert _read_tokens_file(path) == {"github": "new"}


def _make_conn(server_name: str = "test-server") -> _Connection:
    """Build a _Connection whose token-age machinery is initialized."""
    conn = _Connection(ServerConfig(name=server_name, url="https://example.com/mcp"))
    conn._last_token_value = "some-token"
    conn._token_set_at = 0.0
    return conn


def _httpx_status_error(code: int) -> Exception:
    # The status error of whichever HTTP library the installed SDK uses, with a
    # message that carries no status code, so only the isinstance check can
    # recognise it.
    request = http_lib.Request("POST", "https://example.com/mcp")
    response = http_lib.Response(code, request=request)
    return http_lib.HTTPStatusError("request failed", request=request, response=response)


class TestCheckAuthError:
    def test_httpx_401_raises_token_expired(self):
        conn = _make_conn()
        with pytest.raises(TokenExpiredError) as info:
            conn._check_auth_error(_httpx_status_error(401))
        assert info.value.server_name == "test-server"

    def test_httpx_403_raises_token_expired(self):
        conn = _make_conn()
        with pytest.raises(TokenExpiredError):
            conn._check_auth_error(_httpx_status_error(403))

    def test_httpx_500_does_not_raise(self):
        conn = _make_conn()
        # Returns None; caller is expected to re-raise the original exception.
        assert conn._check_auth_error(_httpx_status_error(500)) is None

    def test_rejected_status_seen_by_the_client_raises_token_expired(self):
        # mcp 2.x reports a 401 as a JSON-RPC error with no status in it, so the
        # response hook's record is the only sign it was auth.
        conn = _make_conn()
        exc = RuntimeError("Server returned an error response")
        assert conn._check_auth_error(exc) is None
        with pytest.raises(TokenExpiredError):
            conn._check_auth_error(exc, auth_rejected=True)

    def test_message_contains_401(self):
        conn = _make_conn()
        with pytest.raises(TokenExpiredError):
            conn._check_auth_error(RuntimeError("Got status 401 from server"))

    def test_message_contains_403(self):
        conn = _make_conn()
        with pytest.raises(TokenExpiredError):
            conn._check_auth_error(RuntimeError("Got status 403 from server"))

    def test_message_contains_unauthorized(self):
        conn = _make_conn()
        with pytest.raises(TokenExpiredError):
            conn._check_auth_error(RuntimeError("Request unauthorized"))

    def test_message_contains_forbidden(self):
        conn = _make_conn()
        with pytest.raises(TokenExpiredError):
            conn._check_auth_error(RuntimeError("Action FORBIDDEN by policy"))

    def test_unrelated_exception_passes_through(self):
        conn = _make_conn()
        assert conn._check_auth_error(ValueError("totally unrelated")) is None

    def test_exception_group_with_401_raises(self):
        """anyio's TaskGroup wraps inner errors; the check should recurse."""
        conn = _make_conn()
        eg = ExceptionGroup("task group failed", [_httpx_status_error(401)])
        with pytest.raises(TokenExpiredError):
            conn._check_auth_error(eg)

    def test_exception_group_with_unrelated_passes_through(self):
        conn = _make_conn()
        eg = ExceptionGroup("task group failed", [ValueError("unrelated")])
        assert conn._check_auth_error(eg) is None


class TestAuthWatch:
    @pytest.mark.parametrize(("status", "rejected"), [(200, False), (401, True), (403, True), (500, False)])
    def test_records_only_auth_rejections(self, status, rejected):
        watch = _AuthWatch()
        asyncio.run(watch.on_response(MagicMock(status_code=status)))
        assert watch.rejected is rejected

    def test_stays_rejected_after_a_later_success(self):
        watch = _AuthWatch()
        asyncio.run(watch.on_response(MagicMock(status_code=401)))
        asyncio.run(watch.on_response(MagicMock(status_code=200)))
        assert watch.rejected
