# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

MCP Tool Retrieval Proxy — a middleware layer that sits on top of MCP servers and replaces static tool definitions with semantic search. Instead of sending all tool definitions in the prompt, the LLM gets three meta-tools (`search_tools`, `call_tool`, `load_mcp_server`) and discovers tools on demand via vector similarity search.

## Commands

```bash
# Install dependencies (uses uv)
uv sync --group dev

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_vector_store.py

# Run a single test
uv run pytest tests/test_vector_store.py::test_name -v

# Run the server
uv run extensible-mcp --config config.json
```

## Architecture

The proxy is a FastMCP server that connects to downstream MCP servers as a client, indexes their tools, and exposes three meta-tools:

1. **`search_tools(query)`** — embeds the query and runs cosine similarity against the tool index, returns matching tool definitions
2. **`call_tool(tool_name, arguments)`** — proxies the call to the correct downstream server using the qualified name (`{server}__{tool}`)
3. **`load_mcp_server(server_name, url)`** — connects to a remote (Streamable HTTP) MCP server at runtime and indexes its tools

### Filter pipelines

Three independent pipelines (`FilterPipeline`, `CallFilterPipeline`, `ServerLoadFilterPipeline`) gate search, call, and server-load operations respectively. The only structurally enforced rule is `DiscoveredToolsFilter`: the LLM cannot call any tool it has not previously surfaced via `search_tools`. The other shipped filters (`AccessControlFilter`, `RegoPolicyFilter`, `ServerLoadAccessControlFilter`, `SimilarityThresholdFilter`) are reference implementations — third parties are expected to write their own and inject them via the `extra_search_filters`, `extra_call_filters`, `extra_load_filters` kwargs on `create_server`. Custom filters run after the built-ins.

### Key modules (`src/extensible_mcp/`)

- **`server.py`** — FastMCP server setup, lifespan management, handler registration for the three meta-tools. `create_server(config, *, extra_*_filters=...)` is the public entry point for embedding the proxy in custom code; `main()` is the CLI entry.
- **`client_manager.py`** — Manages stdio and Streamable HTTP connections to downstream MCP servers. Handles connect/reconnect, tool indexing, and proxying `call_tool` requests. Tools are namespaced as `{server_name}__{tool_name}`.
- **`vector_store.py`** — In-memory vector index using FastEmbed (ONNX runtime) with `all-MiniLM-L6-v2`. Encodes tool definitions and does cosine similarity search via normalized dot product.
- **`filters.py`** — Filter Protocols (`ToolFilter`, `CallFilter`, `ServerLoadFilter`), pipeline classes, and the built-in reference filters. The Protocols and request/result dataclasses are re-exported from `extensible_mcp/__init__.py` for third-party imports.
- **`_mcp_compat.py`** — Bridges the MCP SDK's 1.x → 2.x break (HTTP transport tuple, `httpx` vs `httpx2`, camelCase vs snake_case model fields). Anything touching those goes through it; CI runs the suite on both SDK majors.
- **`config.py`** — Loads JSON config (same `mcpServers` format as Claude Desktop). Config resolution order: `--config` flag → `EXTENSIBLE_MCP_CONFIG` env var → platform-specific default paths → `./config.json`.
- **`types.py`** — Shared dataclasses: `ServerConfig`, `ToolRecord` (builds its own `embedding_text` from name + description + params), `SearchResult`, `CallRequest`, `CallFilterResult`, `ServerLoadRequest`, `ServerLoadResult`.

### Testing

Tests use `pytest` with `pytest-asyncio` (auto mode). The mock MCP server in `tests/mock_server.py` provides fake tool definitions for integration tests without real downstream servers.
