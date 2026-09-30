# Copyright (c) 2026 Matthew Fuchs
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan
import mcp.types as mcp_types

from ._mcp_compat import result_is_error
from .client_manager import ClientManager, TokenExpiredError
from .config import Config, find_config_path, load_config
from .filters import (
    AccessControlFilter,
    CallFilter,
    CallFilterPipeline,
    DiscoveredToolsFilter,
    FilterPipeline,
    RegoPolicyFilter,
    ServerLoadAccessControlFilter,
    ServerLoadFilter,
    ServerLoadFilterPipeline,
    SimilarityThresholdFilter,
    ToolFilter,
)
from .types import CallRequest, ServerLoadRequest
from .vector_store import VectorStore

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extensible MCP Proxy")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    return parser.parse_args()


def _build_pipelines(
    config: Config,
    *,
    extra_search_filters: Sequence[ToolFilter] = (),
    extra_call_filters: Sequence[CallFilter] = (),
    extra_load_filters: Sequence[ServerLoadFilter] = (),
) -> tuple[FilterPipeline, CallFilterPipeline, ServerLoadFilterPipeline, DiscoveredToolsFilter]:
    ac = AccessControlFilter(
        deny=config.filters.access_control.deny,
        deny_patterns=config.filters.access_control.deny_patterns,
        allow_servers=config.filters.access_control.allow_servers or None,
    )

    search_pipeline = FilterPipeline([
        SimilarityThresholdFilter(config.filters.similarity_threshold),
        ac,
    ])
    for f in extra_search_filters:
        search_pipeline.add(f)

    discovered_filter = DiscoveredToolsFilter()
    call_pipeline = CallFilterPipeline([ac, discovered_filter])

    if config.filters.rego_policy:
        call_pipeline.add(RegoPolicyFilter(config.filters.rego_policy))

    for f in extra_call_filters:
        call_pipeline.add(f)

    lc = config.filters.load_control
    server_load_pipeline = ServerLoadFilterPipeline()
    if lc.deny_names or lc.deny_name_patterns or lc.deny_url_patterns or lc.allow_url_patterns:
        server_load_pipeline.add(ServerLoadAccessControlFilter(
            deny_names=lc.deny_names,
            deny_name_patterns=lc.deny_name_patterns,
            deny_url_patterns=lc.deny_url_patterns,
            allow_url_patterns=lc.allow_url_patterns,
        ))

    for f in extra_load_filters:
        server_load_pipeline.add(f)

    return search_pipeline, call_pipeline, server_load_pipeline, discovered_filter


def _format_search_results(results: list[Any]) -> str:
    if not results:
        return "No matching tools found. Try a different search query."
    lines: list[str] = []
    for r in results:
        tool = r.tool
        lines.append(f"## {tool.qualified_name}")
        lines.append(f"**Description:** {tool.description}")
        lines.append(f"**Parameters:**")
        lines.append(f"```json\n{json.dumps(tool.input_schema, indent=2)}\n```")
        lines.append(f"**Similarity:** {r.score:.3f}")
        lines.append("")
    header = (
        f"Found {len(results)} matching tool(s). "
        "Use `call_tool` with the tool name and arguments to invoke one.\n\n"
    )
    return header + "\n".join(lines)


async def _setup(
    config: Config,
    *,
    extra_search_filters: Sequence[ToolFilter] = (),
    extra_call_filters: Sequence[CallFilter] = (),
    extra_load_filters: Sequence[ServerLoadFilter] = (),
) -> dict[str, Any]:
    """Connect to downstream servers, build vector index, return lifespan context."""
    search_pipeline, call_pipeline, server_load_pipeline, discovered_filter = _build_pipelines(
        config,
        extra_search_filters=extra_search_filters,
        extra_call_filters=extra_call_filters,
        extra_load_filters=extra_load_filters,
    )

    client_mgr = ClientManager(tokens_file=config.tokens_file)
    logger.info(
        "Tokens file: %s",
        config.tokens_file if config.tokens_file else "(none configured)",
    )
    logger.info("Connecting to %d downstream server(s)...", len(config.servers))
    tools = await client_mgr.connect_all(config.servers)
    logger.info("Indexed %d tools total", len(tools))

    vector_store = VectorStore()
    logger.info("Building vector index...")
    vector_store.index(tools)
    logger.info("Vector index ready")

    return {
        "vector_store": vector_store,
        "filter_pipeline": search_pipeline,
        "call_filter_pipeline": call_pipeline,
        "server_load_filter_pipeline": server_load_pipeline,
        "client_manager": client_mgr,
        "discovered_filter": discovered_filter,
    }


def create_server(
    config: Config,
    *,
    extra_search_filters: Sequence[ToolFilter] = (),
    extra_call_filters: Sequence[CallFilter] = (),
    extra_load_filters: Sequence[ServerLoadFilter] = (),
) -> FastMCP:
    """Create a configured FastMCP proxy server for the given config.

    The ``extra_*_filters`` arguments accept user-defined filters (anything
    matching the ``ToolFilter``, ``CallFilter``, or ``ServerLoadFilter``
    Protocol). They are appended to the corresponding pipeline after the
    built-in reference filters; the discovered-tools guarantee is always
    enforced regardless.
    """

    @lifespan
    async def configured_lifespan(server: FastMCP):
        ctx = await _setup(
            config,
            extra_search_filters=extra_search_filters,
            extra_call_filters=extra_call_filters,
            extra_load_filters=extra_load_filters,
        )
        yield ctx
        logger.info("Shutting down, closing connections...")
        await ctx["client_manager"].close_all()

    server = FastMCP(
        name="extensible-mcp",
        instructions=(
            "This server is a proxy to other MCP tool servers. "
            "You do not have direct tool definitions — discover them on demand.\n\n"
            "Workflow:\n"
            "1. Use `search_tools` with a natural language query to find relevant tools.\n"
            "2. Use `call_tool` with the qualified name from search results "
            "(e.g. 'github__create_issue') and arguments matching the returned schema.\n"
            "3. You may only call tools you have previously discovered via search_tools.\n\n"
            "Dynamic servers:\n"
            "- Use `load_mcp_server` to connect to a new remote MCP server by URL.\n"
            "- If the server requires authentication, ask the user to add a token "
            "for that server name to the tokens file before loading.\n\n"
            "Authentication errors:\n"
            "- If a call_tool fails with an authentication/token error, ask the user "
            "to update the token in the tokens file, then retry the same call.\n"
            "- Never ask the user to provide tokens in the conversation. "
            "Tokens are managed through the tokens file only."
        ),
        lifespan=configured_lifespan,
    )

    @server.tool(
        name="search_tools",
        description=(
            "Search for available tools by describing what you want to do. "
            "Returns matching tool definitions with their names, descriptions, "
            "and parameter schemas. Use the tool name from results with `call_tool` to invoke."
        ),
    )
    async def search_tools_handler(query: str, ctx: Context, top_k: int = 5) -> str:
        vs: VectorStore = ctx.lifespan_context["vector_store"]
        pipeline: FilterPipeline = ctx.lifespan_context["filter_pipeline"]
        discovered_filter: DiscoveredToolsFilter = ctx.lifespan_context["discovered_filter"]
        results = vs.search(query, top_k=top_k)
        results = pipeline.apply(results, query)
        discovered_filter.register([r.tool.qualified_name for r in results])
        return _format_search_results(results)

    @server.tool(
        name="call_tool",
        description=(
            "Call a tool on a downstream MCP server. "
            "Use the qualified tool name (e.g. 'github__create_issue') from search_tools results. "
            "Pass the arguments as a JSON object matching the tool's parameter schema."
        ),
    )
    async def call_tool_handler(
        tool_name: str, arguments: dict[str, Any], ctx: Context
    ) -> str:
        call_pipeline: CallFilterPipeline = ctx.lifespan_context["call_filter_pipeline"]
        client_mgr: ClientManager = ctx.lifespan_context["client_manager"]

        parts = tool_name.split("__", 1)
        server_name = parts[0] if len(parts) == 2 else ""

        request = CallRequest(
            tool_name=tool_name, arguments=arguments, server_name=server_name
        )
        filter_result = await call_pipeline.apply(request)
        if not filter_result.allowed:
            return f"Error: {filter_result.reason}"

        tool_name = filter_result.tool_name
        arguments = filter_result.arguments

        if tool_name not in client_mgr.get_qualified_names():
            return f"Error: Unknown tool '{tool_name}'. Use search_tools to find available tools."

        try:
            result: mcp_types.CallToolResult = await client_mgr.call_tool(
                tool_name, arguments
            )
        except TokenExpiredError as e:
            return (
                f"Error: Authentication failed for server '{e.server_name}' "
                f"(token unchanged for {e.token_age_minutes} minutes). "
                f"The token may have expired. Ask the user to update the token "
                f"for '{e.server_name}' in the tokens file, then retry the call."
            )
        except Exception as e:
            return f"Error calling tool '{tool_name}': {e}"

        if result_is_error(result):
            texts = [
                block.text
                for block in result.content
                if isinstance(block, mcp_types.TextContent)
            ]
            return f"Tool error: {' '.join(texts)}"

        output_parts: list[str] = []
        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                output_parts.append(block.text)
            else:
                output_parts.append(f"[{type(block).__name__}]")
        return "\n".join(output_parts) if output_parts else "(no output)"

    @server.tool(
        name="load_mcp_server",
        description=(
            "Dynamically connect to a new remote MCP server by URL. "
            "Indexes all of the server's tools and makes them available for "
            "search_tools and call_tool. The server_name is used as a namespace "
            "prefix for tool names (e.g. 'myserver__tool_name'). "
            "If the server requires authentication, the user must add the token "
            "to the tokens file before calling this."
        ),
    )
    async def load_mcp_server_handler(
        server_name: str, url: str, ctx: Context
    ) -> str:
        server_load_pipeline: ServerLoadFilterPipeline = ctx.lifespan_context[
            "server_load_filter_pipeline"
        ]
        vs: VectorStore = ctx.lifespan_context["vector_store"]
        client_mgr: ClientManager = ctx.lifespan_context["client_manager"]

        load_request = ServerLoadRequest(server_name=server_name, url=url)
        load_result = await server_load_pipeline.apply(load_request)
        if not load_result.allowed:
            return f"Error: {load_result.reason}"

        if server_name in client_mgr._connections:
            return f"Error: Server '{server_name}' is already connected."

        try:
            tools = await client_mgr.connect_url(server_name, url)
        except Exception as e:
            return f"Error connecting to '{url}': {e}"

        logger.info("Indexing %d tools from '%s'...", len(tools), server_name)
        vs.add(tools)
        logger.info("Indexing complete for '%s'", server_name)
        return (
            f"Successfully connected to '{server_name}' at {url}. "
            f"Indexed {len(tools)} tool(s). They are now available via search_tools and call_tool."
        )

    return server


def main() -> None:
    args = _parse_args()
    config_path = find_config_path(args.config)
    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)
    server = create_server(config)
    server.run()
