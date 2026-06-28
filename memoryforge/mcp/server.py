"""Minimal MCP server for MemoryForge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server

from memoryforge.mcp.tools import (
    autoload_markdown_tool,
    build_context_bundle_tool,
    ensure_project_memory_tool,
    recall_memory_tool,
    rlm_chunk_get_tool,
    rlm_load_tool,
    rlm_run_tool,
    rlm_search_tool,
    store_conversation_tool,
)

if Server is not None:
    app = Server("memoryforge")

    @app.list_tools()
    async def handle_list_tools() -> list[Any]:
        return [
            types.Tool(
                name="ensure_project_memory",
                description="Ensure lightweight .memoryforge state exists for the current project. Does not index unless auto_index is explicitly true.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "project_root": {"type": "string"},
                        "auto_index": {"type": "boolean", "default": False},
                    },
                },
            ),
            types.Tool(
                name="autoload_markdown",
                description="Index changed Markdown files into RLM/LTM using the project autoload manifest",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "project_root": {"type": "string"},
                        "chunk_size": {"type": "integer", "default": 12000},
                        "overlap": {"type": "integer", "default": 1000},
                        "max_files": {"type": "integer", "default": 200},
                        "max_file_bytes": {"type": "integer", "default": 1000000},
                    },
                },
            ),
            types.Tool(
                name="recall_memory",
                description="Recall long-term project memory through BM25 and vector indexes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                        "session_id": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8},
                        "include_content": {"type": "boolean", "default": True},
                    },
                    "required": ["agent_id", "query"],
                },
            ),
            types.Tool(
                name="build_context_bundle",
                description="Build grounded LCM/LTM context for the core model without answering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "query": {"type": "string"},
                        "system_prompt": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                        "include_content": {"type": "boolean", "default": True},
                        "recall_content_policy": {
                            "type": "string",
                            "enum": ["snippet", "champion", "full", "auto", "preview"],
                            "default": "champion",
                        },
                        "long_term_token_budget": {"type": "integer"},
                        "model_context_limit": {"type": "integer"},
                        "reserved_output_tokens": {"type": "integer"},
                        "compaction_buffer": {"type": "integer"},
                        "soft_threshold_fraction": {"type": "number"},
                    },
                    "required": ["agent_id", "session_id", "query"],
                },
            ),
            types.Tool(
                name="store_conversation",
                description="Store conversation turns in MemoryForge LCM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "turns": {"type": "array"},
                    },
                    "required": ["agent_id", "turns"],
                },
            ),
            types.Tool(
                name="rlm_load",
                description="Load oversized content or a file into RLM chunks and LTM without sub-agent analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "name": {"type": "string"},
                        "chunk_size": {"type": "integer", "default": 12000},
                        "overlap": {"type": "integer", "default": 1000},
                    },
                    "required": ["agent_id"],
                },
            ),
            types.Tool(
                name="rlm_search",
                description="Search RLM chunks and return chunk IDs/previews",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                        "buffer_id": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8},
                        "mode": {"type": "string", "default": "hybrid"},
                    },
                    "required": ["agent_id", "query"],
                },
            ),
            types.Tool(
                name="rlm_chunk_get",
                description="Fetch full RLM chunk content by chunk ID",
                inputSchema={
                    "type": "object",
                    "properties": {"chunk_id": {"type": "string"}},
                    "required": ["chunk_id"],
                },
            ),
            types.Tool(
                name="rlm_run",
                description="Run RLM with real sub-agents for explicit deep analysis over large context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "name": {"type": "string"},
                        "buffer_id": {"type": "string"},
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                        "batch_size": {"type": "integer"},
                        "chunk_size": {"type": "integer", "default": 12000},
                        "overlap": {"type": "integer", "default": 1000},
                        "runner": {"type": "string", "default": "auto"},
                        "model": {"type": "string"},
                        "base_url": {"type": "string"},
                        "project_root": {"type": "string"},
                        "timeout": {"type": "number", "default": 900},
                        "max_workers": {"type": "integer", "default": 1},
                        "max_retries": {"type": "integer", "default": 0},
                        "allow_partial": {"type": "boolean", "default": False},
                        "synthesize": {"type": "boolean", "default": True},
                        "recursive": {"type": "boolean", "default": True},
                        "max_recursive_rounds": {"type": "integer", "default": 2},
                        "recursive_token_limit": {"type": "integer"},
                    },
                    "required": ["agent_id"],
                },
            ),
        ]
    @app.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        db_path = _resolve_mcp_db_path()
        handlers = {
            "ensure_project_memory": ensure_project_memory_tool,
            "autoload_markdown": autoload_markdown_tool,
            "recall_memory": recall_memory_tool,
            "build_context_bundle": build_context_bundle_tool,
            "store_conversation": store_conversation_tool,
            "rlm_load": rlm_load_tool,
            "rlm_search": rlm_search_tool,
            "rlm_chunk_get": rlm_chunk_get_tool,
            "rlm_run": rlm_run_tool,
        }
        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        result = handler(db_path, arguments)
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


def _resolve_mcp_db_path() -> str:
    configured = os.environ.get("MEMORYFORGE_DB")
    if configured:
        return configured
    root = Path(os.getcwd()).resolve()
    db_path = root / ".memoryforge" / "memory.db"
    if os.environ.get("MEMORYFORGE_AUTO_INIT", "1").lower() not in {"0", "false", "no"}:
        from memoryforge.init import ensure_project_initialized

        ensure_project_initialized(
            str(root),
            agent_id=os.environ.get("MEMORYFORGE_AGENT_ID", "codex"),
            configure_codex=False,
            auto_index=False,
        )
    return str(db_path)

def run_server() -> None:
    if Server is None:
        raise RuntimeError("mcp is not installed; install memoryforge with MCP dependencies")
    import asyncio

    async def run_stdio() -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(run_stdio())



