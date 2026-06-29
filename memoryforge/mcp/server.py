"""Minimal MCP server for MemoryForge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import Server

from memoryforge.mcp.tools import (
    build_context_bundle_tool,
    index_analyze_tool,
    recall_memory_tool,
)

if Server is not None:
    app = Server("memoryforge")

    @app.list_tools()
    async def handle_list_tools() -> list[Any]:
        return [
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
                name="index_analyze",
                description=(
                    "Index project Markdown and return host-subagent RLM analysis plans; "
                    "does not call a model"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "project_root": {"type": "string"},
                        "chunk_size": {"type": "integer", "default": 12000},
                        "overlap": {"type": "integer", "default": 1000},
                        "max_files": {"type": "integer", "default": 200},
                        "max_file_bytes": {"type": "integer", "default": 1000000},
                        "analyze_min_bytes": {"type": "integer", "default": 20000},
                        "analyze_max_files": {"type": "integer", "default": 5},
                        "limit": {"type": "integer", "default": 10000},
                        "batch_size": {"type": "integer"},
                        "force": {"type": "boolean", "default": False},
                    },
                    "required": ["agent_id"],
                },
            ),
        ]

    @app.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        db_path = _resolve_mcp_db_path()
        handlers = {
            "recall_memory": recall_memory_tool,
            "build_context_bundle": build_context_bundle_tool,
            "index_analyze": index_analyze_tool,
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
