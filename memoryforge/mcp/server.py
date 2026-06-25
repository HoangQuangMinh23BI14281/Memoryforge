"""Minimal MCP server for MemoryForge."""

from __future__ import annotations

import json
import os
from typing import Any

import mcp.types as types
from mcp.server import Server

from memoryforge.mcp.tools import (
    active_recall_tool,
    build_context_bundle_tool,
    build_runtime_context_bundle_tool,
    find_contradictions_tool,
    ingest_file_tool,
    recall_conversation_tool,
    recall_memory_tool,
    record_contradiction_tool,
    record_correction_tool,
    rlm_aggregate_tool,
    rlm_chunk_get_tool,
    rlm_dispatch_tool,
    rlm_load_tool,
    rlm_record_tool,
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
                name="store_conversation",
                description="Store conversation turns in MemoryForge",
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
                name="recall",
                description="Search conversation memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["agent_id", "query"],
                },
            ),
            types.Tool(
                name="recall_conversation",
                description="Search stored conversation turns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["agent_id", "query"],
                },
            ),
            types.Tool(
                name="recall_memory",
                description="Recall long-term memory through BM25 and vector indexes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                        "session_id": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "include_content": {"type": "boolean", "default": False},
                    },
                    "required": ["agent_id", "query"],
                },
            ),
            types.Tool(
                name="record_correction",
                description="Record a user correction as high-confidence long-term memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "corrected_fact": {"type": "string"},
                        "wrong_item_id": {"type": "string"},
                        "wrong_raw_ref": {"type": "string"},
                        "session_id": {"type": "string"},
                        "source": {"type": "string", "default": "user"},
                    },
                    "required": ["agent_id", "corrected_fact"],
                },
            ),
            types.Tool(
                name="record_contradiction",
                description="Record a contested memory relation without choosing a winner",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "statement": {"type": "string"},
                        "conflicting_item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "conflicting_raw_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "session_id": {"type": "string"},
                        "source": {"type": "string", "default": "user"},
                    },
                    "required": ["agent_id", "statement"],
                },
            ),
            types.Tool(
                name="find_contradictions",
                description="List memories marked as conflicting through contradiction metadata",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                        "include_content": {"type": "boolean", "default": False},
                    },
                    "required": ["agent_id"],
                },
            ),
            types.Tool(
                name="build_context_bundle",
                description="Build a MemoryForge context bundle for the active core model without answering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "query": {"type": "string"},
                        "system_prompt": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                        "include_content": {"type": "boolean", "default": False},
                        "recall_content_policy": {
                            "type": "string",
                            "enum": ["snippet", "champion", "full", "auto", "preview"],
                            "default": "snippet",
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
                name="build_runtime_context_bundle",
                description=(
                    "Validate active runtime delivery and build a MemoryForge context bundle "
                    "for the core model without answering"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "query": {"type": "string"},
                        "project_root": {"type": "string"},
                        "runtime": {"type": "string", "enum": ["auto", "codex"], "default": "auto"},
                        "system_prompt": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                        "include_content": {"type": "boolean", "default": False},
                        "recall_content_policy": {
                            "type": "string",
                            "enum": ["snippet", "champion", "full", "auto", "preview"],
                            "default": "snippet",
                        },
                        "long_term_token_budget": {"type": "integer"},
                        "model_context_limit": {"type": "integer"},
                        "reserved_output_tokens": {"type": "integer"},
                        "compaction_buffer": {"type": "integer"},
                        "soft_threshold_fraction": {"type": "number"},
                    },
                    "required": ["agent_id", "session_id", "query", "project_root"],
                },
            ),
            types.Tool(
                name="active_recall",
                description="Proactively surface recent durable evidence for the active core model without answering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "focus": {"type": "string"},
                        "project_root": {"type": "string"},
                        "limit": {"type": "integer", "default": 8},
                        "include_content": {"type": "boolean", "default": False},
                    },
                    "required": ["agent_id"],
                },
            ),
            types.Tool(
                name="ingest_file",
                description="Ingest a file into immutable file chunks and long-term memory without adding it to LCM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "path": {"type": "string"},
                        "name": {"type": "string"},
                        "chunk_size": {"type": "integer", "default": 12000},
                        "overlap": {"type": "integer", "default": 1000},
                    },
                    "required": ["agent_id", "path"],
                },
            ),
            types.Tool(
                name="rlm_load",
                description="Load oversized content or a file into RLM buffers/chunks and LTM without adding it to LCM",
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
                description="Search RLM chunks; returns chunk IDs and previews only",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                        "buffer_id": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
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
                name="rlm_dispatch",
                description="Create pass-by-reference chunk batches for external sub-agents",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "buffer_id": {"type": "string"},
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                        "batch_size": {"type": "integer"},
                    },
                    "required": ["agent_id"],
                },
            ),
            types.Tool(
                name="rlm_record",
                description="Record a sub-agent finding as an LCM SummaryDAG leaf",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "run_id": {"type": "string"},
                        "chunk_ids": {"type": "array", "items": {"type": "string"}},
                        "analysis": {"type": "string"},
                        "batch_index": {"type": "integer"},
                    },
                    "required": ["agent_id", "run_id", "chunk_ids", "analysis"],
                },
            ),
            types.Tool(
                name="rlm_aggregate",
                description="Aggregate recorded RLM findings into an LCM SummaryDAG parent",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "run_id": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["agent_id", "run_id"],
                },
            ),
            types.Tool(
                name="rlm_run",
                description="Run RLM with spawned sub-agents and store results in LCM",
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
        db_path = os.environ.get("MEMORYFORGE_DB", "~/.memoryforge/memory.db")
        if name == "store_conversation":
            result = store_conversation_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name in {"recall", "recall_conversation"}:
            result = recall_conversation_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "recall_memory":
            result = recall_memory_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "record_correction":
            result = record_correction_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "record_contradiction":
            result = record_contradiction_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "find_contradictions":
            result = find_contradictions_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "build_context_bundle":
            result = build_context_bundle_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "build_runtime_context_bundle":
            result = build_runtime_context_bundle_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "active_recall":
            result = active_recall_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "ingest_file":
            result = ingest_file_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "rlm_load":
            result = rlm_load_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "rlm_search":
            result = rlm_search_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "rlm_chunk_get":
            result = rlm_chunk_get_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "rlm_dispatch":
            result = rlm_dispatch_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "rlm_record":
            result = rlm_record_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "rlm_aggregate":
            result = rlm_aggregate_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        if name == "rlm_run":
            result = rlm_run_tool(db_path, arguments)
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        raise ValueError(f"Unknown tool: {name}")


def run_server() -> None:
    if Server is None:
        raise RuntimeError("mcp is not installed; install memoryforge with MCP dependencies")
    import asyncio

    async def run_stdio() -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(run_stdio())
