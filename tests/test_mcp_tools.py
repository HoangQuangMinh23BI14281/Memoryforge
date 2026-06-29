import os
import sys

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from memoryforge import MemoryForge
from memoryforge.mcp.tools import (
    build_context_bundle_tool,
    index_analyze_tool,
    recall_memory_tool,
)


def test_mcp_server_stdio_handshake(tmp_path):
    async def smoke() -> None:
        env = os.environ.copy()
        env["MEMORYFORGE_DB"] = str(tmp_path / "memory.db")
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        server = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from memoryforge.mcp.server import run_server; run_server()"],
            env=env,
        )
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()

        assert initialized.serverInfo.name == "memoryforge"
        tool_names = {tool.name for tool in tools.tools}
        assert tool_names == {"recall_memory", "build_context_bundle", "index_analyze"}
        assert "ensure_project_memory" not in tool_names
        assert "autoload_markdown" not in tool_names
        assert "rlm_run" not in tool_names
        assert "record_turn" not in tool_names
        assert "record_contradiction" not in tool_names
        assert "rlm_dispatch" not in tool_names
        assert "build_runtime_context_bundle" not in tool_names

    anyio.run(smoke)


def test_public_mcp_tools_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Health route uses check."}],
            session_id="s1",
        )
    finally:
        mf.close()

    recalled = recall_memory_tool(
        db_path,
        {"agent_id": "agent", "query": "Health route", "include_content": True},
    )
    assert recalled["results"]

    bundle = build_context_bundle_tool(
        db_path,
        {
            "agent_id": "agent",
            "session_id": "s1",
            "query": "Health route",
            "top_k": 3,
        },
    )
    assert bundle["diagnostics"]["bundle_only"] is True
    assert bundle["diagnostics"]["answer_model_used"] is False
    assert bundle["messages"][0]["source"] == "active_recall"
    assert any(message["source"] == "long_term" for message in bundle["messages"])


def test_recall_memory_tool_truncates_large_content_for_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    db_path = str(tmp_path / "memory.db")
    long_text = "collector endpoint 4318 replaced stale 4317. " * 500
    mf = MemoryForge(db_path)
    try:
        mf.long_term.index_raw_item("codex", "note", "architecture", long_text)
    finally:
        mf.close()

    recalled = recall_memory_tool(
        db_path,
        {
            "agent_id": "codex",
            "query": "collector endpoint stale 4317 replaced 4318",
            "include_content": True,
            "top_k": 3,
        },
    )

    assert recalled["results"]
    first = recalled["results"][0]
    assert first["content_truncated"] in {True, False}
    assert first["content_chars"] >= len(first["content"])
    assert len(first["content"]) <= 1604


def test_index_analyze_tool_returns_host_subagent_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")
    monkeypatch.setenv("MEMORYFORGE_PROGRESS", "0")
    project = tmp_path / "project"
    project.mkdir()
    (project / "large.md").write_text(
        "Telemetry endpoint 4318 replaced stale 4317.\n\n" * 40,
        encoding="utf-8",
    )

    payload = index_analyze_tool(
        str(project / ".memoryforge" / "memory.db"),
        {
            "agent_id": "codex",
            "project_root": str(project),
            "chunk_size": 120,
            "overlap": 0,
            "max_files": 1,
            "analyze_min_bytes": 0,
            "analyze_max_files": 1,
            "batch_size": 1,
        },
    )

    analyze = payload["analyze"]
    plan = analyze["plans"][0]
    batch = plan["batches"][0]

    assert payload["external_model_calls"] is False
    assert payload["raw"]["enabled"] is True
    assert analyze["mode"] == "host_subagent_plan"
    assert plan["expected_batch_count"] == plan["batch_count"]
    assert "rlm_chunk:" in batch["host_subagent_prompt"]
