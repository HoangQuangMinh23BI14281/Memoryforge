import os
import sys

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from memoryforge.init import init_project
from memoryforge.mcp.tools import (
    active_recall_tool,
    build_context_bundle_tool,
    build_runtime_context_bundle_tool,
    find_contradictions_tool,
    ingest_file_tool,
    recall_conversation_tool,
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
        assert any(tool.name == "rlm_run" for tool in tools.tools)
        assert any(tool.name == "record_contradiction" for tool in tools.tools)
        assert any(tool.name == "find_contradictions" for tool in tools.tools)
        assert any(tool.name == "build_runtime_context_bundle" for tool in tools.tools)

    anyio.run(smoke)


def test_mcp_tools_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    db_path = str(tmp_path / "memory.db")
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text(
        """
class App:
    def get(self, path):
        return path

app = App()

@app.get("/health")
def health():
    result = check()
    return result

def check():
    return True
""",
        encoding="utf-8",
    )

    stored = store_conversation_tool(
        db_path,
        {
            "agent_id": "agent",
            "session_id": "s1",
            "turns": [{"role": "user", "content": "Health route uses check."}],
        },
    )
    assert stored["count"] == 1
    assert recall_conversation_tool(db_path, {"agent_id": "agent", "query": "Health"})["results"]
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
    correction = record_correction_tool(
        db_path,
        {
            "agent_id": "agent",
            "session_id": "s1",
            "corrected_fact": "Health route uses check and returns True.",
        },
    )
    assert correction["metadata"]["kind"] == "correction"
    assert correction["metadata"]["confidence"] == "high"
    contradiction = record_contradiction_tool(
        db_path,
        {
            "agent_id": "agent",
            "session_id": "s1",
            "statement": "Health route is owned by runtime.",
            "conflicting_item_ids": [correction["item_id"]],
        },
    )
    contradictions = find_contradictions_tool(
        db_path,
        {"agent_id": "agent", "query": "Health route", "include_content": True},
    )
    contradiction_ids = {item["item_id"] for item in contradictions["results"]}
    assert contradiction["item_id"] in contradiction_ids
    assert contradictions["diagnostics"]["answer_model_used"] is False
    active = active_recall_tool(db_path, {"agent_id": "agent", "session_id": "s1", "limit": 3})
    assert active["diagnostics"]["query_required"] is False
    assert any(item["metadata"]["kind"] == "correction" for item in active["results"])

    file_ingest = ingest_file_tool(
        db_path,
        {"agent_id": "agent", "path": str(project / "app.py"), "chunk_size": 1000},
    )
    assert file_ingest["long_term_item_ids"]
    assert file_ingest["source_type"] == "rlm_chunk"

    loaded = rlm_load_tool(
        db_path,
        {
            "agent_id": "agent",
            "content": "Alice owns auth.\n\nBob owns billing.\n\nCarol owns search.",
            "name": "oversized-prompt",
            "chunk_size": 1000,
        },
    )
    searched = rlm_search_tool(
        db_path,
        {"agent_id": "agent", "query": "Alice auth", "buffer_id": loaded["buffer_id"]},
    )
    chunk_id = searched["results"][0]["chunk_id"]
    assert "Alice" in rlm_chunk_get_tool(db_path, {"chunk_id": chunk_id})["chunk"]["content"]
    dispatched = rlm_dispatch_tool(
        db_path,
        {"agent_id": "agent", "buffer_id": loaded["buffer_id"], "batch_size": 1},
    )
    recorded = rlm_record_tool(
        db_path,
        {
            "agent_id": "agent",
            "run_id": dispatched["run_id"],
            "chunk_ids": [chunk_id],
            "analysis": "Alice owns auth.",
            "batch_index": 0,
        },
    )
    aggregated = rlm_aggregate_tool(db_path, {"agent_id": "agent", "run_id": dispatched["run_id"]})
    assert recorded["summary_node_id"] in aggregated["child_node_ids"]

    auto_run = rlm_run_tool(
        db_path,
        {
            "agent_id": "agent",
            "content": "Dana owns reliability.\n\nEli owns release notes." * 80,
            "limit": 1,
            "batch_size": 1,
            "chunk_size": 1000,
            "runner": "mock",
        },
    )
    assert auto_run["runner"] == "mock"
    assert auto_run["aggregate"]["summary_node_id"]


def test_mcp_runtime_context_bundle_validates_core_runtime_delivery(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    initialized = init_project(str(project), agent_id="agent", force=True)

    stored = store_conversation_tool(
        initialized["db_path"],
        {
            "agent_id": "agent",
            "session_id": "s1",
            "turns": [{"role": "user", "content": "Project Atlas uses SQLite memory."}],
        },
    )
    bundle = build_runtime_context_bundle_tool(
        initialized["db_path"],
        {
            "agent_id": "agent",
            "session_id": "s1",
            "query": "Project Atlas memory",
            "project_root": str(project),
            "top_k": 3,
        },
    )

    assert stored["count"] == 1
    assert bundle["runtime"]["runtime"] == "codex"
    assert bundle["delivery"] == "mcp:build_context_bundle"
    assert bundle["context_bundle"]["diagnostics"]["bundle_only"] is True
    assert bundle["context_bundle"]["diagnostics"]["answer_model_used"] is False
    assert bundle["context_bundle"]["diagnostics"]["runtime"]["db_path_verified"] is True
