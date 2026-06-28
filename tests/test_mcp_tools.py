import os
import sys

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

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
        assert any(tool.name == "ensure_project_memory" for tool in tools.tools)
        assert any(tool.name == "autoload_markdown" for tool in tools.tools)
        assert any(tool.name == "rlm_run" for tool in tools.tools)
        tool_names = {tool.name for tool in tools.tools}
        assert "recall" not in tool_names
        assert "record_contradiction" not in tool_names
        assert "rlm_dispatch" not in tool_names
        assert "build_runtime_context_bundle" not in tool_names

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

    ensured = ensure_project_memory_tool(
        db_path,
        {"agent_id": "agent", "project_root": str(project), "auto_index": True},
    )
    autoloaded = autoload_markdown_tool(
        db_path,
        {"agent_id": "agent", "project_root": str(project)},
    )
    stored = store_conversation_tool(
        db_path,
        {
            "agent_id": "agent",
            "session_id": "s1",
            "turns": [{"role": "user", "content": "Health route uses check."}],
        },
    )
    assert ensured["project_root"] == str(project.resolve())
    assert ensured["hooks_enabled"] is False
    assert autoloaded["enabled"] is True
    assert stored["count"] == 1
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

def test_recall_memory_tool_truncates_large_content_for_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    db_path = str(tmp_path / "memory.db")
    project = tmp_path / "project"
    project.mkdir()
    long_text = ("collector endpoint 4318 replaced stale 4317. " * 500)
    (project / "architecture.md").write_text(long_text, encoding="utf-8")

    ensured = ensure_project_memory_tool(
        db_path,
        {"agent_id": "codex", "project_root": str(project), "auto_index": True},
    )
    recalled = recall_memory_tool(
        ensured["db_path"],
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
