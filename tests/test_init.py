import json
import sqlite3
from pathlib import Path

import pytest

from memoryforge import MemoryForge
from memoryforge.init import handle_hook_event, init_project
from memoryforge.init.bootstrap import _codex_hook_group
from memoryforge.runtime import RuntimeIntegrationError, resolve_runtime_integration


def _isolate_home(monkeypatch, home):
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def test_init_writes_agent_configs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _isolate_home(monkeypatch, home)
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (project / "README.md").write_text("Long project notes", encoding="utf-8")
    src = project / "src"
    src.mkdir()
    (src / "main.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")

    result = init_project(str(project), agent_id="agent", force=True)

    assert result["indexed"]["enabled"] is False
    assert result["indexed"]["files"] == 0
    assert result["indexed"]["chunks"] == 0
    assert result["indexed"]["long_term_items"] == 0
    assert result["hooks_enabled"] is False
    assert not (project / ".codex").exists()
    assert not (project / ".memoryforge" / "hooks").exists()
    assert (project / ".memoryforge" / "memory.db").exists()
    memoryforge_config = json.loads(
        (project / ".memoryforge" / "config.json").read_text(encoding="utf-8")
    )
    assert memoryforge_config["auto_index"] is False
    assert memoryforge_config["hooks_enabled"] is False
    assert memoryforge_config["capture"]["enabled"] is False
    assert memoryforge_config["subagent"]["runner"] == "codex"
    assert memoryforge_config["subagent"]["model"] == "gpt-5.4"
    assert memoryforge_config["codex_mcp"]["ok"] is False
    assert memoryforge_config["codex_mcp"]["skipped"] == "codex MCP registration not requested"
    assert ".codex" not in "\n".join(result["written"])
    assert str(project / "AGENTS.md") in result["written"]
    assert result["codex_init"]["ok"] is True


def test_init_does_not_touch_existing_codex_dir(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _isolate_home(monkeypatch, home)
    codex_dir = project / ".codex"
    codex_dir.mkdir()
    marker = codex_dir / "config.toml"
    marker.write_text("user-owned = true\n", encoding="utf-8")

    init_project(str(project), agent_id="agent", force=True)

    assert marker.read_text(encoding="utf-8") == "user-owned = true\n"
    assert (project / "AGENTS.md").exists()


def test_init_writes_root_agents_instructions(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _isolate_home(monkeypatch, home)
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")

    result = init_project(str(project), agent_id="agent", force=True)
    agents_text = (project / "AGENTS.md").read_text(encoding="utf-8")

    assert "MemoryForge Project Memory" in agents_text
    assert "recall_memory" in agents_text
    assert "index_analyze" in agents_text
    assert "lcm-sessions" in agents_text
    assert "build_context_bundle" in agents_text
    assert "memoryforge index" in agents_text
    assert "record_turn" not in agents_text
    assert "autoload_markdown" not in agents_text
    assert "rlm_run" not in agents_text
    assert str(project / "AGENTS.md") in result["written"]


def test_init_install_hooks_is_explicit_and_writes_lightweight_codex_hooks(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _isolate_home(monkeypatch, home)
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")

    result = init_project(
        str(project),
        agent_id="agent",
        auto_index=False,
        install_hooks=True,
        force=True,
    )
    hooks_path = project / ".codex" / "hooks.json"
    hooks_config = json.loads(hooks_path.read_text(encoding="utf-8"))

    assert result["hooks_enabled"] is True
    assert result["hooks"]["ok"] is True
    assert result["hooks"]["model_calls"] is False
    assert result["hooks"]["codex_account_required"] is False
    assert result["capture"]["enabled"] is True
    assert result["capture"]["model_calls"] is False
    assert result["capture"]["codex_account_required"] is False
    assert hooks_path.exists()
    assert (project / ".memoryforge" / "hooks").exists()
    assert set(result["hooks"]["events"]) == {
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
        "Stop",
    }
    assert "MemoryForge UserPromptSubmit" in json.dumps(hooks_config)
    assert "memoryforge-hook" in json.dumps(hooks_config)
    hook_runner_text = (project / ".memoryforge" / "hooks" / "memoryforge-hook.sh").read_text(
        encoding="utf-8"
    )
    assert "uv run --no-sync python -m memoryforge.cli.main hook" in hook_runner_text
    assert "MEMORYFORGE_HOOK_LOG" in hook_runner_text
    assert "--runtime linux-wsl" in hook_runner_text


def test_codex_hook_cmd_runner_uses_windows_quoting_even_from_posix() -> None:
    group = _codex_hook_group(
        "SessionStart",
        root=Path("C:/Users/ADMIN/OneDrive/Desktop/New folder/testv5"),
        db_path=Path("C:/Users/ADMIN/OneDrive/Desktop/New folder/testv5/.memoryforge/memory.db"),
        agent_id="codex",
        hook_runner=Path(
            "C:/Users/ADMIN/OneDrive/Desktop/New folder/testv5/.memoryforge/hooks/memoryforge-hook.cmd"
        ),
    )

    command = group["hooks"][0]["command"]
    command_windows = group["hooks"][0]["commandWindows"]

    assert command == command_windows
    assert command.startswith('"C:/Users/ADMIN/OneDrive/Desktop/New folder/testv5/')
    assert "'C:" not in command


def test_hook_ingests_prompt_and_referenced_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (project / "notes.md").write_text(
        "MemoryForge stores long markdown losslessly", encoding="utf-8"
    )
    result = init_project(str(project), agent_id="agent", auto_index=False, force=True)

    hook_result = handle_hook_event(
        "user-prompt-submit",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"prompt": "Please read notes.md", "session_id": "s1"}),
    )
    stop_result = handle_hook_event(
        "stop",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"session_id": "s1"}),
    )
    mf = MemoryForge(result["db_path"])
    try:
        hits = mf.search("agent", "long markdown", top_k=5)
        long_term_hits = mf.recall_long_term(
            "agent", "long markdown", top_k=5, include_content=True
        )
    finally:
        mf.close()

    assert hook_result["pending"]["count"] == 1
    assert stop_result["committed"]["committed"] == 1
    assert len(stop_result["committed"]["turn_ids"]) == 1
    assert not hits
    assert any(hit["source_type"] == "rlm_chunk" for hit in long_term_hits)


def test_hook_captures_completed_codex_turn_with_tool_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    project = tmp_path / "project"
    project.mkdir()
    result = init_project(str(project), agent_id="agent", auto_index=False, force=True)

    prompt_result = handle_hook_event(
        "UserPromptSubmit",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"prompt": "Run the build check", "session_id": "s1"}),
    )
    tool_result = handle_hook_event(
        "PostToolUse",
        result["db_path"],
        "agent",
        str(project),
        json.dumps(
            {
                "session_id": "s1",
                "tool_name": "Bash",
                "tool_call_id": "tool-1",
                "output": "build check passed",
            }
        ),
    )
    stop_result = handle_hook_event(
        "Stop",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"session_id": "s1", "assistant_response": "Build check passed."}),
    )
    mf = MemoryForge(result["db_path"])
    try:
        messages = mf.lcm_store.get_messages("s1", include_summaries=False)
        context = mf.lcm_build_context("s1")
        recall = mf.recall_long_term("agent", "build check passed", top_k=5)
    finally:
        mf.close()

    assert prompt_result["pending"]["count"] == 1
    assert tool_result["pending"]["turn_count"] == 2
    assert stop_result["committed"]["committed"] == 3
    assert [message.role for message in messages] == ["user", "assistant", "assistant"]
    assert messages[1].parts[0].part_type == "tool"
    assert messages[1].parts[0].tool_name == "Bash"
    assert messages[1].parts[0].content == "build check passed"
    assert context.token_estimate > 0
    assert any(hit["source_type"] == "message" for hit in recall)


def test_hook_keeps_prompt_pending_until_stop_commits(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    result = init_project(str(project), agent_id="agent", auto_index=False, force=True)

    hook_result = handle_hook_event(
        "user-prompt-submit",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"prompt": "Pending prompt should wait for stop", "session_id": "s1"}),
    )
    mf = MemoryForge(result["db_path"])
    try:
        long_term_hits = mf.recall_long_term(
            "agent", "Pending prompt wait stop", top_k=5, include_content=True
        )
    finally:
        mf.close()

    assert hook_result["pending"]["count"] == 1
    assert long_term_hits == []


def test_hook_discards_pending_prompt_on_explicit_retract_event(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    result = init_project(str(project), agent_id="agent", auto_index=False, force=True)

    hook_result = handle_hook_event(
        "user-prompt-submit",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"prompt": "Retracted prompt should not become durable", "session_id": "s1"}),
    )
    retract_result = handle_hook_event(
        "discard-pending",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"session_id": "s1"}),
    )
    stop_result = handle_hook_event(
        "stop",
        result["db_path"],
        "agent",
        str(project),
        json.dumps({"session_id": "s1"}),
    )
    mf = MemoryForge(result["db_path"])
    try:
        long_term_hits = mf.recall_long_term(
            "agent", "Retracted prompt durable", top_k=5, include_content=True
        )
    finally:
        mf.close()

    assert hook_result["pending"]["count"] == 1
    assert retract_result["pending_discarded"]["files"] == 1
    assert stop_result["committed"]["committed"] == 0
    assert long_term_hits == []


def test_session_start_hook_skips_markdown_indexing_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    result = init_project(str(project), agent_id="agent", auto_index=False, force=True)
    (project / "notes.md").write_text("Atlas session-start indexing amber-17", encoding="utf-8")

    hook_result = handle_hook_event(
        "session-start",
        result["db_path"],
        "agent",
        str(project),
        "{}",
    )
    mf = MemoryForge(result["db_path"])
    try:
        hits = mf.recall_long_term("agent", "session-start amber-17", top_k=3)
    finally:
        mf.close()

    assert hook_result["indexed"]["enabled"] is False
    assert hook_result["indexed"]["files"] == 0
    assert hook_result["indexed"]["skipped"] == "hook session-start does not auto-index by default"
    assert hits == []


def test_session_start_hook_indexes_markdown_only_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")
    monkeypatch.setenv("MEMORYFORGE_HOOK_AUTO_INDEX", "1")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    result = init_project(str(project), agent_id="agent", auto_index=False, force=True)
    (project / "notes.md").write_text("Atlas session-start indexing amber-17", encoding="utf-8")

    hook_result = handle_hook_event(
        "session-start",
        result["db_path"],
        "agent",
        str(project),
        "{}",
    )
    mf = MemoryForge(result["db_path"])
    try:
        hits = mf.recall_long_term("agent", "session-start amber-17", top_k=3)
    finally:
        mf.close()

    assert hook_result["indexed"]["enabled"] is True
    assert hook_result["indexed"]["files"] == 1
    assert any(hit["source_type"] == "rlm_chunk" for hit in hits)


def test_autoload_manifest_survives_session_start_without_auto_index(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("Autoload manifest remembers amber cache", encoding="utf-8")
    result = init_project(str(project), agent_id="agent", auto_index=True, force=True)

    hook_result = handle_hook_event(
        "session-start",
        result["db_path"],
        "agent",
        str(project),
        "{}",
    )
    conn = sqlite3.connect(result["db_path"])
    try:
        row = conn.execute(
            "SELECT path, buffer_id FROM autoload_files WHERE path = ?",
            ("README.md",),
        ).fetchone()
    finally:
        conn.close()

    assert result["indexed"]["files"] == 1
    assert hook_result["indexed"]["enabled"] is False
    assert hook_result["indexed"]["files"] == 0
    assert row is not None
    assert row[0] == "README.md"


def test_init_accepts_non_python_project(tmp_path, monkeypatch):
    project = tmp_path / "plain-project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _isolate_home(monkeypatch, home)
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")

    result = init_project(str(project), agent_id="agent", force=True)

    assert (project / ".memoryforge" / "memory.db").exists()
    assert result["project_root"] == str(project.resolve())


def test_runtime_context_requires_registered_mcp(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    with pytest.raises(RuntimeIntegrationError, match="Could not identify active Codex MemoryForge instructions"):
        resolve_runtime_integration(project)

    initialized = init_project(str(project), agent_id="agent", configure_codex=True, force=True)
    runtime = resolve_runtime_integration(project, runtime="codex")

    assert initialized["codex_mcp"]["ok"] is True
    assert runtime.mcp_configured is True
    assert runtime.config_path == str(project / "AGENTS.md")

def test_init_defaults_to_codex_agent(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _isolate_home(monkeypatch, home)
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")

    result = init_project(str(project), force=True)
    config = json.loads((project / ".memoryforge" / "config.json").read_text(encoding="utf-8"))

    assert result["db_path"] == config["db_path"]
    assert config["agent_id"] == "codex"


def test_init_reindexes_unchanged_markdown_when_agent_changes(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    _isolate_home(monkeypatch, home)
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")
    (project / "README.md").write_text("collector port is 4318", encoding="utf-8")

    first = init_project(str(project), agent_id="default", auto_index=True, force=True)
    second = init_project(str(project), agent_id="codex", auto_index=True, force=True)

    conn = sqlite3.connect(first["db_path"])
    try:
        rows = conn.execute(
            "SELECT agent_id, COUNT(*) FROM long_term_items GROUP BY agent_id ORDER BY agent_id"
        ).fetchall()
    finally:
        conn.close()

    counts = {agent_id: count for agent_id, count in rows}
    assert first["indexed"]["files"] == 1
    assert second["indexed"]["files"] == 1
    assert counts["default"] >= 1
    assert counts["codex"] >= 1
