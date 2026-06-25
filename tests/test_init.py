import json

import pytest

from memoryforge import MemoryForge
from memoryforge.init import handle_hook_event, init_project


def test_init_writes_agent_configs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (project / "README.md").write_text("Long project notes", encoding="utf-8")
    src = project / "src"
    src.mkdir()
    (src / "main.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    result = init_project(str(project), agent_id="agent", force=True)

    assert result["indexed"]["enabled"] is False
    assert (project / ".codex" / "config.toml").exists()
    assert (project / ".memoryforge" / "hooks" / "memoryforge-hook.sh").exists()
    config_text = (project / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "# --- MemoryForge MCP server ---" in config_text
    assert "[mcp_servers.memoryforge]" in config_text
    assert 'args = ["run", "memoryforge-mcp"]' in config_text
    memoryforge_config = json.loads(
        (project / ".memoryforge" / "config.json").read_text(encoding="utf-8")
    )
    assert memoryforge_config["subagent"]["runner"] == "codex"
    assert memoryforge_config["subagent"]["model"] == "gpt-5.4"
    hooks = json.loads((project / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert set(hooks["hooks"]) == {"SessionStart", "UserPromptSubmit", "PreCompact", "Stop"}
    assert "memoryforge-hook.sh user-prompt-submit" in json.dumps(
        hooks["hooks"]["UserPromptSubmit"]
    )


def test_init_merges_codex_hooks_without_clobbering_user_hooks(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    codex_dir = project / ".codex"
    codex_dir.mkdir()
    (codex_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [{"hooks": [{"type": "command", "command": "echo keep"}]}],
                    "SessionEnd": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": ".memoryforge/hooks/memoryforge-hook.sh session-end",
                                }
                            ]
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    init_project(str(project), agent_id="agent", force=True)
    init_project(str(project), agent_id="agent", force=True)

    hooks = json.loads((codex_dir / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    stop_commands = json.dumps(hooks["Stop"])
    assert "echo keep" in stop_commands
    assert stop_commands.count("memoryforge-hook.sh stop") == 1
    assert "SessionEnd" not in hooks


def test_init_refuses_user_managed_memoryforge_mcp_without_force(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    codex_dir = project / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text(
        "\n".join(
            [
                "[mcp_servers.memoryforge]",
                'command = "custom-memoryforge"',
                'args = ["serve"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not configure Codex MCP delivery"):
        init_project(str(project), agent_id="agent")


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
