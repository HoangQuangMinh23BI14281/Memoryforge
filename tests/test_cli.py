from __future__ import annotations

import json
from typing import Any

from memoryforge.cli.commands import run_command
from memoryforge.cli.parser import build_parser


def test_init_command_skips_indexing_by_default(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_init_project(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("memoryforge.cli.commands.init_project", fake_init_project)

    args = build_parser().parse_args(["init", "."])
    assert run_command(args) == 0

    assert captured["auto_index"] is False
    assert captured["configure_codex"] is False


def test_init_command_indexes_only_when_requested(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_init_project(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("memoryforge.cli.commands.init_project", fake_init_project)

    args = build_parser().parse_args(["init", ".", "--index"])
    assert run_command(args) == 0

    assert captured["auto_index"] is True


def test_index_command_runs_raw_index_by_default(monkeypatch):
    raw_captured: dict[str, Any] = {}
    planner_called = False

    def fake_index_project_markdown(**kwargs: Any) -> dict[str, Any]:
        raw_captured.update(kwargs)
        return {"enabled": True, "mode": "raw"}

    def fake_plan_project_markdown_analysis(**kwargs: Any) -> dict[str, Any]:
        nonlocal planner_called
        planner_called = True
        return {"enabled": True, "mode": "host_subagent_plan"}

    monkeypatch.setattr("memoryforge.cli.commands.index_project_markdown", fake_index_project_markdown)
    monkeypatch.setattr(
        "memoryforge.cli.commands.plan_project_markdown_analysis",
        fake_plan_project_markdown_analysis,
    )

    args = build_parser().parse_args(
        ["index", ".", "--agent-id", "codex", "--runner", "mock", "--max-files", "1"]
    )
    assert run_command(args) == 0

    assert raw_captured["agent_id"] == "codex"
    assert raw_captured["max_files"] == 1
    assert planner_called is False


def test_index_command_prepares_host_subagent_plan_only_when_requested(monkeypatch):
    raw_captured: dict[str, Any] = {}
    planner_captured: dict[str, Any] = {}

    def fake_index_project_markdown(**kwargs: Any) -> dict[str, Any]:
        raw_captured.update(kwargs)
        return {"enabled": True, "mode": "raw"}

    def fake_plan_project_markdown_analysis(**kwargs: Any) -> dict[str, Any]:
        planner_captured.update(kwargs)
        return {"enabled": True, "mode": "host_subagent_plan"}

    monkeypatch.setattr("memoryforge.cli.commands.index_project_markdown", fake_index_project_markdown)
    monkeypatch.setattr(
        "memoryforge.cli.commands.plan_project_markdown_analysis",
        fake_plan_project_markdown_analysis,
    )

    args = build_parser().parse_args(
        [
            "index",
            ".",
            "--agent-id",
            "codex",
            "--analyze",
            "--runner",
            "codex",
            "--analyze-min-bytes",
            "20000",
            "--analyze-max-files",
            "2",
        ]
    )
    assert run_command(args) == 0

    assert raw_captured["agent_id"] == "codex"
    assert planner_captured["agent_id"] == "codex"
    assert "runner" not in planner_captured
    assert planner_captured["min_file_bytes"] == 20_000
    assert planner_captured["max_files"] == 2


def test_index_analyze_outputs_host_subagent_argv_plan(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "disabled")
    monkeypatch.setenv("MEMORYFORGE_PROGRESS", "0")
    project = tmp_path / "project"
    project.mkdir()
    (project / "large.md").write_text(
        "Telemetry endpoint 4318 replaced stale 4317.\n\n" * 40,
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "--db",
            str(project / ".memoryforge" / "memory.db"),
            "index",
            str(project),
            "--agent-id",
            "codex",
            "--chunk-size",
            "120",
            "--overlap",
            "0",
            "--max-files",
            "1",
            "--analyze",
            "--analyze-min-bytes",
            "0",
            "--analyze-max-files",
            "1",
            "--batch-size",
            "1",
        ]
    )

    assert run_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    analyze = payload["analyze"]
    plan = analyze["plans"][0]
    batch = plan["batches"][0]

    assert analyze["mode"] == "host_subagent_plan"
    assert analyze["external_model_calls"] is False
    assert plan["expected_batch_count"] == plan["batch_count"]
    assert "--expected-batches" in plan["aggregate_command_argv"]
    assert batch["record_command_argv"][:3] == ["uv", "run", "memoryforge"]
    assert batch["fetch_command_argvs"][0][:3] == ["uv", "run", "memoryforge"]
    assert "rlm_chunk:" in batch["host_subagent_prompt"]


def test_init_command_configures_codex_only_when_requested(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_init_project(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("memoryforge.cli.commands.init_project", fake_init_project)

    args = build_parser().parse_args(["init", ".", "--configure-codex"])
    assert run_command(args) == 0

    assert captured["configure_codex"] is True


def test_hook_command_does_not_block_on_tty_stdin(monkeypatch, capsys):
    captured: dict[str, Any] = {}

    class TtyStdin:
        closed = False

        def isatty(self) -> bool:
            return True

        def read(self) -> str:
            raise AssertionError("hook command must not read from interactive stdin")

    def fake_handle_hook_event(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("sys.stdin", TtyStdin())
    monkeypatch.setattr("memoryforge.cli.commands.handle_hook_event", fake_handle_hook_event)

    args = build_parser().parse_args(
        ["hook", "SessionStart", "--db", "memory.db", "--agent-id", "agent", "--project-root", "."]
    )
    assert run_command(args) == 0
    output = capsys.readouterr().out

    assert captured["stdin_text"] == ""
    assert captured["source"] == "memoryforge_hook"
    assert captured["runtime"] == "auto"
    assert '"ok": true' in output
