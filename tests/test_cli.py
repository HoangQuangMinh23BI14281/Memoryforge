from __future__ import annotations

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


def test_init_command_configures_codex_only_when_requested(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_init_project(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("memoryforge.cli.commands.init_project", fake_init_project)

    args = build_parser().parse_args(["init", ".", "--configure-codex"])
    assert run_command(args) == 0

    assert captured["configure_codex"] is True
