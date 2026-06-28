"""Project bootstrap and Codex delivery wiring."""

from __future__ import annotations

import json
import os
import shlex
import stat
from pathlib import Path
from typing import Any

from memoryforge.agents.codex_sync import load_codex_defaults
from memoryforge.db import init_memoryforge_schema
from memoryforge.init.autoload import index_project_markdown
from memoryforge.init.codex import (
    ensure_codex_mcp_registered,
    install_codex_agents_md,
    run_codex_init,
)

DEFAULT_AGENT_ID = "codex"


def init_project(
    project_root: str = ".",
    db_path: str | None = None,
    agent_id: str = DEFAULT_AGENT_ID,
    *,
    configure_codex: bool = False,
    auto_index: bool = True,
    install_hooks: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()

    memory_dir = root / ".memoryforge"
    hooks_dir = memory_dir / "hooks"
    memory_dir.mkdir(parents=True, exist_ok=True)

    resolved_db = Path(db_path).expanduser() if db_path else memory_dir / "memory.db"
    if not resolved_db.is_absolute():
        resolved_db = (root / resolved_db).resolve()
    init_memoryforge_schema(str(resolved_db))

    written: list[str] = []
    hook_runner: Path | None = None
    if install_hooks:
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_runner = _write_hook_runner(root, hooks_dir, force=force)
        written.append(str(hook_runner))

    codex_init = (
        run_codex_init(root) if configure_codex else _codex_init_skipped(root / "AGENTS.md")
    )
    install_codex_agents_md(root / "AGENTS.md")
    written.append(str(root / "AGENTS.md"))
    codex_mcp = ensure_codex_mcp_registered() if configure_codex else _codex_mcp_skipped()

    indexed = {"files": 0, "chunks": 0, "long_term_items": 0, "enabled": False}
    if auto_index:
        indexed = index_project_markdown(
            db_path=str(resolved_db),
            agent_id=agent_id,
            project_root=str(root),
        )

    codex_defaults = load_codex_defaults(root)
    subagent_config: dict[str, Any] = {
        "runner": "codex",
        "model": (
            os.environ.get("MEMORYFORGE_SUBAGENT_MODEL")
            or os.environ.get("MEMORYFORGE_MODEL")
            or codex_defaults.model
            or "gpt-5.4"
        ),
        "codex_sync": bool(codex_defaults.base_url),
    }

    config_path = memory_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "db_path": str(resolved_db),
                "agent_id": agent_id,
                "project_root": str(root),
                "auto_index": bool(auto_index),
                "hooks_enabled": bool(install_hooks),
                "subagent": subagent_config,
                "codex_init": codex_init,
                "codex_mcp": codex_mcp,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(str(config_path))

    return {
        "project_root": str(root),
        "db_path": str(resolved_db),
        "indexed": indexed,
        "written": written,
        "hooks_enabled": bool(install_hooks),
        "codex_init": codex_init,
        "codex_mcp": codex_mcp,
    }


def ensure_project_initialized(
    project_root: str = ".",
    *,
    agent_id: str = DEFAULT_AGENT_ID,
    configure_codex: bool = False,
    auto_index: bool = True,
    install_hooks: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    memory_dir = root / ".memoryforge"
    config_path = memory_dir / "config.json"
    db_path = memory_dir / "memory.db"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
        resolved_db = Path(str(config.get("db_path") or db_path)).expanduser()
        if not resolved_db.is_absolute():
            resolved_db = (root / resolved_db).resolve()
        indexed = {"files": 0, "chunks": 0, "long_term_items": 0, "enabled": False}
        if auto_index:
            indexed = index_project_markdown(
                db_path=str(resolved_db),
                agent_id=agent_id,
                project_root=str(root),
            )
        return {
            "project_root": str(root),
            "db_path": str(resolved_db),
            "indexed": indexed,
            "written": [],
            "hooks_enabled": bool(config.get("hooks_enabled", False)),
            "codex_init": {"ok": True, "skipped": "already initialized"},
            "codex_mcp": (
                ensure_codex_mcp_registered()
                if configure_codex
                else config.get("codex_mcp") or _codex_mcp_skipped()
            ),
        }
    return init_project(
        str(root),
        db_path=str(db_path),
        agent_id=agent_id,
        configure_codex=configure_codex,
        auto_index=auto_index,
        install_hooks=install_hooks,
        force=False,
    )


def _codex_init_skipped(agents_path: Path) -> dict[str, Any]:
    return {
        "ok": True,
        "skipped": "codex /init not requested",
        "agents_path": str(agents_path),
    }


def _codex_mcp_skipped() -> dict[str, Any]:
    return {
        "ok": False,
        "skipped": "codex MCP registration not requested",
        "name": "memoryforge",
    }


def _write_hook_runner(root: Path, hooks_dir: Path, *, force: bool) -> Path:
    if os.name == "nt":
        runner = hooks_dir / "memoryforge-hook.cmd"
        if runner.exists() and not force:
            return runner
        runner.write_text(
            "\r\n".join(
                [
                    "@echo off",
                    "setlocal",
                    f'cd /d "{root}"',
                    'if exist ".venv\\Scripts\\memoryforge.exe" (',
                    '  ".venv\\Scripts\\memoryforge.exe" hook %*',
                    '  exit /b %errorlevel%',
                    ')',
                    'where uv >nul 2>nul',
                    'if %errorlevel%==0 (',
                    '  uv run memoryforge hook %*',
                    '  exit /b %errorlevel%',
                    ')',
                    'where memoryforge.exe >nul 2>nul',
                    'if %errorlevel%==0 (',
                    '  memoryforge.exe hook %*',
                    '  exit /b %errorlevel%',
                    ')',
                    'echo MemoryForge hook runner not found 1>&2',
                    'exit /b 1',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return runner
    runner = hooks_dir / "memoryforge-hook.sh"
    if runner.exists() and not force:
        return runner
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env sh",
                "set -eu",
                f"cd {shlex.quote(str(root))}",
                'if [ -x ".venv/bin/memoryforge" ]; then',
                '  exec .venv/bin/memoryforge hook "$@"',
                "fi",
                "if command -v uv >/dev/null 2>&1; then",
                '  exec uv run memoryforge hook "$@"',
                "fi",
                "if command -v memoryforge >/dev/null 2>&1; then",
                '  exec memoryforge hook "$@"',
                "fi",
                'echo "MemoryForge hook runner not found" >&2',
                "exit 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return runner
