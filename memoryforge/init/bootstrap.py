"""Project bootstrap and Codex delivery wiring."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
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
    auto_index: bool = False,
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
    codex_hooks: dict[str, Any] = {"ok": False, "skipped": "hooks not requested"}
    capture_config: dict[str, Any] = {
        "enabled": False,
        "mode": "disabled",
        "primary": "install and trust WSL/Linux hooks for Codex CLI lifecycle capture",
    }
    if install_hooks:
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_runner = _write_hook_runner(root, hooks_dir, force=force)
        written.append(str(hook_runner))
        codex_hooks = _write_codex_hooks_config(
            root,
            resolved_db,
            agent_id=agent_id,
            hook_runner=hook_runner,
            force=force,
        )
        written.append(str(codex_hooks["path"]))
        capture_config = {
            "enabled": True,
            "mode": "linux_wsl" if os.name != "nt" else "windows",
            "primary": "project-local hook runner",
            "runner_path": str(hook_runner),
            "log_path": str(hooks_dir / "memoryforge-hook.log"),
            "model_calls": False,
            "codex_account_required": False,
        }

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
                "hooks": codex_hooks,
                "capture": capture_config,
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
        "hooks": codex_hooks,
        "capture": capture_config,
        "codex_init": codex_init,
        "codex_mcp": codex_mcp,
    }


def ensure_project_initialized(
    project_root: str = ".",
    *,
    agent_id: str = DEFAULT_AGENT_ID,
    configure_codex: bool = False,
    auto_index: bool = False,
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
            "hooks": config.get("hooks") or {"ok": False, "skipped": "already initialized"},
            "capture": config.get("capture")
            or {
                "enabled": bool(config.get("hooks_enabled", False)),
                "mode": "legacy",
                "primary": "legacy hook config" if config.get("hooks_enabled", False) else "disabled",
            },
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


def _write_codex_hooks_config(
    root: Path,
    db_path: Path,
    *,
    agent_id: str,
    hook_runner: Path,
    force: bool,
) -> dict[str, Any]:
    codex_dir = root / ".codex"
    hooks_path = codex_dir / "hooks.json"
    codex_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_json_object(hooks_path)
    hooks = existing.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}

    for event_name in _codex_hook_events():
        groups = hooks.get(event_name)
        if not isinstance(groups, list):
            groups = []
        hooks[event_name] = [
            group
            for group in groups
            if not _is_memoryforge_hook_group(group)
        ]
        hooks[event_name].append(
            _codex_hook_group(
                event_name,
                root=root,
                db_path=db_path,
                agent_id=agent_id,
                hook_runner=hook_runner,
            )
        )

    payload = dict(existing)
    payload["hooks"] = hooks
    if hooks_path.exists() and not force:
        # Merging is safe without force; only previous MemoryForge hook entries are replaced.
        pass
    hooks_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "path": str(hooks_path),
        "events": list(_codex_hook_events()),
        "trust_required": True,
        "model_calls": False,
        "codex_account_required": False,
        "note": "Run /hooks in Codex to review and trust the local MemoryForge runner.",
    }


def _codex_hook_events() -> tuple[str, ...]:
    return (
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "PreCompact",
        "PostCompact",
        "Stop",
    )


def _codex_hook_group(
    event_name: str,
    *,
    root: Path,
    db_path: Path,
    agent_id: str,
    hook_runner: Path,
) -> dict[str, Any]:
    windows_command = _hook_command(
        hook_runner,
        event_name,
        db_path=db_path,
        agent_id=agent_id,
        project_root=root,
        windows=True,
    )
    command = (
        windows_command
        if os.name == "nt" or hook_runner.suffix.lower() in {".cmd", ".bat", ".exe"}
        else _hook_command(
            hook_runner,
            event_name,
            db_path=db_path,
            agent_id=agent_id,
            project_root=root,
            windows=False,
        )
    )
    hook: dict[str, Any] = {
        "type": "command",
        "command": command,
        "commandWindows": windows_command,
        "timeout": 30,
        "statusMessage": f"MemoryForge {event_name}",
    }
    group: dict[str, Any] = {"hooks": [hook]}
    if event_name == "SessionStart":
        group["matcher"] = "startup|resume|compact"
    elif event_name in {"PostToolUse", "PreCompact", "PostCompact"}:
        group["matcher"] = "*" if event_name == "PostToolUse" else "manual|auto"
    return group


def _hook_command(
    hook_runner: Path,
    event_name: str,
    *,
    db_path: Path,
    agent_id: str,
    project_root: Path,
    windows: bool,
) -> str:
    argv = [
        str(hook_runner),
        event_name,
        "--db",
        str(db_path),
        "--agent-id",
        agent_id,
        "--project-root",
        str(project_root),
    ]
    return subprocess.list2cmdline(argv) if windows else shlex.join(argv)


def _is_memoryforge_hook_group(group: Any) -> bool:
    if not isinstance(group, dict):
        return False
    return "memoryforge hook" in json.dumps(group).lower() or "memoryforge-hook" in json.dumps(
        group
    ).lower()


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


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
                    'set "PYTHONIOENCODING=utf-8"',
                    'set "MEMORYFORGE_HOOK_STRICT=0"',
                    'set "MEMORYFORGE_HOOK_LOG=.memoryforge\\hooks\\memoryforge-hook.log"',
                    'echo [%date% %time%] memoryforge hook %* >> "%MEMORYFORGE_HOOK_LOG%"',
                    'if exist ".venv\\Scripts\\python.exe" (',
                    '  ".venv\\Scripts\\python.exe" -m memoryforge.cli.main hook %* --source memoryforge_hook --runtime windows >> "%MEMORYFORGE_HOOK_LOG%" 2>&1',
                    '  exit /b 0',
                    ')',
                    'where uv >nul 2>nul',
                    'if %errorlevel%==0 (',
                    '  uv run --no-sync python -m memoryforge.cli.main hook %* --source memoryforge_hook --runtime windows >> "%MEMORYFORGE_HOOK_LOG%" 2>&1',
                    '  exit /b 0',
                    ')',
                    'where python >nul 2>nul',
                    'if %errorlevel%==0 (',
                    '  python -m memoryforge.cli.main hook %* --source memoryforge_hook --runtime windows >> "%MEMORYFORGE_HOOK_LOG%" 2>&1',
                    '  exit /b 0',
                    ')',
                    'echo MemoryForge hook runner not found >> "%MEMORYFORGE_HOOK_LOG%"',
                    'exit /b 0',
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
                "set +e",
                f"cd {shlex.quote(str(root))}",
                "export PYTHONIOENCODING=utf-8",
                "export MEMORYFORGE_HOOK_STRICT=0",
                'MEMORYFORGE_HOOK_LOG=".memoryforge/hooks/memoryforge-hook.log"',
                'printf "[%s] memoryforge hook %s\\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$MEMORYFORGE_HOOK_LOG"',
                'if [ -x ".venv/bin/python" ]; then',
                '  .venv/bin/python -m memoryforge.cli.main hook "$@" --source memoryforge_hook --runtime linux-wsl >> "$MEMORYFORGE_HOOK_LOG" 2>&1',
                "  exit 0",
                "fi",
                "if command -v uv >/dev/null 2>&1; then",
                '  uv run --no-sync python -m memoryforge.cli.main hook "$@" --source memoryforge_hook --runtime linux-wsl >> "$MEMORYFORGE_HOOK_LOG" 2>&1',
                "  exit 0",
                "fi",
                "if command -v python >/dev/null 2>&1; then",
                '  python -m memoryforge.cli.main hook "$@" --source memoryforge_hook --runtime linux-wsl >> "$MEMORYFORGE_HOOK_LOG" 2>&1',
                "  exit 0",
                "fi",
                'echo "MemoryForge hook runner not found" >> "$MEMORYFORGE_HOOK_LOG"',
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return runner
