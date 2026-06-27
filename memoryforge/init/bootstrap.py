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
    CodexMCPRegistrar,
    RegisterStatus,
    build_global_memoryforge_mcp_spec,
    build_memoryforge_mcp_spec,
    install_codex_agents_md,
    install_codex_hooks,
    remove_codex_hooks,
)


def init_project(
    project_root: str = ".",
    db_path: str | None = None,
    agent_id: str = "default",
    *,
    configure_codex: bool = True,
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
    if configure_codex:
        written.extend(
            str(path)
            for path in _write_codex_settings(
                root,
                resolved_db,
                agent_id,
                hook_runner=hook_runner,
                install_hooks=install_hooks,
                force=force,
            )
        )

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
    }


def ensure_project_initialized(
    project_root: str = ".",
    *,
    agent_id: str = "default",
    configure_codex: bool = False,
    auto_index: bool = True,
    install_hooks: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    db_path = root / ".memoryforge" / "memory.db"
    return init_project(
        str(root),
        db_path=str(db_path),
        agent_id=agent_id,
        configure_codex=configure_codex,
        auto_index=auto_index,
        install_hooks=install_hooks,
        force=False,
    )


def install_codex_global(*, force: bool = False) -> dict[str, Any]:
    codex_dir = Path.home() / ".codex"
    registrar = CodexMCPRegistrar(codex_dir)
    result = registrar.register_server(build_global_memoryforge_mcp_spec(), force=force)
    if result.status in {RegisterStatus.MISMATCH, RegisterStatus.FAILED}:
        raise ValueError(f"Could not configure global Codex MCP delivery: {result.detail}")
    agents_path = codex_dir / "AGENTS.md"
    install_codex_agents_md(agents_path)
    return {
        "codex_dir": str(codex_dir),
        "config_path": str(codex_dir / "config.toml"),
        "agents_path": str(agents_path),
        "status": result.status.value,
        "detail": result.detail,
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


def _write_codex_settings(
    root: Path,
    db_path: Path,
    agent_id: str,
    *,
    hook_runner: Path | None,
    install_hooks: bool,
    force: bool,
) -> list[Path]:
    codex_dir = root / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    config_path = codex_dir / "config.toml"
    hooks_path = codex_dir / "hooks.json"
    agents_path = codex_dir / "AGENTS.md"

    result = CodexMCPRegistrar(codex_dir).register_server(
        build_memoryforge_mcp_spec(db_path), force=force
    )
    if result.status in {RegisterStatus.MISMATCH, RegisterStatus.FAILED}:
        raise ValueError(f"Could not configure Codex MCP delivery: {result.detail}")

    written = [config_path]
    if install_hooks:
        if hook_runner is None:
            raise ValueError("hook_runner is required when install_hooks=True")
        install_codex_hooks(
            hooks_path,
            db_path=db_path,
            agent_id=agent_id,
            project_root=root,
            hook_runner=hook_runner,
        )
        written.append(hooks_path)
    else:
        remove_codex_hooks(hooks_path)
    install_codex_agents_md(agents_path)
    written.append(agents_path)
    return written
