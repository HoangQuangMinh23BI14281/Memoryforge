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
)


def init_project(
    project_root: str = ".",
    db_path: str | None = None,
    agent_id: str = "default",
    *,
    configure_codex: bool = True,
    auto_index: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()

    memory_dir = root / ".memoryforge"
    hooks_dir = memory_dir / "hooks"
    memory_dir.mkdir(parents=True, exist_ok=True)
    hooks_dir.mkdir(parents=True, exist_ok=True)

    resolved_db = Path(db_path).expanduser() if db_path else memory_dir / "memory.db"
    if not resolved_db.is_absolute():
        resolved_db = (root / resolved_db).resolve()
    init_memoryforge_schema(str(resolved_db))

    hook_runner = _write_hook_runner(root, hooks_dir, force=force)
    written = [str(hook_runner)]
    if configure_codex:
        written.extend(
            str(path)
            for path in _write_codex_settings(
                root, resolved_db, agent_id, hook_runner, force=force
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
    }


def ensure_project_initialized(
    project_root: str = ".",
    *,
    agent_id: str = "default",
    configure_codex: bool = False,
    auto_index: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    db_path = root / ".memoryforge" / "memory.db"
    return init_project(
        str(root),
        db_path=str(db_path),
        agent_id=agent_id,
        configure_codex=configure_codex,
        auto_index=auto_index,
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
    runner = hooks_dir / "memoryforge-hook.sh"
    if runner.exists() and not force:
        return runner
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(root))}",
                "if command -v uv >/dev/null 2>&1; then",
                '  exec uv run memoryforge hook "$@"',
                "fi",
                'exec memoryforge hook "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return runner


def _write_codex_settings(
    root: Path, db_path: Path, agent_id: str, hook_runner: Path, *, force: bool
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

    install_codex_hooks(
        hooks_path,
        db_path=db_path,
        agent_id=agent_id,
        project_root=root,
        hook_runner=hook_runner,
    )
    install_codex_agents_md(agents_path)
    return [config_path, hooks_path, agents_path]

