"""Active core-runtime integration helpers.

Runtime integration is intentionally separate from sub-agent execution. These
helpers only verify that MemoryForge can deliver context to the user's active
runtime and return bundle metadata for that runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


class RuntimeIntegrationError(RuntimeError):
    """Raised when MemoryForge cannot deliver context to the active runtime."""


@dataclass(frozen=True)
class RuntimeIntegration:
    runtime: str
    project_root: str
    delivery: str
    mcp_configured: bool
    hooks_configured: bool
    memoryforge_configured: bool
    config_path: str | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "project_root": self.project_root,
            "delivery": self.delivery,
            "mcp_configured": self.mcp_configured,
            "hooks_configured": self.hooks_configured,
            "memoryforge_configured": self.memoryforge_configured,
            "config_path": self.config_path,
            "diagnostics": self.diagnostics,
        }


def resolve_runtime_integration(
    project_root: str | Path,
    *,
    runtime: str = "auto",
    expected_db_path: str | Path | None = None,
) -> RuntimeIntegration:
    """Detect an active runtime and verify MemoryForge context delivery.

    Only Codex CLI is supported today. Unknown or unconfigured runtimes raise
    instead of silently falling back to a separate answer model.
    """

    root = Path(project_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeIntegrationError(f"project_root does not exist or is not a directory: {root}")

    requested = runtime.lower()
    if requested not in {"auto", "codex"}:
        raise RuntimeIntegrationError(f"Unsupported MemoryForge runtime: {runtime}")

    agents_path = root / "AGENTS.md"
    mf_config_path = root / ".memoryforge" / "config.json"
    agents_configured = _agents_memoryforge_configured(agents_path)
    if requested == "auto" and not agents_configured:
        raise RuntimeIntegrationError(
            "Could not identify active Codex MemoryForge instructions. Run `memoryforge init <project>` first."
        )

    memoryforge_config = _load_json(mf_config_path)
    mcp_result = memoryforge_config.get("codex_mcp")
    mcp_configured = agents_configured and isinstance(mcp_result, dict) and bool(mcp_result.get("ok"))
    hooks_configured = False
    memoryforge_configured = bool(memoryforge_config)
    if not memoryforge_configured:
        raise RuntimeIntegrationError(
            "MemoryForge project config is missing. Run `memoryforge init <project>` first."
        )
    if not agents_configured:
        raise RuntimeIntegrationError(
            "Project AGENTS.md is missing MemoryForge instructions. Run `memoryforge init <project>` first."
        )
    if not mcp_configured:
        raise RuntimeIntegrationError(
            "MemoryForge MCP is not registered for Codex CLI. Run `memoryforge init <project>` again or `codex mcp add memoryforge -- uv run memoryforge-mcp`."
        )
    memoryforge_db_path = _memoryforge_config_db_path(memoryforge_config, root)
    if memoryforge_db_path is None:
        raise RuntimeIntegrationError(
            "MemoryForge project config is missing db_path. Run `memoryforge init <project>` first."
        )
    if expected_db_path is not None:
        expected = _resolve_project_path(expected_db_path, root)
        if expected != memoryforge_db_path:
            raise RuntimeIntegrationError(
                "Validated runtime delivery does not match the active MemoryForge database."
            )

    return RuntimeIntegration(
        runtime="codex",
        project_root=str(root),
        delivery="mcp:build_context_bundle",
        mcp_configured=mcp_configured,
        hooks_configured=hooks_configured,
        memoryforge_configured=memoryforge_configured,
        config_path=str(agents_path),
        diagnostics={
            "answer_model_used": False,
            "subagent_runner_used": False,
            "fail_loud": True,
            "db_path_verified": True,
            "db_path": str(memoryforge_db_path),
        },
    )


def _agents_memoryforge_configured(agents_path: Path) -> bool:
    try:
        text = agents_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "<!-- MemoryForge instructions start -->" in text and "recall_memory" in text


def _codex_memoryforge_mcp_configured(config_path: Path) -> bool:
    return _codex_memoryforge_mcp_server(config_path) is not None


def _codex_memoryforge_mcp_server(config_path: Path) -> dict[str, Any] | None:
    data = _load_toml(config_path)
    servers = data.get("mcp_servers")
    if not isinstance(servers, dict):
        return None
    memoryforge = servers.get("memoryforge")
    if not isinstance(memoryforge, dict):
        return None
    command = memoryforge.get("command")
    args = memoryforge.get("args")
    if command not in {"uv", "memoryforge-mcp"}:
        return None
    if command == "memoryforge-mcp":
        return memoryforge
    if isinstance(args, list) and "memoryforge-mcp" in [str(arg) for arg in args]:
        return memoryforge
    return None


def _codex_memoryforge_mcp_db_path(config_path: Path, project_root: Path) -> Path | None:
    server = _codex_memoryforge_mcp_server(config_path)
    if server is None:
        return None
    env = server.get("env")
    if not isinstance(env, dict):
        return None
    db_path = env.get("MEMORYFORGE_DB")
    if not isinstance(db_path, str) or not db_path:
        return None
    return _resolve_project_path(db_path, project_root)


def _memoryforge_config_db_path(config: dict[str, Any], project_root: Path) -> Path | None:
    db_path = config.get("db_path")
    if not isinstance(db_path, str) or not db_path:
        return None
    return _resolve_project_path(db_path, project_root)


def _resolve_project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _codex_hooks_configured(hooks_path: Path) -> bool:
    data = _load_json(hooks_path)
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return False
    for hook_entries in hooks.values():
        if "memoryforge-hook" in json.dumps(hook_entries):
            return True
    return False


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
