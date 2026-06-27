"""Codex CLI project integration writers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


MEMORYFORGE_MCP_NAME = "memoryforge"
MEMORYFORGE_MCP_MARKER_START = "# --- MemoryForge MCP server ---"
MEMORYFORGE_MCP_MARKER_END = "# --- end MemoryForge MCP server ---"
MEMORYFORGE_AGENTS_MARKER_START = "<!-- MemoryForge instructions start -->"
MEMORYFORGE_AGENTS_MARKER_END = "<!-- MemoryForge instructions end -->"



MEMORYFORGE_AGENTS_CONTENT = """# MemoryForge Project Memory

MemoryForge is configured for this project. Use its MCP tools before answering questions that depend on project history, prior conversation, Markdown notes, or oversized source context.

- Use `recall_memory` for durable RLM/LTM recall. It combines BM25 and vector recall with RRF-style ranking where available.
- Use `build_context_bundle` or `build_runtime_context_bundle` when a prompt needs grounded context for the core model.
- Use `rlm_load`, `rlm_search`, and `rlm_chunk_get` for large files or documents instead of reading everything into the prompt.
- Use `rlm_run` only when the task needs real Codex CLI sub-agent analysis over large context.
- Store useful session evidence in LCM through MemoryForge hooks/MCP; do not create alternate memory folders.
- The project-local database is `.memoryforge/memory.db` unless `MEMORYFORGE_DB` explicitly overrides it.
"""
class RegisterStatus(str, Enum):
    REGISTERED = "registered"
    ALREADY = "already"
    MISMATCH = "mismatch"
    FAILED = "failed"


@dataclass(frozen=True)
class ServerSpec:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisterResult:
    status: RegisterStatus
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {RegisterStatus.REGISTERED, RegisterStatus.ALREADY}


class CodexMCPRegistrar:
    """Register MemoryForge's stdio MCP server in project-local Codex config."""

    def __init__(self, codex_dir: Path) -> None:
        self.codex_dir = codex_dir
        self.config_path = codex_dir / "config.toml"

    def get_server(self, server_name: str) -> ServerSpec | None:
        data = self._load_toml()
        servers = data.get("mcp_servers")
        if not isinstance(servers, dict):
            return None
        entry = servers.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        existing = self.get_server(spec.name)
        content = self._read_text()
        has_marker = MEMORYFORGE_MCP_MARKER_START in content and MEMORYFORGE_MCP_MARKER_END in content

        if existing is not None and _specs_equivalent(existing, spec):
            if has_marker:
                return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")
            if force and _looks_like_memoryforge_server(existing):
                content = _remove_mcp_server_sections(content, spec.name)
                return self._write_content(_append_or_replace_marker_block(content, spec))
            return RegisterResult(RegisterStatus.ALREADY, "matching user-managed configuration")

        if existing is not None and not has_marker:
            if force and _looks_like_memoryforge_server(existing):
                content = _remove_mcp_server_sections(content, spec.name)
                return self._write_content(_append_or_replace_marker_block(content, spec))
            return RegisterResult(
                RegisterStatus.MISMATCH,
                "user-managed [mcp_servers."
                f"{spec.name}] entry differs; rerun with --force to replace MemoryForge-owned entries",
            )

        return self._write_content(_append_or_replace_marker_block(content, spec))

    def unregister_server(self, server_name: str = MEMORYFORGE_MCP_NAME) -> bool:
        if server_name != MEMORYFORGE_MCP_NAME or not self.config_path.exists():
            return False
        content = self._read_text()
        if MEMORYFORGE_MCP_MARKER_START not in content or MEMORYFORGE_MCP_MARKER_END not in content:
            return False
        new_content = _remove_marker_block(content)
        try:
            self.config_path.write_text(new_content, encoding="utf-8")
        except OSError:
            return False
        return True

    def _load_toml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        try:
            data = tomllib.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _read_text(self) -> str:
        try:
            return self.config_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _write_content(self, content: str) -> RegisterResult:
        try:
            self.codex_dir.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return RegisterResult(RegisterStatus.FAILED, str(exc))
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote {self.config_path}")


def build_memoryforge_mcp_spec(db_path: Path) -> ServerSpec:
    return ServerSpec(
        name=MEMORYFORGE_MCP_NAME,
        command="uv",
        args=("run", "memoryforge-mcp"),
        env={"MEMORYFORGE_DB": str(db_path)},
    )



def build_global_memoryforge_mcp_spec() -> ServerSpec:
    return ServerSpec(
        name=MEMORYFORGE_MCP_NAME,
        command="memoryforge-mcp",
    )


def install_codex_agents_md(agents_path: Path) -> None:
    existing = ""
    try:
        existing = agents_path.read_text(encoding="utf-8")
    except OSError:
        existing = ""
    block = (
        f"{MEMORYFORGE_AGENTS_MARKER_START}\n"
        f"{MEMORYFORGE_AGENTS_CONTENT.rstrip()}\n"
        f"{MEMORYFORGE_AGENTS_MARKER_END}"
    )
    if MEMORYFORGE_AGENTS_MARKER_START in existing and MEMORYFORGE_AGENTS_MARKER_END in existing:
        start = existing.index(MEMORYFORGE_AGENTS_MARKER_START)
        end = existing.index(MEMORYFORGE_AGENTS_MARKER_END) + len(MEMORYFORGE_AGENTS_MARKER_END)
        before = existing[:start].rstrip()
        after = existing[end:].lstrip()
        pieces = [piece for piece in (before, block, after) if piece]
        content = "\n\n".join(pieces) + "\n"
    elif existing.strip():
        content = f"{existing.rstrip()}\n\n{block}\n"
    else:
        content = f"{block}\n"
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    agents_path.write_text(content, encoding="utf-8")
def install_codex_hooks(
    hooks_path: Path,
    *,
    db_path: Path,
    agent_id: str,
    project_root: Path,
    hook_runner: Path,
) -> None:
    payload = _read_json(hooks_path)
    hooks = payload.get("hooks")
    payload["hooks"] = _merge_memoryforge_hooks(
        hooks if isinstance(hooks, dict) else {},
        db_path=db_path,
        agent_id=agent_id,
        project_root=project_root,
        hook_runner=hook_runner,
    )
    _write_json(hooks_path, payload)


def _merge_memoryforge_hooks(
    existing: dict[str, Any],
    *,
    db_path: Path,
    agent_id: str,
    project_root: Path,
    hook_runner: Path,
) -> dict[str, Any]:
    hooks: dict[str, Any] = {}
    for event, value in existing.items():
        entries = _without_memoryforge_hooks(value)
        if entries:
            hooks[str(event)] = entries
    for event, hook_event, matcher, status in (
        ("SessionStart", "session-start", "startup|resume|clear|compact", "Preparing MemoryForge"),
        ("UserPromptSubmit", "user-prompt-submit", None, "Recording prompt evidence"),
        ("PreCompact", "pre-compact", "manual|auto", "Recording pre-compaction context"),
        ("Stop", "stop", None, "Finalizing MemoryForge context"),
    ):
        entries = _without_memoryforge_hooks(hooks.get(event))
        entries.append(
            _hook_group(
                _hook_command(hook_runner, hook_event, db_path, agent_id, project_root),
                matcher=matcher,
                status_message=status,
            )
        )
        hooks[event] = entries
    return hooks


def _without_memoryforge_hooks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if _looks_like_memoryforge_hook(item):
            continue
        entries.append(item)
    return entries


def _looks_like_memoryforge_hook(item: dict[str, Any]) -> bool:
    dumped = json.dumps(item)
    return any(
        marker in dumped
        for marker in (
            "memoryforge hook",
            "memoryforge.exe hook",
            "memoryforge-hook",
            "memoryforge.cli.main hook",
        )
    )

def _hook_group(
    command: str, *, matcher: str | None, status_message: str
) -> dict[str, Any]:
    handler: dict[str, Any] = {
        "type": "command",
        "command": command,
        "timeout": 20,
        "statusMessage": status_message,
    }
    group: dict[str, Any] = {"hooks": [handler]}
    if matcher is not None:
        group["matcher"] = matcher
    return group


def _hook_command(
    hook_runner: Path,
    event: str,
    db_path: Path,
    agent_id: str,
    project_root: Path,
) -> str:
    if hook_runner.suffix.lower() == ".cmd":
        runner = project_root / ".venv" / "Scripts" / "memoryforge.exe"
        return (
            f"{_cmd_path_arg(_relative_to_project(runner, project_root))} hook {event} "
            f"--db {_cmd_path_arg(_relative_to_project(db_path, project_root))} "
            f"--agent-id {_cmd_value_arg(agent_id)} "
            "--project-root ."
        )
    return (
        f"{shlex.quote(str(hook_runner))} {hook_event_arg(event)} "
        f"--db {shlex.quote(str(db_path))} "
        f"--agent-id {shlex.quote(agent_id)} "
        f"--project-root {shlex.quote(str(project_root))}"
    )


def hook_event_arg(event: str) -> str:
    return shlex.quote(event)


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _cmd_path_arg(value: str) -> str:
    value = value.replace("/", "\\")
    if " " not in value and "	" not in value:
        return value
    return '"' + value.replace('"', '""') + '"'


def _cmd_value_arg(value: str) -> str:
    if " " not in value and "	" not in value:
        return value
    return '"' + value.replace('"', '""') + '"'


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    args_value = entry.get("args")
    args = tuple(str(arg) for arg in args_value) if isinstance(args_value, list) else ()
    env_value = entry.get("env")
    env = {str(k): str(v) for k, v in env_value.items()} if isinstance(env_value, dict) else {}
    return ServerSpec(name=name, command=str(entry.get("command", "")), args=args, env=env)


def _specs_equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    return a.name == b.name and a.command == b.command and a.args == b.args and a.env == b.env


def _looks_like_memoryforge_server(spec: ServerSpec) -> bool:
    if spec.name != MEMORYFORGE_MCP_NAME:
        return False
    return spec.command == "memoryforge-mcp" or (
        spec.command == "uv" and "memoryforge-mcp" in spec.args
    )


def _append_or_replace_marker_block(content: str, spec: ServerSpec) -> str:
    block = _render_mcp_block(spec)
    if MEMORYFORGE_MCP_MARKER_START in content and MEMORYFORGE_MCP_MARKER_END in content:
        start = content.index(MEMORYFORGE_MCP_MARKER_START)
        end = content.index(MEMORYFORGE_MCP_MARKER_END) + len(MEMORYFORGE_MCP_MARKER_END)
        before = content[:start].rstrip("\n")
        after = content[end:].lstrip("\n")
        if before and after:
            return f"{before}\n\n{block}\n\n{after}"
        if before:
            return f"{before}\n\n{block}\n"
        if after:
            return f"{block}\n\n{after}"
        return f"{block}\n"
    if content.strip():
        return f"{content.rstrip()}\n\n{block}\n"
    return f"{block}\n"


def _remove_marker_block(content: str) -> str:
    start = content.index(MEMORYFORGE_MCP_MARKER_START)
    end = content.index(MEMORYFORGE_MCP_MARKER_END) + len(MEMORYFORGE_MCP_MARKER_END)
    before = content[:start].rstrip("\n")
    after = content[end:].lstrip("\n")
    if before and after:
        return f"{before}\n\n{after}"
    if before or after:
        return f"{before or after}".rstrip("\n") + "\n"
    return ""


def _remove_mcp_server_sections(content: str, server_name: str) -> str:
    target = f"mcp_servers.{server_name}"
    output: list[str] = []
    skipping = False
    for line in content.splitlines():
        table = _toml_table_name(line)
        if table == target or table.startswith(f"{target}."):
            skipping = True
            continue
        if skipping and table:
            skipping = False
        if not skipping:
            output.append(line)
    return "\n".join(output).strip() + ("\n" if output else "")


def _toml_table_name(line: str) -> str:
    stripped = line.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return ""
    return stripped.strip("[]").strip()


def _render_mcp_block(spec: ServerSpec) -> str:
    lines = [
        MEMORYFORGE_MCP_MARKER_START,
        f"[mcp_servers.{spec.name}]",
        f"command = {_toml_str(spec.command)}",
    ]
    if spec.args:
        lines.append(f"args = [{', '.join(_toml_str(arg) for arg in spec.args)}]")
    if spec.env:
        lines.append("")
        lines.append(f"[mcp_servers.{spec.name}.env]")
        for key, value in spec.env.items():
            lines.append(f"{key} = {_toml_str(value)}")
    lines.append(MEMORYFORGE_MCP_MARKER_END)
    return "\n".join(lines)


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


