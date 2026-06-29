"""Codex AGENTS.md integration writers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

MEMORYFORGE_AGENTS_MARKER_START = "<!-- MemoryForge instructions start -->"
MEMORYFORGE_AGENTS_MARKER_END = "<!-- MemoryForge instructions end -->"
MEMORYFORGE_MCP_NAME = "memoryforge"

MEMORYFORGE_AGENTS_CONTENT = """# MemoryForge Project Memory

MemoryForge is the project memory layer for this repository. Use it before raw grep when the user asks about prior decisions, docs, setup, architecture, roadmap, or long-running context.

- Use MCP `recall_memory` first for factual project-memory questions.
- Use MCP `build_context_bundle` when the answer needs grounded LCM/LTM context before responding.
- WSL/Linux hooks are the required Codex CLI lifecycle path. Completed Codex turns are auto-captured into LCM after `/hooks` trust. The hook is local-only: it does not call a model and does not run `codex exec`.
- If hooks are not active or not trusted, stop and tell the user MemoryForge LCM capture is unavailable for this Codex session.
- Use MCP `index_analyze` when large Markdown needs RLM host-subagent analysis. It indexes raw chunks, returns `plans[].batches[].host_subagent_prompt`, and does not call a model. Treat each returned prompt as a host-subagent task. Each subagent fetches chunks, cites `rlm_chunk:<id>`, then records its output with the returned `record_command_argv`. Run the returned `aggregate_command_argv` only after every planned batch is recorded.
- For explicit CLI indexing, use `uv run memoryforge index . --agent-id codex`; add `--analyze` only when host-subagent RLM summaries are needed.
- Do not call low-level RLM MCP tools from Codex interactive mode.
- Inspect LCM state with `uv run memoryforge --db .memoryforge/memory.db lcm-sessions`, `lcm-messages`, `lcm-context`, and `lcm-summary`.
- Fall back to `rg`, `Get-Content`, or direct file reads only if MemoryForge MCP is unavailable or returns no relevant evidence.
- The project-local database is `.memoryforge/memory.db` unless `MEMORYFORGE_DB` explicitly overrides it.
"""

_WINDOWS_CREATE_NO_WINDOW = 0x08000000


def _run_hidden(
    command: list[str], *, cwd: Path | None = None, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "text": True,
        "capture_output": True,
        "timeout": timeout_s,
        "check": False,
    }
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if os.name == "nt":
        kwargs["creationflags"] = _WINDOWS_CREATE_NO_WINDOW
    return subprocess.run(command, **kwargs)


def run_codex_init(root: Path, *, timeout_s: float = 5.0) -> dict[str, Any]:
    """Best-effort Codex /init call. Fall back quickly if Codex is slow or unavailable."""

    agents_path = root / "AGENTS.md"
    if agents_path.exists():
        return {
            "ok": True,
            "skipped": "AGENTS.md already exists",
            "agents_path": str(agents_path),
        }

    binary = shutil.which("codex")
    if not binary:
        return {
            "ok": False,
            "skipped": "codex cli not found on PATH",
            "agents_path": str(agents_path),
        }

    output_file = Path(tempfile.gettempdir()) / "memoryforge-codex-init-last-message.txt"
    try:
        output_file.unlink(missing_ok=True)
    except OSError:
        pass
    command = [
        binary,
        "exec",
        "--cd",
        str(root),
        "--skip-git-repo-check",
        "--color",
        "never",
        "--output-last-message",
        str(output_file),
        "/init",
    ]
    try:
        completed = _run_hidden(command, cwd=root, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "timeout": timeout_s,
            "skipped": "codex /init timed out; falling back to direct AGENTS patch",
            "agents_path": str(agents_path),
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "agents_path": str(agents_path),
        }

    payload: dict[str, Any] = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "agents_path": str(agents_path),
    }
    try:
        last_message = output_file.read_text(encoding="utf-8").strip()
    except OSError:
        last_message = ""
    if last_message:
        payload["message"] = last_message[:400]
    if completed.returncode != 0:
        payload["skipped"] = "codex /init failed; falling back to direct AGENTS patch"
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        if stderr:
            payload["stderr"] = stderr[-400:]
        if stdout:
            payload["stdout"] = stdout[-400:]
    return payload


def ensure_codex_mcp_registered(*, timeout_s: float = 10.0) -> dict[str, Any]:
    """Ensure the global Codex MCP registry contains a stdio memoryforge entry."""

    binary = shutil.which("codex")
    if not binary:
        return {"ok": False, "skipped": "codex cli not found on PATH", "name": MEMORYFORGE_MCP_NAME}

    get_cmd = [binary, "mcp", "get", MEMORYFORGE_MCP_NAME, "--json"]
    try:
        existing = _run_hidden(get_cmd, timeout_s=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "name": MEMORYFORGE_MCP_NAME,
        }

    if existing.returncode == 0:
        try:
            payload = json.loads(existing.stdout)
        except json.JSONDecodeError:
            payload = {}
        command = str(payload.get("command") or "")
        args = [str(arg) for arg in payload.get("args") or []]
        if command == "uv" and args == ["run", "memoryforge-mcp"]:
            return {"ok": True, "skipped": "already configured", "name": MEMORYFORGE_MCP_NAME}
        remove_cmd = [binary, "mcp", "remove", MEMORYFORGE_MCP_NAME]
        removed = _run_hidden(remove_cmd, timeout_s=timeout_s)
        if removed.returncode != 0:
            return {
                "ok": False,
                "name": MEMORYFORGE_MCP_NAME,
                "returncode": removed.returncode,
                "stderr": (removed.stderr or "").strip()[-400:],
                "message": "failed to remove mismatched MCP registration",
            }

    add_cmd = [binary, "mcp", "add", MEMORYFORGE_MCP_NAME, "--", "uv", "run", "memoryforge-mcp"]
    added = _run_hidden(add_cmd, timeout_s=timeout_s)
    result = {
        "ok": added.returncode == 0,
        "name": MEMORYFORGE_MCP_NAME,
        "returncode": added.returncode,
    }
    if added.returncode != 0:
        stdout = (added.stdout or "").strip()
        stderr = (added.stderr or "").strip()
        if stdout:
            result["stdout"] = stdout[-400:]
        if stderr:
            result["stderr"] = stderr[-400:]
    return result


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

