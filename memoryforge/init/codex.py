"""Codex AGENTS.md integration writers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

MEMORYFORGE_AGENTS_MARKER_START = "<!-- MemoryForge instructions start -->"
MEMORYFORGE_AGENTS_MARKER_END = "<!-- MemoryForge instructions end -->"

MEMORYFORGE_AGENTS_CONTENT = """# MemoryForge Project Memory

Use MemoryForge MCP as the first path for repository memory questions about docs, decisions, setup, schema, architecture, roadmap, prior conversation, or long project context.

- Do not start routine fact lookups with `ensure_project_memory(auto_index=true)`.
- Call `ensure_project_memory` only to verify that the project is initialized or when MemoryForge reports missing project state.
- Call `autoload_markdown` only when Markdown may have changed or when recall returns stale or empty evidence.
- Use `recall_memory` first for factual project-memory questions.
- Use `build_context_bundle` when the answer needs grounded multi-source context for the core model.
- Use `rlm_load`, `rlm_search`, and `rlm_chunk_get` for large files or documents instead of reading everything into the prompt.
- Use `rlm_run` only when the task needs real Codex CLI sub-agent analysis over large context.
- Fall back to `rg`, `Get-Content`, or direct file reads only if MemoryForge MCP is unavailable or returns no relevant evidence.
- Prefer MemoryForge provenance over raw workspace grep when both are available.
- The project-local database is `.memoryforge/memory.db` unless `MEMORYFORGE_DB` explicitly overrides it.
"""


def run_codex_init(root: Path, *, timeout_s: float = 180.0) -> dict[str, Any]:
    """Ask Codex CLI to create/update the project AGENTS.md via its /init flow."""

    binary = shutil.which("codex")
    if not binary:
        return {
            "ok": False,
            "skipped": "codex cli not found on PATH",
            "agents_path": str(root / "AGENTS.md"),
        }
    command = [
        binary,
        "exec",
        "--cd",
        str(root),
        "--skip-git-repo-check",
        "/init",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "agents_path": str(root / "AGENTS.md"),
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "agents_path": str(root / "AGENTS.md"),
    }


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
        end = existing.index(MEMORYFORGE_AGENTS_MARKER_END) + len(
            MEMORYFORGE_AGENTS_MARKER_END
        )
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
