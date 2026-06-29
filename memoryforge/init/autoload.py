"""Project autoload helpers for Markdown RLM/LTM indexing."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from memoryforge.api import MemoryForge

DEFAULT_MARKDOWN_GLOBS = ("*.md", "*.markdown")
SKIP_FILE_NAMES = {"AGENTS.md"}

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".memoryforge",
    ".codex",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}

AUTOLOAD_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS autoload_files (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL,
    buffer_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_autoload_files_sha256 ON autoload_files(sha256);
CREATE INDEX IF NOT EXISTS idx_autoload_files_indexed_at ON autoload_files(indexed_at);
"""


def ensure_autoload_schema(db_path: str) -> None:
    path = Path(db_path).expanduser()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(AUTOLOAD_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def index_project_markdown(
    *,
    db_path: str,
    agent_id: str,
    project_root: str,
    chunk_size: int = 12_000,
    overlap: int = 1_000,
    max_files: int = 200,
    max_file_bytes: int = 1_000_000,
    min_file_bytes: int = 0,
) -> dict[str, Any]:
    """Index project Markdown files into RLM buffers and LTM/vector recall.

    This is intentionally indexing-only: it does not spawn Codex subagents for
    every project document. Deep model analysis remains available through
    explicit `memoryforge index --analyze` host-subagent plans.
    """

    root = Path(project_root).expanduser().resolve()
    files, discovery = _discover_markdown_files_with_stats(
        root,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        min_file_bytes=min_file_bytes,
    )
    result: dict[str, Any] = {
        "enabled": True,
        "files": 0,
        "chunks": 0,
        "long_term_items": 0,
        "deduped_files": 0,
        "unchanged_files": 0,
        "skipped_files": 0,
        **discovery,
        "errors": [],
    }
    ensure_autoload_schema(db_path)
    if not files:
        return result
    _progress(_discovery_progress("autoload", root, discovery))
    mf = MemoryForge(db_path)
    try:
        for index, path in enumerate(files, start=1):
            relative_path = path.relative_to(root).as_posix()
            try:
                fingerprint = _file_fingerprint(path)
                if _autoload_file_unchanged(db_path, relative_path, fingerprint, agent_id=agent_id):
                    result["unchanged_files"] += 1
                    _progress(f"autoload: [{index}/{len(files)}] unchanged {relative_path}")
                    continue
                _progress(f"autoload: [{index}/{len(files)}] indexing {relative_path}")
                loaded = mf.rlm.load(
                    agent_id=agent_id,
                    value=path,
                    name=relative_path,
                    source_path=str(path),
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                item_ids = mf.long_term.index_rlm_buffer(agent_id, str(loaded["buffer_id"]))
                _record_autoload_file(
                    db_path,
                    relative_path,
                    fingerprint,
                    buffer_id=str(loaded["buffer_id"]),
                )
                result["files"] += 1
                result["chunks"] += int(loaded.get("chunk_count") or 0)
                result["long_term_items"] += len(item_ids)
                if loaded.get("deduped"):
                    result["deduped_files"] += 1
            except Exception as exc:  # pragma: no cover - surfaced in returned diagnostics
                result["errors"].append({"path": relative_path, "error": str(exc)})
                _progress(f"autoload: [{index}/{len(files)}] error {relative_path}: {exc}")
        result["skipped_files"] = (
            max(0, len(files) - int(result["files"]) - int(result["unchanged_files"]))
            + int(discovery["skipped_by_min_file_bytes"])
            + int(discovery["skipped_by_max_files"])
        )
        _progress(
            "autoload: done files={files} chunks={chunks} long_term_items={items} unchanged={unchanged}".format(
                files=result["files"],
                chunks=result["chunks"],
                items=result["long_term_items"],
                unchanged=result["unchanged_files"],
            )
        )
        return result
    finally:
        mf.close()


def index_project_markdown_with_rlm(
    *,
    db_path: str,
    agent_id: str,
    project_root: str,
    chunk_size: int = 12_000,
    overlap: int = 1_000,
    max_files: int = 200,
    max_file_bytes: int = 1_000_000,
    min_file_bytes: int = 0,
    runner: str | None = "auto",
    model: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 900.0,
    limit: int = 10_000,
    batch_size: int | None = None,
    max_workers: int = 1,
    max_retries: int = 0,
    allow_partial: bool = False,
    synthesize: bool = True,
    recursive: bool = True,
    max_recursive_rounds: int = 2,
    recursive_token_limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Deprecated compatibility wrapper for host-managed RLM analysis plans.

    Older versions spawned configured sub-agent runners from this function.
    Current MemoryForge does not spawn `codex exec` from project indexing; it
    returns the same host-subagent plan as ``plan_project_markdown_analysis``.
    External runner options are preserved in the returned diagnostics only.
    """

    result = plan_project_markdown_analysis(
        db_path=db_path,
        agent_id=agent_id,
        project_root=project_root,
        chunk_size=chunk_size,
        overlap=overlap,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        min_file_bytes=min_file_bytes,
        limit=limit,
        batch_size=batch_size,
        force=force,
    )
    result["deprecated_wrapper"] = "index_project_markdown_with_rlm"
    result["ignored_external_runner_options"] = {
        "runner": runner,
        "model": model,
        "base_url_configured": bool(base_url),
        "timeout_s": timeout_s,
        "max_workers": max_workers,
        "max_retries": max_retries,
        "allow_partial": allow_partial,
        "synthesize": synthesize,
        "recursive": recursive,
        "max_recursive_rounds": max_recursive_rounds,
        "recursive_token_limit": recursive_token_limit,
    }
    return result


def plan_project_markdown_analysis(
    *,
    db_path: str,
    agent_id: str,
    project_root: str,
    chunk_size: int = 12_000,
    overlap: int = 1_000,
    max_files: int = 200,
    max_file_bytes: int = 1_000_000,
    min_file_bytes: int = 0,
    limit: int = 10_000,
    batch_size: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare host-managed RLM sub-agent work without spawning a model process.

    This mirrors the rlm-rs style: MemoryForge owns lossless chunks, dispatch
    plans, and result recording; the host agent that is already running owns
    actual sub-agent execution.
    """

    root = Path(project_root).expanduser().resolve()
    files, discovery = _discover_markdown_files_with_stats(
        root,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        min_file_bytes=min_file_bytes,
    )
    result: dict[str, Any] = {
        "enabled": True,
        "mode": "host_subagent_plan",
        "external_model_calls": False,
        "files": 0,
        "chunks": 0,
        "source_long_term_items": 0,
        "planned_batches": 0,
        "deduped_files": 0,
        "unchanged_files": 0,
        "loaded_files": 0,
        "skipped_files": 0,
        **discovery,
        "errors": [],
        "plans": [],
        "next_steps": [
            "Ask the active Codex host to run one subagent per batch in parallel.",
            "Each host subagent fetches chunks with the batch fetch_command_argv values.",
            "Record each finished batch with the batch record_command_argv value.",
            "Run the plan aggregate_command_argv only after all batch records are stored.",
        ],
    }
    ensure_autoload_schema(db_path)
    if not files:
        return result
    _progress(_discovery_progress("analyze-plan", root, discovery))
    mf = MemoryForge(db_path)
    try:
        for index, path in enumerate(files, start=1):
            relative_path = path.relative_to(root).as_posix()
            try:
                fingerprint = _file_fingerprint(path)
                buffer_id = (
                    None
                    if force
                    else _autoload_buffer_id(
                        db_path,
                        relative_path,
                        fingerprint,
                        agent_id=agent_id,
                    )
                )
                loaded: dict[str, Any] | None = None
                source_item_count = 0
                if buffer_id:
                    result["unchanged_files"] += 1
                    _progress(f"analyze-plan: [{index}/{len(files)}] dispatch existing {relative_path}")
                else:
                    _progress(f"analyze-plan: [{index}/{len(files)}] indexing {relative_path}")
                    loaded = mf.rlm.load(
                        agent_id=agent_id,
                        value=path,
                        name=relative_path,
                        source_path=str(path),
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                    buffer_id = str(loaded["buffer_id"])
                    item_ids = mf.long_term.index_rlm_buffer(agent_id, buffer_id)
                    source_item_count = len(item_ids)
                    _record_autoload_file(
                        db_path,
                        relative_path,
                        fingerprint,
                        buffer_id=buffer_id,
                    )
                    result["loaded_files"] += 1
                    result["source_long_term_items"] += source_item_count
                    if loaded.get("deduped"):
                        result["deduped_files"] += 1

                plan = mf.rlm_dispatch(
                    agent_id=agent_id,
                    buffer_id=buffer_id,
                    limit=limit,
                    batch_size=batch_size,
                )
                host_plan = _host_subagent_plan(
                    db_path=db_path,
                    agent_id=agent_id,
                    project_root=str(root),
                    relative_path=relative_path,
                    plan=plan,
                )
                result["files"] += 1
                result["chunks"] += int(plan.get("chunk_count") or 0)
                result["planned_batches"] += int(plan.get("batch_count") or 0)
                result["plans"].append(
                    {
                        "path": relative_path,
                        "buffer_id": buffer_id,
                        "loaded": loaded is not None,
                        "source_long_term_items": source_item_count,
                        "chunk_count": int(plan.get("chunk_count") or 0),
                        "batch_count": int(plan.get("batch_count") or 0),
                        "expected_batch_count": int(plan.get("batch_count") or 0),
                        "run_id": plan["run_id"],
                        "aggregate_command": host_plan["aggregate_command"],
                        "aggregate_command_argv": host_plan["aggregate_command_argv"],
                        "completion_contract": host_plan["completion_contract"],
                        "batches": host_plan["batches"],
                    }
                )
            except Exception as exc:  # pragma: no cover - surfaced in returned diagnostics
                result["errors"].append({"path": relative_path, "error": str(exc)})
                _progress(f"analyze-plan: [{index}/{len(files)}] error {relative_path}: {exc}")
        result["skipped_files"] = (
            max(0, len(files) - int(result["files"]))
            + int(discovery["skipped_by_min_file_bytes"])
            + int(discovery["skipped_by_max_files"])
        )
        _progress(
            "analyze-plan: done files={files} chunks={chunks} batches={batches} unchanged={unchanged}".format(
                files=result["files"],
                chunks=result["chunks"],
                batches=result["planned_batches"],
                unchanged=result["unchanged_files"],
            )
        )
        return result
    finally:
        mf.close()


def _discover_markdown_files(
    root: Path,
    *,
    max_files: int,
    max_file_bytes: int,
    min_file_bytes: int = 0,
) -> list[Path]:
    return _discover_markdown_files_with_stats(
        root,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        min_file_bytes=min_file_bytes,
    )[0]


def _discover_markdown_files_with_stats(
    root: Path,
    *,
    max_files: int,
    max_file_bytes: int,
    min_file_bytes: int = 0,
) -> tuple[list[Path], dict[str, Any]]:
    files: list[Path] = []
    eligible_files = 0
    skipped_by_min_file_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".markdown"}:
            continue
        if _is_skipped(path, root):
            continue
        try:
            size = path.stat().st_size
            if size > max_file_bytes:
                continue
        except OSError:
            continue
        if size < min_file_bytes:
            skipped_by_min_file_bytes += 1
            continue
        eligible_files += 1
        if len(files) < max_files:
            files.append(path)
    skipped_by_max_files = max(0, eligible_files - len(files))
    return files, {
        "eligible_files": eligible_files,
        "selected_files": len(files),
        "max_files": max_files,
        "max_file_bytes": max_file_bytes,
        "min_file_bytes": min_file_bytes,
        "skipped_by_min_file_bytes": skipped_by_min_file_bytes,
        "skipped_by_max_files": skipped_by_max_files,
        "limited_by_max_files": skipped_by_max_files > 0,
    }


def _discovery_progress(label: str, root: Path, discovery: dict[str, Any]) -> str:
    selected = int(discovery["selected_files"])
    eligible = int(discovery["eligible_files"])
    max_files = int(discovery["max_files"])
    min_file_bytes = int(discovery.get("min_file_bytes") or 0)
    min_suffix = f", min_file_bytes={min_file_bytes}" if min_file_bytes else ""
    if discovery.get("limited_by_max_files"):
        return (
            f"{label}: selected {selected}/{eligible} markdown files under {root} "
            f"(max_files={max_files}{min_suffix})"
        )
    if min_suffix:
        return f"{label}: selected {selected}/{eligible} markdown files under {root} ({min_suffix.lstrip(', ')})"
    return f"{label}: selected {selected} markdown files under {root}"


def _is_skipped(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return relative.name in SKIP_FILE_NAMES or any(part in SKIP_DIR_NAMES for part in relative.parts[:-1])


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "size": int(stat.st_size), "mtime": float(stat.st_mtime)}


def _autoload_file_unchanged(
    db_path: str,
    relative_path: str,
    fingerprint: dict[str, Any],
    *,
    agent_id: str,
    require_worker: bool = False,
) -> bool:
    conn = sqlite3.connect(str(Path(db_path).expanduser()))
    try:
        conn.executescript(AUTOLOAD_SCHEMA_SQL)
        row = conn.execute(
            "SELECT sha256, size, mtime, buffer_id FROM autoload_files WHERE path = ?",
            (relative_path,),
        ).fetchone()
        if row is None:
            return False
        fingerprint_matches = (
            str(row[0]) == str(fingerprint["sha256"])
            and int(row[1]) == int(fingerprint["size"])
            and float(row[2]) == float(fingerprint["mtime"])
        )
        if not fingerprint_matches:
            return False
        buffer_id = str(row[3] or "")
        if not buffer_id:
            return False
        buffer_row = conn.execute(
            "SELECT agent_id FROM rlm_buffers WHERE buffer_id = ?",
            (buffer_id,),
        ).fetchone()
        if buffer_row is None or str(buffer_row[0]) != agent_id:
            return False
        item_row = conn.execute(
            "SELECT 1 FROM long_term_items WHERE agent_id = ? AND source_type = 'rlm_chunk' AND source_id IN (SELECT chunk_id FROM rlm_chunks WHERE buffer_id = ?) LIMIT 1",
            (agent_id, buffer_id),
        ).fetchone()
        if item_row is None:
            return False
        if require_worker:
            return _buffer_has_worker_items(conn, agent_id=agent_id, buffer_id=buffer_id)
        return True
    finally:
        conn.close()


def _autoload_buffer_id(
    db_path: str,
    relative_path: str,
    fingerprint: dict[str, Any],
    *,
    agent_id: str,
) -> str | None:
    conn = sqlite3.connect(str(Path(db_path).expanduser()))
    try:
        conn.executescript(AUTOLOAD_SCHEMA_SQL)
        row = conn.execute(
            "SELECT sha256, size, mtime, buffer_id FROM autoload_files WHERE path = ?",
            (relative_path,),
        ).fetchone()
        if row is None:
            return None
        fingerprint_matches = (
            str(row[0]) == str(fingerprint["sha256"])
            and int(row[1]) == int(fingerprint["size"])
            and float(row[2]) == float(fingerprint["mtime"])
        )
        if not fingerprint_matches:
            return None
        buffer_id = str(row[3] or "")
        if not buffer_id:
            return None
        buffer_row = conn.execute(
            "SELECT agent_id FROM rlm_buffers WHERE buffer_id = ?",
            (buffer_id,),
        ).fetchone()
        if buffer_row is None or str(buffer_row[0]) != agent_id:
            return None
        item_row = conn.execute(
            "SELECT 1 FROM long_term_items WHERE agent_id = ? AND source_type = 'rlm_chunk' AND source_id IN (SELECT chunk_id FROM rlm_chunks WHERE buffer_id = ?) LIMIT 1",
            (agent_id, buffer_id),
        ).fetchone()
        return buffer_id if item_row is not None else None
    finally:
        conn.close()


def _buffer_has_worker_items(conn: sqlite3.Connection, *, agent_id: str, buffer_id: str) -> bool:
    rows = conn.execute(
        "SELECT chunk_id FROM rlm_chunks WHERE buffer_id = ? LIMIT 50",
        (buffer_id,),
    ).fetchall()
    for row in rows:
        chunk_ref = f"rlm_chunk:{row[0]}"
        item_row = conn.execute(
            """
            SELECT 1
            FROM long_term_items
            WHERE agent_id = ?
              AND source_type IN ('rlm_summary', 'rlm_analysis')
              AND metadata LIKE ?
            LIMIT 1
            """,
            (agent_id, f"%{chunk_ref}%"),
        ).fetchone()
        if item_row is not None:
            return True
    return False


def _record_autoload_file(
    db_path: str,
    relative_path: str,
    fingerprint: dict[str, Any],
    *,
    buffer_id: str,
) -> None:
    conn = sqlite3.connect(str(Path(db_path).expanduser()))
    try:
        conn.executescript(AUTOLOAD_SCHEMA_SQL)
        conn.execute(
            """
            INSERT INTO autoload_files
            (path, sha256, size, mtime, indexed_at, buffer_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                sha256 = excluded.sha256,
                size = excluded.size,
                mtime = excluded.mtime,
                indexed_at = excluded.indexed_at,
                buffer_id = excluded.buffer_id
            """,
            (
                relative_path,
                str(fingerprint["sha256"]),
                int(fingerprint["size"]),
                float(fingerprint["mtime"]),
                time.time(),
                buffer_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _progress(message: str) -> None:
    if os.environ.get("MEMORYFORGE_PROGRESS") == "0":
        return
    if not sys.stderr:
        return
    if os.environ.get("MEMORYFORGE_PROGRESS") == "1" or sys.stderr.isatty():
        print(f"[memoryforge] {message}", file=sys.stderr, flush=True)


def _host_subagent_plan(
    *,
    db_path: str,
    agent_id: str,
    project_root: str,
    relative_path: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    batches = []
    expected_batch_count = int(plan.get("batch_count") or 0)
    for batch in plan.get("batches", []):
        chunk_ids = [str(chunk_id) for chunk_id in batch.get("chunk_ids", [])]
        batch_index = int(batch.get("batch_index", 0))
        fetch_command_argvs = [
            _memoryforge_argv(db_path, "rlm-chunk-get", [chunk_id]) for chunk_id in chunk_ids
        ]
        record_command_argv = _record_command_argv(
            db_path=db_path,
            agent_id=agent_id,
            run_id=str(plan["run_id"]),
            batch_index=batch_index,
            chunk_ids=chunk_ids,
        )
        batches.append(
            {
                "batch_index": batch_index,
                "chunk_ids": chunk_ids,
                "fetch_commands": [_argv_to_command(argv) for argv in fetch_command_argvs],
                "fetch_command_argvs": fetch_command_argvs,
                "record_command": _argv_to_command([*record_command_argv, "<analysis.md>"]),
                "record_command_argv": [*record_command_argv, "<analysis.md>"],
                "analysis_file_placeholder": "<analysis.md>",
                "host_subagent_prompt": _host_subagent_prompt(
                    db_path=db_path,
                    agent_id=agent_id,
                    project_root=project_root,
                    relative_path=relative_path,
                    run_id=str(plan["run_id"]),
                    batch_index=batch_index,
                    expected_batch_count=expected_batch_count,
                    chunk_ids=chunk_ids,
                    fetch_command_argvs=fetch_command_argvs,
                    record_command_argv=[*record_command_argv, "<analysis.md>"],
                ),
            }
        )
    aggregate_command_argv = _aggregate_command_argv(
        db_path=db_path,
        agent_id=agent_id,
        run_id=str(plan["run_id"]),
        expected_batch_count=expected_batch_count,
    )
    return {
        "aggregate_command": _argv_to_command(aggregate_command_argv),
        "aggregate_command_argv": aggregate_command_argv,
        "completion_contract": {
            "run_id": str(plan["run_id"]),
            "expected_batch_count": expected_batch_count,
            "aggregate_after_recorded_batches": expected_batch_count,
            "idempotency_key": "run_id + batch_index",
        },
        "batches": batches,
    }


def _memoryforge_argv(db_path: str, subcommand: str, extra_args: list[str]) -> list[str]:
    return ["uv", "run", "memoryforge", "--db", db_path, subcommand, *extra_args]


def _record_command_argv(
    *,
    db_path: str,
    agent_id: str,
    run_id: str,
    batch_index: int,
    chunk_ids: list[str],
) -> list[str]:
    args = ["--agent-id", agent_id, "--run-id", run_id, "--batch-index", str(batch_index)]
    for chunk_id in chunk_ids:
        args.extend(["--chunk-id", chunk_id])
    args.extend(["--analysis-file"])
    return _memoryforge_argv(db_path, "rlm-record", args)


def _aggregate_command_argv(
    *,
    db_path: str,
    agent_id: str,
    run_id: str,
    expected_batch_count: int,
) -> list[str]:
    return _memoryforge_argv(
        db_path,
        "aggregate",
        [
            "--agent-id",
            agent_id,
            "--run-id",
            run_id,
            "--expected-batches",
            str(expected_batch_count),
        ],
    )


def _host_subagent_prompt(
    *,
    db_path: str,
    agent_id: str,
    project_root: str,
    relative_path: str,
    run_id: str,
    batch_index: int,
    expected_batch_count: int,
    chunk_ids: list[str],
    fetch_command_argvs: list[list[str]],
    record_command_argv: list[str],
) -> str:
    chunk_lines = "\n".join(f"- {chunk_id}" for chunk_id in chunk_ids)
    fetch_lines = "\n".join(_argv_to_command(argv) for argv in fetch_command_argvs)
    return "\n".join(
        [
            "Analyze this MemoryForge RLM batch using only the listed chunk refs.",
            "",
            f"Project root: {project_root}",
            f"Source file: {relative_path}",
            f"Agent ID: {agent_id}",
            f"Run ID: {run_id}",
            f"Batch index: {batch_index}",
            f"Expected batches for this run: {expected_batch_count}",
            "",
            "Chunk IDs:",
            chunk_lines,
            "",
            "Fetch commands:",
            fetch_lines,
            "",
            "Completion contract:",
            "- Return concise findings with explicit rlm_chunk:<id> citations.",
            "- Preserve exact IDs, paths, decisions, constraints, and unresolved questions.",
            "- Do not invent facts outside the fetched chunks.",
            "- Save the analysis to a local Markdown file and record it with:",
            _argv_to_command(record_command_argv),
        ]
    ).strip()


def _argv_to_command(argv: list[str]) -> str:
    if os.name == "nt":
        import subprocess

        return subprocess.list2cmdline(argv)
    import shlex

    return shlex.join(argv)
