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
    conn = sqlite3.connect(str(Path(db_path).expanduser()))
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
) -> dict[str, Any]:
    """Index project Markdown files into RLM buffers and LTM/vector recall.

    This is intentionally indexing-only: it does not spawn Codex subagents for
    every project document. Deep model analysis remains available through
    rlm_run/MCP rlm_run when a user or agent asks for it.
    """

    root = Path(project_root).expanduser().resolve()
    files = _discover_markdown_files(root, max_files=max_files, max_file_bytes=max_file_bytes)
    result: dict[str, Any] = {
        "enabled": True,
        "files": 0,
        "chunks": 0,
        "long_term_items": 0,
        "deduped_files": 0,
        "unchanged_files": 0,
        "skipped_files": 0,
        "errors": [],
    }
    ensure_autoload_schema(db_path)
    if not files:
        return result
    _progress(f"autoload: discovered {len(files)} markdown files under {root}")
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
        result["skipped_files"] = max(
            0, len(files) - int(result["files"]) - int(result["unchanged_files"])
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


def _discover_markdown_files(
    root: Path,
    *,
    max_files: int,
    max_file_bytes: int,
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".markdown"}:
            continue
        if _is_skipped(path, root):
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        files.append(path)
    return files


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
        return item_row is not None
    finally:
        conn.close()


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
