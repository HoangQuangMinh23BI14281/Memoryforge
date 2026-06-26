"""Shared helpers for the RLM subsystem."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
import uuid
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    is_memory = str(path) == ":memory:"
    if not is_memory:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def new_id(prefix: str) -> str:
    timestamp_ns = time.time_ns()
    return f"{prefix}_{timestamp_ns:020x}_{uuid.uuid4().hex[:12]}"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(query: str) -> list[str]:
    return re.findall(r"[\w]+", query.lower())


def is_transient_sqlite_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "locked" in message or "busy" in message or "disk i/o" in message


def preview(text: str, limit: int = 240) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
