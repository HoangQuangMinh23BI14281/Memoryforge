"""Long-term memory utility helpers."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def tokenize_query(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[\w]+", query.lower()):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}_{digest}"


def preview(text: str, max_chars: int = 420) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"
