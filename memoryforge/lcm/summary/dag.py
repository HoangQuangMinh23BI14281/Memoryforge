"""Append-only summary DAG for LCM compaction."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from memoryforge.lcm.compaction.file_ids import (
    append_lossless_footers,
    collect_file_ids_from_nodes,
    collect_source_refs_from_nodes,
    extract_file_ids,
    extract_source_refs,
)


@dataclass
class SummaryNode:
    id: str
    session_id: str
    level: int
    kind: str
    content: str
    span_start: str
    span_end: str
    parent_node_ids: list[str]
    file_ids: list[str]
    source_refs: list[str]
    superseded: bool
    created_at: float


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    is_memory = str(path) == ":memory:"
    if not is_memory:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def _new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000):016x}_{uuid.uuid4().hex[:8]}"


class SummaryDAG:
    """Append-only DAG where condensation supersedes children instead of deleting them."""

    def __init__(self, db_path: str):
        self.conn = _connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_nodes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                level INTEGER NOT NULL DEFAULT 0,
                span_start_message_id TEXT NOT NULL,
                span_end_message_id TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                parent_node_ids TEXT NOT NULL DEFAULT '[]',
                file_ids TEXT NOT NULL DEFAULT '[]',
                source_refs TEXT NOT NULL DEFAULT '[]',
                superseded INTEGER NOT NULL DEFAULT 0,
                kind TEXT NOT NULL DEFAULT 'leaf'
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_summary_session
            ON summary_nodes(session_id, superseded)
            """
        )
        self._ensure_column("summary_nodes", "file_ids", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("summary_nodes", "source_refs", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("summary_nodes", "kind", "TEXT NOT NULL DEFAULT 'leaf'")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def create_leaf(
        self,
        session_id: str,
        content: str,
        span_start: str,
        span_end: str,
        file_ids: list[str] | None = None,
        source_refs: list[str] | None = None,
    ) -> str:
        node_id = _new_id("sum")
        node_file_ids = _dedupe_file_ids([*(file_ids or []), *extract_file_ids(content)])
        node_source_refs = _dedupe_refs([*(source_refs or []), *extract_source_refs(content)])
        stored_content = append_lossless_footers(
            content,
            file_ids=node_file_ids,
            source_refs=node_source_refs,
        )
        self.conn.execute(
            """
            INSERT INTO summary_nodes
            (id, session_id, level, span_start_message_id, span_end_message_id,
             content, token_count, created_at, parent_node_ids, file_ids, source_refs,
             superseded, kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                session_id,
                0,
                span_start,
                span_end,
                stored_content,
                len(stored_content) // 4,
                time.time(),
                "[]",
                json.dumps(node_file_ids),
                json.dumps(node_source_refs),
                0,
                "leaf",
            ),
        )
        self.conn.commit()
        return node_id

    def condense(
        self,
        child_ids: list[str],
        condensed_content: str,
        file_ids: list[str] | None = None,
        source_refs: list[str] | None = None,
    ) -> str:
        if not child_ids:
            raise ValueError("child_ids cannot be empty")

        placeholders = ",".join("?" for _ in child_ids)
        rows = self.conn.execute(
            f"""
            SELECT id, session_id, span_start_message_id, span_end_message_id, level,
                   created_at, content, parent_node_ids, superseded, file_ids, source_refs, kind
            FROM summary_nodes
            WHERE id IN ({placeholders})
            ORDER BY created_at
            """,
            child_ids,
        ).fetchall()
        if len(rows) != len(child_ids):
            found = {row[0] for row in rows}
            missing = [child_id for child_id in child_ids if child_id not in found]
            raise ValueError(f"Unknown child summary nodes: {missing}")

        session_ids = {row[1] for row in rows}
        if len(session_ids) != 1:
            raise ValueError("Cannot condense summary nodes from multiple sessions")

        node_id = _new_id("sum")
        session_id = rows[0][1]
        span_start = rows[0][2]
        span_end = rows[-1][3]
        level = max(int(row[4]) for row in rows) + 1
        child_nodes = [
            self._row_to_node(
                (
                    row[0],
                    row[1],
                    row[4],
                    row[6],
                    row[2],
                    row[3],
                    row[7],
                    row[8],
                    row[5],
                    row[9],
                    row[10],
                    row[11],
                )
            )
            for row in rows
        ]
        node_file_ids = _dedupe_file_ids(
            [
                *collect_file_ids_from_nodes(child_nodes),
                *extract_file_ids(condensed_content),
                *(file_ids or []),
            ]
        )
        node_source_refs = _dedupe_refs(
            [
                *collect_source_refs_from_nodes(child_nodes),
                *extract_source_refs(condensed_content),
                *(source_refs or []),
            ]
        )
        stored_content = append_lossless_footers(
            condensed_content,
            file_ids=node_file_ids,
            source_refs=node_source_refs,
        )

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO summary_nodes
                (id, session_id, level, span_start_message_id, span_end_message_id,
                 content, token_count, created_at, parent_node_ids, file_ids, source_refs,
                 superseded, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    session_id,
                    level,
                    span_start,
                    span_end,
                    stored_content,
                    len(stored_content) // 4,
                    time.time(),
                    json.dumps(child_ids),
                    json.dumps(node_file_ids),
                    json.dumps(node_source_refs),
                    0,
                    "condensed",
                ),
            )
            self.conn.execute(
                f"UPDATE summary_nodes SET superseded = 1 WHERE id IN ({placeholders})",
                child_ids,
            )

        return node_id

    def get_active_summaries(self, session_id: str) -> list[SummaryNode]:
        rows = self.conn.execute(
            """
            SELECT id, session_id, level, content,
                   span_start_message_id, span_end_message_id,
                   parent_node_ids, superseded, created_at, file_ids, source_refs, kind
            FROM summary_nodes
            WHERE session_id = ? AND superseded = 0
            ORDER BY level DESC, created_at DESC
            """,
            (session_id,),
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def get_node(self, node_id: str) -> SummaryNode | None:
        row = self.conn.execute(
            """
            SELECT id, session_id, level, content,
                   span_start_message_id, span_end_message_id,
                   parent_node_ids, superseded, created_at, file_ids, source_refs, kind
            FROM summary_nodes
            WHERE id = ?
            """,
            (node_id,),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def find_by_source_ref(
        self,
        session_id: str,
        source_ref: str,
        *,
        include_superseded: bool = True,
        kind: str | None = None,
        span_start: str | None = None,
        span_end: str | None = None,
    ) -> SummaryNode | None:
        clauses = ["session_id = ?", "source_refs LIKE ?"]
        params: list[str | int] = [session_id, f"%{source_ref}%"]
        if not include_superseded:
            clauses.append("superseded = ?")
            params.append(0)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if span_start is not None:
            clauses.append("span_start_message_id = ?")
            params.append(span_start)
        if span_end is not None:
            clauses.append("span_end_message_id = ?")
            params.append(span_end)
        rows = self.conn.execute(
            f"""
            SELECT id, session_id, level, content,
                   span_start_message_id, span_end_message_id,
                   parent_node_ids, superseded, created_at, file_ids, source_refs, kind
            FROM summary_nodes
            WHERE {" AND ".join(clauses)}
            ORDER BY superseded ASC, created_at DESC
            """,
            params,
        ).fetchall()
        for row in rows:
            node = self._row_to_node(row)
            if source_ref in node.source_refs:
                return node
        return None

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _row_to_node(row: sqlite3.Row | tuple) -> SummaryNode:
        try:
            return SummaryNode(
                id=str(row[0]),
                session_id=str(row[1]),
                level=int(row[2]),
                kind=str(row[11]) if len(row) > 11 and row[11] else "leaf",
                content=str(row[3]),
                span_start=str(row[4]),
                span_end=str(row[5]),
                parent_node_ids=json.loads(row[6]) if row[6] else [],
                superseded=bool(row[7]),
                created_at=float(row[8]),
                file_ids=json.loads(row[9]) if len(row) > 9 and row[9] else [],
                source_refs=json.loads(row[10]) if len(row) > 10 and row[10] else [],
            )
        except (IndexError, ValueError, json.JSONDecodeError) as e:
            import sys

            print(f"Warning: Failed to unpack summary node row: {e}", file=sys.stderr)
            raise ValueError(f"Invalid summary node row structure: {e}") from e


def _dedupe_file_ids(file_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for file_id in file_ids:
        if file_id not in seen:
            seen.add(file_id)
            deduped.append(file_id)
    return deduped


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for ref in refs:
        if ref and ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped
