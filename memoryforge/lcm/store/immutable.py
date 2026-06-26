"""Append-only message-part store for lossless LCM context."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memoryforge._core import ContentHashTable
from memoryforge.db.schema import (
    LCM_SCHEMA_SQL,
    ensure_lcm_message_part_columns,
    ensure_lcm_session_columns,
)
from memoryforge.lcm.tokens import TokenEstimator


@dataclass(frozen=True)
class MessagePart:
    id: str
    message_id: str
    session_id: str
    part_type: str
    content: str
    content_id: str
    part_index: int
    token_estimate: int
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_state: str | None = None
    is_protected: bool = False
    compacted_at: int | None = None


@dataclass(frozen=True)
class StoredMessage:
    id: str
    session_id: str
    agent_id: str
    role: str
    created_at: int
    is_summary: bool
    parts: list[MessagePart]

    @property
    def content(self) -> str:
        return "\n".join(part.content for part in self.parts if part.part_type == "text")

    @property
    def token_estimate(self) -> int:
        return sum(part.token_estimate for part in self.parts)


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    return conn


def _new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000):016x}_{uuid.uuid4().hex[:8]}"


def _now_ms() -> int:
    return int(time.time() * 1000)


class ImmutableMessageStore:
    """Append-only store: compaction marks parts, but raw content remains recoverable."""

    def __init__(self, db_path: str, *, estimator: TokenEstimator | None = None):
        self.db_path = str(Path(db_path).expanduser())
        self.conn = _connect(self.db_path)
        self.estimator = estimator or TokenEstimator(heuristic_only=True)
        self.content = ContentHashTable(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(LCM_SCHEMA_SQL)
        ensure_lcm_message_part_columns(self.conn)
        ensure_lcm_session_columns(self.conn)
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        valid_tables = {"sessions", "messages", "message_parts", "summary_nodes", "context_items"}
        if table not in valid_tables:
            raise ValueError(f"Invalid table name: {table}")
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            allowed_types = {"TEXT", "INTEGER", "REAL", "BLOB"}
            ddl_upper = ddl.upper()
            if not any(ddl_upper.startswith(t) for t in allowed_types):
                raise ValueError(f"Invalid DDL type: {ddl}")
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self.conn.commit()

    def ensure_session(
        self,
        agent_id: str,
        session_id: str,
        *,
        system_prompt: str = "",
        model_id: str = "",
        provider_id: str = "",
    ) -> None:
        timestamp = _now_ms()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO sessions
                (id, agent, system_prompt, model_id, provider_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    agent = excluded.agent,
                    system_prompt = CASE
                        WHEN excluded.system_prompt != '' THEN excluded.system_prompt
                        ELSE sessions.system_prompt
                    END,
                    model_id = CASE
                        WHEN excluded.model_id != '' THEN excluded.model_id
                        ELSE sessions.model_id
                    END,
                    provider_id = CASE
                        WHEN excluded.provider_id != '' THEN excluded.provider_id
                        ELSE sessions.provider_id
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    agent_id,
                    system_prompt,
                    model_id,
                    provider_id,
                    timestamp,
                    timestamp,
                ),
            )

    def append_text_message(
        self,
        agent_id: str,
        session_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        is_summary: bool = False,
        created_at_ms: int | None = None,
    ) -> str:
        return self.append_message(
            agent_id,
            session_id,
            role,
            [{"part_type": "text", "content": content}],
            message_id=message_id,
            is_summary=is_summary,
            created_at_ms=created_at_ms,
        )

    def append_message(
        self,
        agent_id: str,
        session_id: str,
        role: str,
        parts: list[dict[str, Any]],
        *,
        message_id: str | None = None,
        is_summary: bool = False,
        created_at_ms: int | None = None,
    ) -> str:
        self.ensure_session(agent_id, session_id)
        normalized_role = role if role in {"user", "assistant"} else "assistant"
        resolved_message_id = message_id or _new_id("msg")
        timestamp = created_at_ms if created_at_ms is not None else _now_ms()
        prepared_parts = []
        total_tokens = 0
        for index, part in enumerate(parts):
            content = str(part.get("content", ""))
            content_id, _is_new = self.content.store(content)
            token_count = int(part.get("token_estimate") or self.estimator.estimate(content))
            total_tokens += token_count
            prepared_parts.append((index, part, content, content_id, token_count))
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO messages
                (id, session_id, role, created_at, agent, is_summary, tokens_total, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_message_id,
                    session_id,
                    normalized_role,
                    timestamp,
                    agent_id,
                    1 if is_summary else 0,
                    0,
                    "summary" if is_summary else "conversation",
                ),
            )
            for index, part, content, content_id, token_count in prepared_parts:
                protected = part.get("protected") or part.get("is_protected")
                is_protected = 1 if (protected and protected not in (0, False, "")) else 0
                self.conn.execute(
                    """
                    INSERT INTO message_parts
                    (id, message_id, session_id, part_type, part_index, content_id, content,
                     tool_name, tool_call_id, tool_state, is_protected, token_estimate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(part.get("id") or _new_id("part")),
                        resolved_message_id,
                        session_id,
                        str(part.get("part_type", "text")),
                        index,
                        content_id,
                        content,
                        part.get("tool_name"),
                        part.get("tool_call_id"),
                        part.get("tool_state"),
                        is_protected,
                        token_count,
                    ),
                )
            self.conn.execute(
                "UPDATE messages SET tokens_total = ? WHERE id = ?",
                (total_tokens, resolved_message_id),
            )
            if not is_summary:
                self._append_context_item(session_id, "message", resolved_message_id)
        return resolved_message_id

    def get_messages(
        self, session_id: str, *, include_summaries: bool = True
    ) -> list[StoredMessage]:
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if not include_summaries:
            clauses.append("is_summary = 0")
        rows = self.conn.execute(
            f"""
            SELECT id, session_id, agent, role, created_at, is_summary
            FROM messages
            WHERE {" AND ".join(clauses)}
            ORDER BY created_at, id
            """,
            params,
        ).fetchall()
        parts_by_message = self._parts_for_messages([str(row[0]) for row in rows])
        return [
            StoredMessage(
                id=str(row[0]),
                session_id=str(row[1]),
                agent_id=str(row[2]),
                role=str(row[3]),
                created_at=int(row[4]),
                is_summary=bool(row[5]),
                parts=parts_by_message.get(str(row[0]), []),
            )
            for row in rows
        ]

    def get_messages_by_ids(self, message_ids: list[str]) -> list[StoredMessage]:
        if not message_ids:
            return []
        placeholders = ",".join("?" for _ in message_ids)
        rows = self.conn.execute(
            f"""
            SELECT id, session_id, agent, role, created_at, is_summary
            FROM messages
            WHERE id IN ({placeholders})
            """,
            message_ids,
        ).fetchall()
        parts_by_message = self._parts_for_messages([str(row[0]) for row in rows])
        by_id = {
            str(row[0]): StoredMessage(
                id=str(row[0]),
                session_id=str(row[1]),
                agent_id=str(row[2]),
                role=str(row[3]),
                created_at=int(row[4]),
                is_summary=bool(row[5]),
                parts=parts_by_message.get(str(row[0]), []),
            )
            for row in rows
        }
        return [by_id[message_id] for message_id in message_ids if message_id in by_id]

    def get_message(self, message_id: str) -> StoredMessage | None:
        row = self.conn.execute(
            """
            SELECT id, session_id, agent, role, created_at, is_summary
            FROM messages
            WHERE id = ?
            """,
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        parts = self._parts_for_messages([message_id]).get(message_id, [])
        return StoredMessage(
            id=str(row[0]),
            session_id=str(row[1]),
            agent_id=str(row[2]),
            role=str(row[3]),
            created_at=int(row[4]),
            is_summary=bool(row[5]),
            parts=parts,
        )

    def get_context_items(self, session_id: str) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            """
            SELECT item_type, item_id
            FROM context_items
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def swap_context_items(
        self,
        session_id: str,
        old_item_ids: list[str],
        summary_node_id: str,
    ) -> None:
        if not old_item_ids:
            return
        placeholders = ",".join("?" for _ in old_item_ids)
        with self.conn:
            row = self.conn.execute(
                f"""
                SELECT MIN(position)
                FROM context_items
                WHERE session_id = ? AND item_id IN ({placeholders})
                """,
                [session_id, *old_item_ids],
            ).fetchone()
            if row is None or row[0] is None:
                return
            position = int(row[0])
            self.conn.execute(
                f"""
                DELETE FROM context_items
                WHERE session_id = ? AND item_id IN ({placeholders})
                """,
                [session_id, *old_item_ids],
            )
            self.conn.execute(
                """
                INSERT INTO context_items (session_id, item_type, item_id, position, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, "summary", summary_node_id, position, str(_now_ms())),
            )

    def _append_context_item(self, session_id: str, item_type: str, item_id: str) -> None:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM context_items WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        position = int(row[0]) if row else 1
        self.conn.execute(
            """
            INSERT INTO context_items (session_id, item_type, item_id, position, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, item_type, item_id, position, str(_now_ms())),
        )

    def mark_parts_compacted(self, part_ids: list[str], *, timestamp_ms: int | None = None) -> int:
        if not part_ids:
            return 0
        timestamp = timestamp_ms or _now_ms()
        placeholders = ",".join("?" for _ in part_ids)
        with self.conn:
            cursor = self.conn.execute(
                f"UPDATE message_parts SET compacted_at = ? WHERE id IN ({placeholders}) AND compacted_at IS NULL",
                [timestamp, *part_ids],
            )
        return int(cursor.rowcount)

    def session_token_count(self, session_id: str, *, include_summaries: bool = False) -> int:
        clauses = ["session_id = ?"]
        params: list[Any] = [session_id]
        if not include_summaries:
            clauses.append("is_summary = 0")
        row = self.conn.execute(
            f"""
            SELECT COALESCE(SUM(tokens_total), 0)
            FROM messages
            WHERE {" AND ".join(clauses)}
            """,
            params,
        ).fetchone()
        return int(row[0] or 0)

    def _parts_for_messages(self, message_ids: list[str]) -> dict[str, list[MessagePart]]:
        if not message_ids:
            return {}
        placeholders = ",".join("?" for _ in message_ids)
        rows = self.conn.execute(
            f"""
            SELECT id, message_id, session_id, part_type, content, part_index,
                   token_estimate, tool_name, tool_call_id, tool_state, is_protected, compacted_at,
                   content_id
            FROM message_parts
            WHERE message_id IN ({placeholders})
            ORDER BY message_id, part_index
            """,
            message_ids,
        ).fetchall()
        by_message: dict[str, list[MessagePart]] = {}
        content_cache: dict[str, str | None] = {}
        for row in rows:
            content_id = str(row[12] or "")
            if content_id and content_id not in content_cache:
                content_cache[content_id] = self.content.retrieve(content_id)
            part = MessagePart(
                id=str(row[0]),
                message_id=str(row[1]),
                session_id=str(row[2]),
                part_type=str(row[3]),
                content=content_cache.get(content_id) or str(row[4]),
                content_id=content_id,
                part_index=int(row[5]),
                token_estimate=int(row[6] or 0),
                tool_name=str(row[7]) if row[7] else None,
                tool_call_id=str(row[8]) if row[8] else None,
                tool_state=str(row[9]) if row[9] else None,
                is_protected=bool(row[10]),
                compacted_at=int(row[11]) if row[11] else None,
            )
            by_message.setdefault(part.message_id, []).append(part)
        return by_message

    def close(self) -> None:
        self.content.conn.close()
        self.conn.close()
