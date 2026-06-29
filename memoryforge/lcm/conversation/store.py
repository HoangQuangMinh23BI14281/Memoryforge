"""Turn-level conversation storage for the LCM pipeline."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memoryforge._core import BM25Index, ContentHashTable
from memoryforge.lcm.store import ImmutableMessageStore
from memoryforge.search.vector import VectorIndex


@dataclass
class ConversationTurn:
    role: str
    content: str
    turn_index: int


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))


def _new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000):016x}_{uuid.uuid4().hex[:8]}"


class ConversationChunker:
    """Split conversations into role-based turns."""

    def chunk_conversation(self, messages: Sequence[dict[str, Any]]) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role not in {"user", "assistant", "system", "tool"}:
                role = "user"
            turns.append(ConversationTurn(role=role, content=content, turn_index=index))
        return turns

    def chunk_from_text(self, text: str) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []
        current_role: str | None = None
        current_content: list[str] = []

        def flush() -> None:
            if current_role is None:
                return
            turns.append(
                ConversationTurn(
                    role=current_role,
                    content="\n".join(current_content).strip(),
                    turn_index=len(turns),
                )
            )

        for line in text.splitlines():
            if line.startswith("[user]:"):
                flush()
                current_role = "user"
                current_content = [line[len("[user]:") :].strip()]
            elif line.startswith("[assistant]:"):
                flush()
                current_role = "assistant"
                current_content = [line[len("[assistant]:") :].strip()]
            elif line.startswith("[system]:"):
                flush()
                current_role = "system"
                current_content = [line[len("[system]:") :].strip()]
            elif current_role:
                current_content.append(line)

        flush()
        return turns


class ConversationStore:
    """Conversation facade backed by LCM messages plus shared search indexes."""

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser())
        self.conn = _connect(self.db_path)
        self.chunker = ConversationChunker()
        self.message_store = ImmutableMessageStore(self.db_path)
        self.vector = VectorIndex(self.db_path)
        self.bm25 = BM25Index(self.db_path)
        self.content = ContentHashTable(self.db_path)

    def _init_schema(self) -> None:
        return None

    def store_session(
        self,
        agent_id: str,
        turns: Sequence[ConversationTurn | dict[str, Any]],
        session_id: str | None = None,
        event_date: float | None = None,
    ) -> list[str]:
        session_id = session_id or _new_id("ses")
        event_date = event_date if event_date is not None else time.time()
        normalized = [
            turn
            if isinstance(turn, ConversationTurn)
            else ConversationTurn(
                role=str(turn.get("role", "user")),
                content=str(turn.get("content", "")),
                turn_index=index,
            )
            for index, turn in enumerate(turns)
        ]

        turn_ids: list[str] = []
        for index, turn in enumerate(normalized):
            raw_turn = turns[index] if index < len(turns) else turn
            parts = _parts_from_turn(raw_turn, fallback_content=turn.content)
            message_id = self.message_store.append_message(
                agent_id=agent_id,
                session_id=session_id,
                role=turn.role,
                parts=parts,
                created_at_ms=int(event_date * 1000),
            )
            message = self.message_store.get_message(message_id)
            content_id = message.parts[0].content_id if message and message.parts else ""
            searchable_content = _searchable_content(parts)
            self.bm25.index_turn(agent_id, session_id, turn.role, searchable_content, content_id)
            if content_id:
                self.vector.add(content_id, searchable_content)
            turn_ids.append(message_id)
        return turn_ids

    def store_json_file(
        self,
        agent_id: str,
        file_path: str,
        session_id: str | None = None,
        event_date: float | None = None,
    ) -> list[str]:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            messages = payload.get("turns") or payload.get("messages") or []
            session_id = session_id or payload.get("session_id")
        else:
            messages = payload
        return self.store_session(
            agent_id, self.chunker.chunk_conversation(messages), session_id, event_date
        )

    def search(self, agent_id: str, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        results = []
        for content_id, score in self.bm25.search(agent_id, query, top_k):
            row = self.conn.execute(
                """
                SELECT m.id, m.session_id, m.role, m.created_at, p.id
                FROM messages m
                JOIN message_parts p ON p.message_id = m.id
                WHERE m.agent = ? AND p.content_id = ? AND m.is_summary = 0
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_id, content_id),
            ).fetchone()
            results.append(
                {
                    "content_id": content_id,
                    "score": score,
                    "content": self.content.retrieve(content_id),
                    "turn_id": row[0] if row else None,
                    "session_id": row[1] if row else None,
                    "role": row[2] if row else None,
                    "turn_index": None,
                    "event_date": (float(row[3]) / 1000.0) if row else None,
                    "message_id": row[0] if row else None,
                    "part_id": row[4] if row else None,
                }
            )
        return results

    def get_session(self, agent_id: str, session_id: str) -> list[dict[str, Any]]:
        messages = self.message_store.get_messages(session_id, include_summaries=False)
        return [
            {
                "turn_id": message.id,
                "session_id": session_id,
                "role": message.role,
                "content_id": message.parts[0].content_id if message.parts else "",
                "content": message.content,
                "turn_index": index,
                "event_date": float(message.created_at) / 1000.0,
                "message_id": message.id,
                "part_id": message.parts[0].id if message.parts else None,
            }
            for index, message in enumerate(messages)
            if message.agent_id == agent_id
        ]

    def get_turns_by_ids(self, agent_id: str, turn_ids: list[str]) -> list[dict[str, Any]]:
        if not turn_ids:
            return []
        messages = self.message_store.get_messages_by_ids(turn_ids)
        by_id = {
            message.id: {
                "turn_id": message.id,
                "session_id": message.session_id,
                "role": message.role,
                "content_id": message.parts[0].content_id if message.parts else "",
                "content": message.content,
                "turn_index": index,
                "event_date": float(message.created_at) / 1000.0,
                "message_id": message.id,
                "part_id": message.parts[0].id if message.parts else None,
            }
            for index, message in enumerate(messages)
            if message.agent_id == agent_id
        }
        return [by_id[turn_id] for turn_id in turn_ids if turn_id in by_id]

    def link_turn_message(
        self,
        agent_id: str,
        turn_id: str,
        *,
        message_id: str,
        part_id: str | None = None,
    ) -> None:
        return None

    def close(self) -> None:
        self.message_store.close()
        for component_name in ("content", "bm25", "vector"):
            component = getattr(self, component_name, None)
            conn = getattr(component, "conn", None)
            if conn is not None:
                conn.close()
            setattr(self, component_name, None)
        self.conn.close()


def _parts_from_turn(
    turn: ConversationTurn | dict[str, Any],
    *,
    fallback_content: str,
) -> list[dict[str, Any]]:
    if isinstance(turn, dict):
        raw_parts = turn.get("parts")
        if isinstance(raw_parts, list):
            parts = [dict(part) for part in raw_parts if isinstance(part, dict)]
            if parts:
                return parts
        part_type = str(turn.get("part_type") or "text")
        part: dict[str, Any] = {
            "part_type": part_type,
            "content": str(turn.get("content", fallback_content)),
        }
        for key in ("tool_name", "tool_call_id", "tool_state", "token_estimate", "protected"):
            if key in turn:
                part[key] = turn[key]
        return [part]
    return [{"part_type": "text", "content": fallback_content}]


def _searchable_content(parts: list[dict[str, Any]]) -> str:
    content = "\n".join(str(part.get("content", "")) for part in parts if part.get("content"))
    return content.strip()
