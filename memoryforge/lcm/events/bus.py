"""SQLite-backed event bus for LCM orchestration."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventRecord:
    id: str
    event_type: str
    payload: dict[str, Any]
    agent_id: str | None
    session_id: str | None
    created_at: float


_EVENTS_BY_DB: dict[str, list[EventRecord]] = {}


def _new_id() -> str:
    return f"evt_{int(time.time() * 1000):016x}_{uuid.uuid4().hex[:8]}"


class EventBus:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.handlers: dict[str, list[Callable[[EventRecord], None]]] = {}
        self.records = _EVENTS_BY_DB.setdefault(db_path, [])

    def subscribe(self, event_type: str, handler: Callable[[EventRecord], None]) -> None:
        self.handlers.setdefault(event_type, []).append(handler)

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> EventRecord:
        record = EventRecord(
            id=_new_id(),
            event_type=event_type,
            payload=payload or {},
            agent_id=agent_id,
            session_id=session_id,
            created_at=time.time(),
        )
        json.dumps(record.payload, ensure_ascii=False)
        self.records.append(record)
        for handler in self.handlers.get(event_type, []):
            handler(record)
        for handler in self.handlers.get("*", []):
            handler(record)
        return record

    def publish_coalesced(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        coalesce_key: str,
    ) -> EventRecord:
        payload = dict(payload or {})
        for record in reversed(self.records):
            if record.session_id != session_id:
                continue
            if record.event_type != event_type:
                break
            if record.payload.get("coalesce_key") != coalesce_key:
                break
            count = int(record.payload.get("coalesced_count", 1) or 1) + 1
            record.payload.update(payload)
            record.payload["coalesce_key"] = coalesce_key
            record.payload["coalesced_count"] = count
            record.payload["last_seen_at"] = time.time()
            json.dumps(record.payload, ensure_ascii=False)
            return record
        payload["coalesce_key"] = coalesce_key
        payload["coalesced_count"] = 1
        payload["last_seen_at"] = time.time()
        return self.publish(event_type, payload, agent_id=agent_id, session_id=session_id)

    def list_events(self, session_id: str | None = None, limit: int = 100) -> list[EventRecord]:
        rows = [
            record
            for record in self.records
            if session_id is None or record.session_id == session_id
        ]
        return list(reversed(rows))[:limit]

    def close(self) -> None:
        return None
