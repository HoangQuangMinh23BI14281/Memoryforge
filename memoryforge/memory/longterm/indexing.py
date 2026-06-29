"""Index raw immutable sources into long-term memory views."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memoryforge.memory.longterm.models import MetadataField
from memoryforge.memory.longterm.utils import preview, stable_id

if TYPE_CHECKING:
    from memoryforge._core import ContentHashTable
    from memoryforge.search.vector import VectorIndex


class LongTermIndexingMixin:
    if TYPE_CHECKING:
        conn: sqlite3.Connection
        content: ContentHashTable
        vector: VectorIndex

    def index_messages(
        self,
        agent_id: str,
        message_ids: list[str],
        *,
        metadata_by_message_id: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        if not message_ids:
            return []
        metadata_by_message_id = metadata_by_message_id or {}
        placeholders = ",".join("?" for _message_id in message_ids)
        rows = self.conn.execute(
            f"""
            SELECT m.id, m.session_id, m.role, p.content_id, p.part_index,
                   c.content, p.id, m.created_at
            FROM messages m
            JOIN message_parts p ON p.message_id = m.id
            JOIN content_store c ON c.content_id = p.content_id
            WHERE m.agent = ? AND m.id IN ({placeholders}) AND p.part_type = 'text'
            """,
            (agent_id, *message_ids),
        ).fetchall()
        by_id = {str(row[0]): row for row in rows}
        indexed: list[str] = []
        for message_id in message_ids:
            row = by_id.get(message_id)
            if row is None:
                continue
            event_date = _epoch_seconds(row[7])
            metadata = {
                MetadataField.SESSION_ID: row[1],
                MetadataField.ROLE: row[2],
                MetadataField.PART_INDEX: row[4],
                MetadataField.MESSAGE_ID: row[0],
                MetadataField.PART_ID: row[6],
                MetadataField.EVENT_DATE: event_date,
                MetadataField.TIMESTAMP: _iso_utc(event_date),
                MetadataField.CREATED_AT: event_date,
            }
            metadata.update(metadata_by_message_id.get(str(row[0]), {}))
            indexed.append(
                self.index_raw_item(
                    agent_id=agent_id,
                    source_type="message",
                    source_id=str(row[0]),
                    content_id=str(row[3]),
                    text=str(row[5] or ""),
                    metadata=metadata,
                )
            )
        return indexed

    def merge_item_metadata(
        self,
        agent_id: str,
        item_id: str,
        metadata: dict[str, Any],
    ) -> None:
        self.merge_items_metadata(agent_id, [(item_id, metadata)])

    def merge_items_metadata(
        self,
        agent_id: str,
        updates: list[tuple[str, dict[str, Any]]],
    ) -> None:
        updates = [(item_id, metadata) for item_id, metadata in updates if metadata]
        if not updates:
            return
        transaction_started = _begin_immediate(self.conn)
        try:
            item_ids = [item_id for item_id, _metadata in updates]
            placeholders = ",".join("?" for _item_id in item_ids)
            rows = self.conn.execute(
                f"""
                SELECT item_id, metadata
                FROM long_term_items
                WHERE agent_id = ? AND item_id IN ({placeholders})
                """,
                (agent_id, *item_ids),
            ).fetchall()
            existing_by_id = {str(row[0]): self._metadata_json(row[1]) for row in rows}
            merged_items: list[dict[str, Any]] = []
            for item_id, metadata in updates:
                existing = existing_by_id.get(item_id)
                if existing is None:
                    continue
                existing.update(metadata)
                merged_items.append(
                    {
                        "agent_id": agent_id,
                        "item_id": item_id,
                        "metadata": existing,
                    }
                )
            self._apply_explicit_supersession_policy(merged_items)
            merged_rows = [
                (
                    json.dumps(item["metadata"], ensure_ascii=False, sort_keys=True),
                    agent_id,
                    item["item_id"],
                )
                for item in merged_items
            ]
            if merged_rows:
                self.conn.executemany(
                    """
                    UPDATE long_term_items
                    SET metadata = ?
                    WHERE agent_id = ? AND item_id = ?
                    """,
                    merged_rows,
                )
            if transaction_started:
                self.conn.commit()
        except Exception:
            if transaction_started:
                self.conn.rollback()
            raise

    def index_rlm_buffer(self, agent_id: str, buffer_id: str) -> list[str]:
        buffer_row = self.conn.execute(
            """
            SELECT buffer_id, name, source_path, content_id, metadata
            FROM rlm_buffers
            WHERE agent_id = ? AND buffer_id = ?
            """,
            (agent_id, buffer_id),
        ).fetchone()
        if buffer_row is None:
            return []
        raw_content_id = str(buffer_row[3])
        buffer_metadata = self._metadata_json(buffer_row[4])
        existing_items = self._existing_corpus_items(
            agent_id,
            raw_content_id=raw_content_id,
            source_path=buffer_row[2],
            source_types=("rlm_chunk",),
            chunk_size=_optional_int(buffer_metadata.get("chunk_size")),
            overlap=_optional_int(buffer_metadata.get("overlap")),
        )
        if existing_items:
            return [item["item_id"] for item in existing_items]
        rows = self.conn.execute(
            """
            SELECT c.chunk_id, c.buffer_id, b.name, b.source_path, c.content_id,
                   c.chunk_index, c.content_type, c.strategy,
                   c.byte_start, c.byte_end, c.char_start, c.char_end,
                   c.start_line, c.end_line, c.token_count, c.created_at, s.content,
                   b.content_id, b.metadata
            FROM rlm_chunks c
            JOIN rlm_buffers b ON b.buffer_id = c.buffer_id
            JOIN content_store s ON s.content_id = c.content_id
            WHERE c.agent_id = ? AND c.buffer_id = ?
            ORDER BY c.chunk_index
            """,
            (agent_id, buffer_id),
        ).fetchall()
        return self.index_raw_items([self._rlm_chunk_item(agent_id, row) for row in rows])

    def index_rlm_chunks(self, agent_id: str, chunk_ids: list[str]) -> list[str]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _chunk_id in chunk_ids)
        rows = self.conn.execute(
            f"""
            SELECT c.chunk_id, c.buffer_id, b.name, b.source_path, c.content_id,
                   c.chunk_index, c.content_type, c.strategy,
                   c.byte_start, c.byte_end, c.char_start, c.char_end,
                   c.start_line, c.end_line, c.token_count, c.created_at, s.content,
                   b.content_id, b.metadata
            FROM rlm_chunks c
            JOIN rlm_buffers b ON b.buffer_id = c.buffer_id
            JOIN content_store s ON s.content_id = c.content_id
            WHERE c.agent_id = ? AND c.chunk_id IN ({placeholders})
            """,
            (agent_id, *chunk_ids),
        ).fetchall()
        by_id = {str(row[0]): row for row in rows}
        return self.index_raw_items(
            [self._rlm_chunk_item(agent_id, by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in by_id]
        )

    def index_rlm_worker_result(self, agent_id: str, worker_result: dict[str, Any]) -> list[str]:
        """Index RLM sub-agent analyses and aggregate summaries into LTM.

        RLM chunks remain the lossless deep source. These derived items are the
        shallow recall layer LCM can inject first; their metadata keeps exact
        rlm_chunk refs for later rehydration.
        """

        items: list[dict[str, Any]] = []
        for record in _worker_records(worker_result):
            analysis_text = str(record.get("analysis") or "").strip()
            if not analysis_text:
                continue
            source_id = str(
                record.get("summary_node_id")
                or record.get("analysis_id")
                or f"{record.get('run_id')}:batch:{(record.get('metadata') or {}).get('batch_index')}"
            )
            chunk_ids = _string_list(record.get("chunk_ids"))
            metadata = _rlm_worker_metadata(
                kind="rlm_analysis",
                run_id=record.get("run_id") or _worker_run_id(worker_result),
                chunk_ids=chunk_ids,
                source_refs=_chunk_refs(chunk_ids, record.get("source_refs")),
                summary_node_id=record.get("summary_node_id"),
                message_id=record.get("message_id"),
                extra=dict(record.get("metadata") or {}),
            )
            items.append(
                {
                    "agent_id": agent_id,
                    "source_type": "rlm_analysis",
                    "source_id": source_id,
                    "text": analysis_text,
                    "metadata": metadata,
                }
            )

        aggregate = worker_result.get("aggregate")
        if isinstance(aggregate, dict):
            summary_text = str(aggregate.get("summary") or "").strip()
            if summary_text:
                chunk_ids = _string_list(aggregate.get("source_chunk_ids"))
                source_refs = _chunk_refs(chunk_ids, aggregate.get("source_refs"))
                source_id = str(
                    aggregate.get("aggregate_id")
                    or aggregate.get("summary_node_id")
                    or aggregate.get("run_id")
                    or "aggregate"
                )
                metadata = _rlm_worker_metadata(
                    kind="rlm_summary",
                    run_id=aggregate.get("run_id") or _worker_run_id(worker_result),
                    chunk_ids=chunk_ids,
                    source_refs=source_refs,
                    summary_node_id=aggregate.get("summary_node_id"),
                    message_id=aggregate.get("message_id"),
                    extra={
                        **dict(aggregate.get("metadata") or {}),
                        "aggregate_id": aggregate.get("aggregate_id"),
                        "child_node_ids": _string_list(aggregate.get("child_node_ids")),
                    },
                )
                items.append(
                    {
                        "agent_id": agent_id,
                        "source_type": "rlm_summary",
                        "source_id": source_id,
                        "text": summary_text,
                        "metadata": metadata,
                    }
                )

        return self.index_raw_items(items)

    def index_raw_item(
        self,
        agent_id: str,
        source_type: str,
        source_id: str,
        text: str,
        *,
        content_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.index_raw_items(
            [
                {
                    "agent_id": agent_id,
                    "source_type": source_type,
                    "source_id": source_id,
                    "text": text,
                    "content_id": content_id,
                    "metadata": metadata or {},
                }
            ]
        )[0]

    def index_raw_items(self, items: list[dict[str, Any]]) -> list[str]:
        if not items:
            return []
        prepared: list[dict[str, Any]] = []
        for item in items:
            text = str(item["text"])
            source_type = str(item["source_type"])
            source_id = str(item["source_id"])
            content_id = item.get("content_id")
            if content_id is None:
                content_id, _is_new = self.content.store(text)
            content_id = str(content_id)
            incoming_metadata = dict(item.get("metadata") or {})
            prepared.append(
                {
                    "item_id": stable_id("ltm", str(item["agent_id"]), source_type, source_id),
                    "agent_id": str(item["agent_id"]),
                    "source_type": source_type,
                    "source_id": source_id,
                    "content_id": content_id,
                    "preview": preview(text),
                    "metadata": _memory_metadata(source_type, source_id, incoming_metadata),
                    "text": text,
                }
            )
        vector_items = [(item["content_id"], item["text"]) for item in prepared]
        try:
            self.vector.add_many(vector_items)
        except Exception:
            pass

        now = time.time()
        transaction_started = _begin_immediate(self.conn)
        try:
            self._apply_explicit_supersession_policy(prepared)
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO long_term_items
                (item_id, agent_id, source_type, source_id, content_id, preview, metadata, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["item_id"],
                        item["agent_id"],
                        item["source_type"],
                        item["source_id"],
                        item["content_id"],
                        item["preview"],
                        json.dumps(item["metadata"], ensure_ascii=False, sort_keys=True),
                        now,
                    )
                    for item in prepared
                ],
            )
            self.conn.executemany(
                "DELETE FROM search_fts WHERE scope = ? AND source_id = ?",
                [("long_term", item["item_id"]) for item in prepared],
            )
            self.conn.executemany(
                """
                INSERT INTO search_fts
                (content, scope, agent_id, source_type, source_id, content_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["text"],
                        "long_term",
                        item["agent_id"],
                        item["source_type"],
                        item["item_id"],
                        item["content_id"],
                    )
                    for item in prepared
                ],
            )
            if transaction_started:
                self.conn.commit()
        except Exception:
            if transaction_started:
                self.conn.rollback()
            raise
        return [str(item["item_id"]) for item in prepared]

    def _index_rlm_chunk_row(self, agent_id: str, row: Any) -> str:
        item = self._rlm_chunk_item(agent_id, row)
        return self.index_raw_item(
            agent_id=str(item["agent_id"]),
            source_type=str(item["source_type"]),
            source_id=str(item["source_id"]),
            content_id=str(item["content_id"]),
            text=str(item["text"]),
            metadata=dict(item["metadata"]),
        )

    def _apply_explicit_supersession_policy(self, prepared: list[dict[str, Any]]) -> None:
        now = time.time()
        for item in prepared:
            metadata = item["metadata"]
            merge_key = metadata.get(MetadataField.MERGE_KEY)
            explicit_supersedes = [
                str(item_id)
                for item_id in metadata.get(MetadataField.SUPERSEDES) or []
                if str(item_id)
            ]
            if explicit_supersedes:
                self._supersede_item_ids(
                    agent_id=str(item["agent_id"]),
                    item_id=str(item["item_id"]),
                    metadata=metadata,
                    superseded_ids=explicit_supersedes,
                    now=now,
                )
            if merge_key:
                self._supersede_merge_candidates(
                    agent_id=str(item["agent_id"]),
                    item_id=str(item["item_id"]),
                    metadata=metadata,
                    merge_key=str(merge_key),
                    now=now,
                )

    def _supersede_merge_candidates(
        self,
        *,
        agent_id: str,
        item_id: str,
        metadata: dict[str, Any],
        merge_key: str,
        now: float,
    ) -> None:
        rows = self.conn.execute(
            """
            SELECT item_id, metadata
            FROM long_term_items
            WHERE agent_id = ?
            """,
            (agent_id,),
        ).fetchall()
        superseded_ids: list[str] = []
        updates: list[tuple[str, str, str]] = []
        for row in rows:
            old_item_id = str(row[0])
            if old_item_id == item_id:
                continue
            old_metadata = self._metadata_json(row[1])
            if old_metadata.get(MetadataField.MERGE_KEY) != merge_key:
                continue
            if old_metadata.get(MetadataField.VALID_TO) or old_metadata.get(
                MetadataField.SUPERSEDED_BY
            ):
                continue
            superseded_by = list(old_metadata.get(MetadataField.SUPERSEDED_BY) or [])
            if item_id not in superseded_by:
                superseded_by.append(item_id)
            old_metadata.update(
                {
                    MetadataField.SUPERSEDED_BY: superseded_by,
                    MetadataField.VALID_TO: now,
                    MetadataField.FRESHNESS: now,
                }
            )
            superseded_ids.append(old_item_id)
            updates.append(
                (
                    json.dumps(old_metadata, ensure_ascii=False, sort_keys=True),
                    agent_id,
                    old_item_id,
                )
            )
        if not updates:
            return
        existing_supersedes = list(metadata.get(MetadataField.SUPERSEDES) or [])
        metadata[MetadataField.SUPERSEDES] = _dedupe_ordered(
            [*existing_supersedes, *superseded_ids]
        )
        self.conn.executemany(
            """
            UPDATE long_term_items
            SET metadata = ?
            WHERE agent_id = ? AND item_id = ?
            """,
            updates,
        )

    def _supersede_item_ids(
        self,
        *,
        agent_id: str,
        item_id: str,
        metadata: dict[str, Any],
        superseded_ids: list[str],
        now: float,
    ) -> None:
        if not superseded_ids:
            return
        placeholders = ",".join("?" for _item_id in superseded_ids)
        rows = self.conn.execute(
            f"""
            SELECT item_id, metadata
            FROM long_term_items
            WHERE agent_id = ? AND item_id IN ({placeholders})
            """,
            (agent_id, *superseded_ids),
        ).fetchall()
        updates: list[tuple[str, str, str]] = []
        touched_ids: list[str] = []
        for row in rows:
            old_item_id = str(row[0])
            if old_item_id == item_id:
                continue
            old_metadata = self._metadata_json(row[1])
            superseded_by = list(old_metadata.get(MetadataField.SUPERSEDED_BY) or [])
            if item_id not in superseded_by:
                superseded_by.append(item_id)
            old_metadata.update(
                {
                    MetadataField.SUPERSEDED_BY: superseded_by,
                    MetadataField.VALID_TO: old_metadata.get(MetadataField.VALID_TO) or now,
                    MetadataField.FRESHNESS: now,
                }
            )
            touched_ids.append(old_item_id)
            updates.append(
                (
                    json.dumps(old_metadata, ensure_ascii=False, sort_keys=True),
                    agent_id,
                    old_item_id,
                )
            )
        if not updates:
            return
        metadata[MetadataField.SUPERSEDES] = _dedupe_ordered(
            [*list(metadata.get(MetadataField.SUPERSEDES) or []), *touched_ids]
        )
        self.conn.executemany(
            """
            UPDATE long_term_items
            SET metadata = ?
            WHERE agent_id = ? AND item_id = ?
            """,
            updates,
        )

    @staticmethod
    def _rlm_chunk_item(agent_id: str, row: Any) -> dict[str, Any]:
        raw_content_id = str(row[17] if len(row) > 17 and row[17] is not None else row[4])
        document_key = str(row[3] or row[2] or raw_content_id)
        document_id = stable_id("doc", agent_id, document_key, raw_content_id)
        buffer_metadata = _metadata_json(row[18] if len(row) > 18 else None)
        return {
            "agent_id": agent_id,
            "source_type": "rlm_chunk",
            "source_id": str(row[0]),
            "content_id": str(row[4]),
            "text": str(row[16] or ""),
            "metadata": {
                "document_id": document_id,
                "buffer_id": row[1],
                "buffer_name": row[2],
                "source_path": row[3],
                "raw_content_id": raw_content_id,
                "chunk_index": row[5],
                "content_type": row[6],
                "strategy": row[7],
                "byte_range": {"start": row[8], "end": row[9]},
                "char_range": {"start": row[10], "end": row[11]},
                "start_line": row[12],
                "end_line": row[13],
                "token_count": row[14],
                "chunk_size": buffer_metadata.get("chunk_size"),
                "overlap": buffer_metadata.get("overlap"),
            },
        }

    def _existing_corpus_items(
        self,
        agent_id: str,
        *,
        raw_content_id: str | None,
        source_path: str | None,
        source_types: tuple[str, ...],
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[dict[str, Any]]:
        if not raw_content_id and not source_path:
            return []
        placeholders = ",".join("?" for _source_type in source_types)
        rows = self.conn.execute(
            f"""
            SELECT item_id, source_type, metadata, content_id
            FROM long_term_items
            WHERE agent_id = ? AND source_type IN ({placeholders})
            """,
            (agent_id, *source_types),
        ).fetchall()
        matches: list[dict[str, Any]] = []
        for row in rows:
            metadata = self._metadata_json(row[2])
            if chunk_size is not None and _optional_int(metadata.get("chunk_size")) != chunk_size:
                continue
            if overlap is not None and _optional_int(metadata.get("overlap")) != overlap:
                continue
            metadata_raw_content_id = metadata.get("raw_content_id")
            metadata_source_path = metadata.get("source_path")
            if raw_content_id and (
                metadata_raw_content_id == raw_content_id or str(row[3]) == raw_content_id
            ):
                matches.append(
                    {"item_id": str(row[0]), "source_type": str(row[1]), "metadata": metadata}
                )
            elif source_path and metadata_source_path == source_path:
                matches.append(
                    {"item_id": str(row[0]), "source_type": str(row[1]), "metadata": metadata}
                )
        matches.sort(key=lambda item: int(item["metadata"].get("chunk_index", 0) or 0))
        return matches

    @staticmethod
    def _metadata_json(value: Any) -> dict[str, Any]:
        try:
            metadata = json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return metadata if isinstance(metadata, dict) else {}


def _metadata_json(value: Any) -> dict[str, Any]:
    try:
        metadata = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _memory_metadata(source_type: str, source_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    raw_ref = f"{source_type}:{source_id}"
    raw_refs = [str(ref) for ref in normalized.get(MetadataField.RAW_REFS) or [] if str(ref)]
    if raw_ref not in raw_refs:
        raw_refs.append(raw_ref)
    normalized[MetadataField.RAW_REFS] = raw_refs
    if MetadataField.SOURCE_ORIGIN not in normalized:
        origin = (
            normalized.get(MetadataField.SOURCE)
            or normalized.get(MetadataField.ROLE)
            or ("file" if normalized.get(MetadataField.SOURCE_PATH) else source_type)
        )
        normalized[MetadataField.SOURCE_ORIGIN] = str(origin)
    return normalized


def _worker_records(worker_result: dict[str, Any]) -> list[dict[str, Any]]:
    records = worker_result.get("records")
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _chunk_refs(chunk_ids: list[str], existing_refs: Any = None) -> list[str]:
    refs = [str(ref) for ref in existing_refs if str(ref)] if isinstance(existing_refs, list) else []
    for chunk_id in chunk_ids:
        ref = f"rlm_chunk:{chunk_id}"
        if ref not in refs:
            refs.append(ref)
    return refs


def _worker_run_id(worker_result: dict[str, Any]) -> Any:
    plan = worker_result.get("plan")
    return plan.get("run_id") if isinstance(plan, dict) else None


def _rlm_worker_metadata(
    *,
    kind: str,
    run_id: Any,
    chunk_ids: list[str],
    source_refs: list[str],
    summary_node_id: Any,
    message_id: Any,
    extra: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(extra)
    metadata.update(
        {
            MetadataField.KIND: kind,
            MetadataField.SOURCE_ORIGIN: "rlm_worker",
            MetadataField.CONFIDENCE: metadata.get(MetadataField.CONFIDENCE)
            or "medium",
            "run_id": str(run_id) if run_id else None,
            "chunk_ids": chunk_ids,
            "summary_node_id": str(summary_node_id) if summary_node_id else None,
            "message_id": str(message_id) if message_id else None,
            "derived_from": "rlm_subagent",
            "recall_depth": "shallow",
            MetadataField.RAW_REFS: source_refs,
        }
    )
    return {key: value for key, value in metadata.items() if value not in (None, [], {})}


def _begin_immediate(conn: sqlite3.Connection) -> bool:
    if conn.in_transaction:
        return False
    conn.execute("BEGIN IMMEDIATE")
    return True


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _epoch_seconds(value: Any) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return time.time()
    if timestamp > 10_000_000_000:
        return timestamp / 1000.0
    return timestamp


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
