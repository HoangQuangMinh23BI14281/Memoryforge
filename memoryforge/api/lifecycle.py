"""Durable memory lifecycle operations."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from memoryforge.memory.longterm.models import MemoryConfidence, MetadataField
from memoryforge.memory.longterm.utils import tokenize_query

if TYPE_CHECKING:
    from memoryforge.memory.longterm.store import LongTermMemoryIndex


class MemoryLifecycleMixin:
    if TYPE_CHECKING:

        @property
        def long_term(self) -> LongTermMemoryIndex: ...

        def long_term_source(self, agent_id: str, item_id: str) -> dict[str, Any] | None: ...

    def record_correction(
        self,
        agent_id: str,
        corrected_fact: str,
        *,
        wrong_item_id: str | None = None,
        wrong_raw_ref: str | None = None,
        session_id: str | None = None,
        source: str = "user",
    ) -> dict[str, Any]:
        """Record a user correction as durable high-confidence memory."""

        corrected_text = corrected_fact.strip()
        if not corrected_text:
            raise ValueError("corrected_fact cannot be empty")

        raw_refs: list[str] = []
        supersedes: list[str] = []
        contradicts: list[str] = []
        wrong_source = None
        if wrong_item_id:
            wrong_source = self.long_term_source(agent_id, wrong_item_id)
            if wrong_source is not None:
                supersedes.append(wrong_item_id)
                wrong_raw_ref = wrong_raw_ref or str(wrong_source["raw_ref"])
        if wrong_raw_ref:
            raw_refs.append(wrong_raw_ref)
            contradicts.append(wrong_raw_ref)

        corrected_at = time.time()
        correction_source_id = _stable_correction_id(
            agent_id=agent_id,
            corrected_fact=corrected_text,
            wrong_item_id=wrong_item_id,
            wrong_raw_ref=wrong_raw_ref,
        )
        item_id = self.long_term.index_raw_item(
            agent_id=agent_id,
            source_type="correction",
            source_id=correction_source_id,
            text=corrected_text,
            metadata={
                MetadataField.KIND: "correction",
                MetadataField.CONFIDENCE: MemoryConfidence.HIGH,
                MetadataField.FRESHNESS: corrected_at,
                MetadataField.CORRECTED_AT: corrected_at,
                MetadataField.SOURCE: source,
                MetadataField.SESSION_ID: session_id,
                MetadataField.RAW_REFS: raw_refs,
                MetadataField.SUPERSEDES: supersedes,
                MetadataField.CONTRADICTS: contradicts,
                MetadataField.CORRECTED_FACT: corrected_text,
            },
        )
        if wrong_item_id and wrong_source is not None:
            self._mark_memory_corrected(
                agent_id=agent_id,
                item_id=wrong_item_id,
                correction_item_id=item_id,
                corrected_at=corrected_at,
            )
        preserved_item_ids: list[str] = []
        correction_source = self.long_term_source(agent_id, item_id)
        return {
            "item_id": item_id,
            "raw_ref": f"correction:{correction_source_id}",
            "corrected_fact": corrected_text,
            "wrong_item_id": wrong_item_id,
            "wrong_raw_ref": wrong_raw_ref,
            "supersedes": supersedes,
            "contradicts": contradicts,
            "preserved_item_ids": preserved_item_ids,
            "metadata": correction_source["metadata"] if correction_source else {},
        }

    def record_contradiction(
        self,
        agent_id: str,
        statement: str,
        *,
        conflicting_item_ids: list[str] | None = None,
        conflicting_raw_refs: list[str] | None = None,
        session_id: str | None = None,
        source: str = "user",
    ) -> dict[str, Any]:
        """Record a contested memory without deciding which side is correct."""

        statement_text = statement.strip()
        if not statement_text:
            raise ValueError("statement cannot be empty")
        requested_item_ids = [str(item_id) for item_id in conflicting_item_ids or []]
        raw_refs = [str(ref) for ref in conflicting_raw_refs or [] if str(ref)]
        conflicting_sources: list[dict[str, Any]] = []
        for item_id in requested_item_ids:
            source_item = self.long_term_source(agent_id, item_id)
            if source_item is None:
                raise ValueError(f"Unknown conflicting item_id: {item_id}")
            conflicting_sources.append(source_item)
            raw_refs.append(str(source_item["raw_ref"]))
        raw_refs = _dedupe_ordered(raw_refs)
        if not requested_item_ids and not raw_refs:
            raise ValueError("record_contradiction requires conflicting_item_ids or conflicting_raw_refs")

        recorded_at = time.time()
        source_id = _stable_contradiction_id(
            agent_id=agent_id,
            statement=statement_text,
            conflicting_item_ids=requested_item_ids,
            conflicting_raw_refs=raw_refs,
        )
        item_id = self.long_term.index_raw_item(
            agent_id=agent_id,
            source_type="contradiction",
            source_id=source_id,
            text=statement_text,
            metadata={
                MetadataField.CONFIDENCE: MemoryConfidence.MEDIUM,
                MetadataField.FRESHNESS: recorded_at,
                MetadataField.RECORDED_AT: recorded_at,
                MetadataField.SOURCE: source,
                MetadataField.SESSION_ID: session_id,
                MetadataField.RAW_REFS: raw_refs,
                MetadataField.CONTRADICTS: raw_refs,
                MetadataField.CONFLICTING_ITEM_IDS: requested_item_ids,
            },
        )
        contradiction_source = self.long_term_source(agent_id, item_id)
        contradiction_raw_ref = str((contradiction_source or {}).get("raw_ref") or "")
        for source_item in conflicting_sources:
            self._mark_memory_contradicted(
                agent_id=agent_id,
                item_id=str(source_item["item_id"]),
                contradiction_item_id=item_id,
                contradiction_raw_ref=contradiction_raw_ref,
                recorded_at=recorded_at,
            )
        return {
            "item_id": item_id,
            "raw_ref": f"contradiction:{source_id}",
            "statement": statement_text,
            "conflicting_item_ids": requested_item_ids,
            "contradicts": raw_refs,
            "metadata": contradiction_source["metadata"] if contradiction_source else {},
        }

    def find_contradictions(
        self,
        agent_id: str,
        *,
        query: str | None = None,
        limit: int = 10,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Return memories marked as conflicting through contradiction metadata."""

        focus_tokens = set(tokenize_query(query or ""))
        candidates = self._contradiction_candidates(agent_id, include_content=include_content)
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            score = 1.0
            if focus_tokens:
                candidate_text = " ".join(
                    [
                        str(candidate.get("preview") or ""),
                        str(candidate.get("content") or ""),
                        " ".join(str(entity) for entity in candidate["metadata"].get("entities") or []),
                    ]
                )
                overlap = focus_tokens & set(tokenize_query(candidate_text))
                if not overlap:
                    continue
                score += len(overlap)
            item = dict(candidate)
            item["contradiction_score"] = score
            scored.append(item)
        selected = sorted(
            scored,
            key=lambda item: (
                float(item.get("contradiction_score") or 0.0),
                float(item.get("indexed_at") or 0.0),
            ),
            reverse=True,
        )[: max(1, limit)]
        return {
            "agent_id": agent_id,
            "query": query,
            "results": [self._contradiction_public_item(item) for item in selected],
            "diagnostics": {
                "candidate_count": len(candidates),
                "returned_count": len(selected),
                "query_used": bool(query),
                "answer_model_used": False,
            },
        }

    def _mark_memory_corrected(
        self,
        *,
        agent_id: str,
        item_id: str,
        correction_item_id: str,
        corrected_at: float,
    ) -> None:
        row = self.long_term.conn.execute(
            """
            SELECT metadata
            FROM long_term_items
            WHERE agent_id = ? AND item_id = ?
            """,
            (agent_id, item_id),
        ).fetchone()
        if row is None:
            return
        metadata = _json_dict(row[0])
        superseded_by = list(metadata.get(MetadataField.SUPERSEDED_BY) or [])
        if correction_item_id not in superseded_by:
            superseded_by.append(correction_item_id)
        metadata.update(
            {
                MetadataField.CONFIDENCE: MemoryConfidence.LOW,
                MetadataField.FRESHNESS: corrected_at,
                MetadataField.VALID_TO: corrected_at,
                MetadataField.SUPERSEDED_BY: superseded_by,
            }
        )
        self.long_term.conn.execute(
            """
            UPDATE long_term_items
            SET metadata = ?
            WHERE agent_id = ? AND item_id = ?
            """,
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True), agent_id, item_id),
        )
        self.long_term.conn.commit()

    def _mark_memory_contradicted(
        self,
        *,
        agent_id: str,
        item_id: str,
        contradiction_item_id: str,
        contradiction_raw_ref: str,
        recorded_at: float,
    ) -> None:
        row = self.long_term.conn.execute(
            """
            SELECT metadata
            FROM long_term_items
            WHERE agent_id = ? AND item_id = ?
            """,
            (agent_id, item_id),
        ).fetchone()
        if row is None:
            return
        metadata = _json_dict(row[0])
        contradicted_by = list(metadata.get(MetadataField.CONTRADICTED_BY) or [])
        if contradiction_item_id not in contradicted_by:
            contradicted_by.append(contradiction_item_id)
        contradicts = list(metadata.get(MetadataField.CONTRADICTS) or [])
        if contradiction_raw_ref and contradiction_raw_ref not in contradicts:
            contradicts.append(contradiction_raw_ref)
        metadata.update(
            {
                MetadataField.FRESHNESS: recorded_at,
                MetadataField.CONTRADICTED_BY: contradicted_by,
                MetadataField.CONTRADICTS: contradicts,
            }
        )
        self.long_term.conn.execute(
            """
            UPDATE long_term_items
            SET metadata = ?
            WHERE agent_id = ? AND item_id = ?
            """,
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True), agent_id, item_id),
        )
        self.long_term.conn.commit()

    def _contradiction_candidates(
        self,
        agent_id: str,
        *,
        include_content: bool,
    ) -> list[dict[str, Any]]:
        if include_content:
            rows = self.long_term.conn.execute(
                """
                SELECT l.item_id, l.source_type, l.source_id, l.content_id,
                       l.preview, l.metadata, l.indexed_at, c.content
                FROM long_term_items l
                JOIN content_store c ON c.content_id = l.content_id
                WHERE l.agent_id = ?
                """,
                (agent_id,),
            ).fetchall()
        else:
            rows = self.long_term.conn.execute(
                """
                SELECT item_id, source_type, source_id, content_id,
                       preview, metadata, indexed_at, NULL
                FROM long_term_items
                WHERE agent_id = ?
                """,
                (agent_id,),
            ).fetchall()
        candidates: list[dict[str, Any]] = []
        for row in rows:
            metadata = _json_dict(row[5])
            if not (
                metadata.get(MetadataField.CONTRADICTS)
                or metadata.get(MetadataField.CONTRADICTED_BY)
                or metadata.get(MetadataField.CONFLICTING_ITEM_IDS)
            ):
                continue
            candidates.append(
                {
                    "item_id": str(row[0]),
                    "source_type": str(row[1]),
                    "source_id": str(row[2]),
                    "content_id": str(row[3]),
                    "raw_ref": f"{row[1]}:{row[2]}",
                    "preview": str(row[4] or ""),
                    "metadata": metadata,
                    "indexed_at": float(row[6] or 0.0),
                    "content": row[7] if include_content else None,
                }
            )
        return candidates

    @staticmethod
    def _contradiction_public_item(item: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(item.get("metadata") or {})
        payload = {
            "item_id": item["item_id"],
            "raw_ref": item["raw_ref"],
            "source_type": item["source_type"],
            "source_id": item["source_id"],
            "content_id": item["content_id"],
            "preview": item["preview"],
            "metadata": metadata,
            "contradiction_score": item["contradiction_score"],
            "contradicts": metadata.get(MetadataField.CONTRADICTS) or [],
            "contradicted_by": metadata.get(MetadataField.CONTRADICTED_BY) or [],
            "conflicting_item_ids": metadata.get(MetadataField.CONFLICTING_ITEM_IDS) or [],
        }
        if item.get("content") is not None:
            payload["content"] = item["content"]
        return payload


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _stable_correction_id(
    *,
    agent_id: str,
    corrected_fact: str,
    wrong_item_id: str | None,
    wrong_raw_ref: str | None,
) -> str:
    digest = hashlib.blake2b(
        "\n".join(
            [
                agent_id,
                wrong_item_id or "",
                wrong_raw_ref or "",
                corrected_fact,
            ]
        ).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return f"corr_{digest}"


def _stable_contradiction_id(
    *,
    agent_id: str,
    statement: str,
    conflicting_item_ids: list[str],
    conflicting_raw_refs: list[str],
) -> str:
    digest = hashlib.blake2b(
        "\n".join(
            [
                agent_id,
                statement,
                json.dumps(sorted(conflicting_item_ids), sort_keys=True),
                json.dumps(sorted(conflicting_raw_refs), sort_keys=True),
            ]
        ).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return f"contra_{digest}"
