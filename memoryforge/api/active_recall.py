"""Active recall candidate selection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryforge.api.context_assembly import active_recall_reason
from memoryforge.memory.longterm.models import MemoryConfidence, MetadataField

if TYPE_CHECKING:
    from memoryforge.memory.longterm.store import LongTermMemoryIndex


class ActiveRecallMixin:
    if TYPE_CHECKING:

        @property
        def long_term(self) -> LongTermMemoryIndex: ...

    def active_recall(
        self,
        agent_id: str,
        *,
        session_id: str | None = None,
        focus: str | None = None,
        project_root: str | None = None,
        limit: int = 8,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Surface recent durable evidence without requiring a search query."""

        started = time.perf_counter()
        resolved_limit = max(1, int(limit))

        candidates = self._active_recall_candidates(
            agent_id,
            include_content=include_content,
            limit=max(50, resolved_limit * 8),
        )
        by_id: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            score, reasons = self._active_recall_score(
                candidate,
                session_id=session_id,
            )
            if score <= 0:
                continue
            item = dict(candidate)
            item["active_recall_score"] = round(score, 6)
            item["reasons"] = reasons
            by_id[str(item["item_id"])] = item

        selected = sorted(
            by_id.values(),
            key=lambda item: (
                float(item.get("active_recall_score") or 0.0),
                float(item.get("indexed_at") or 0.0),
            ),
            reverse=True,
        )[:resolved_limit]
        raw_refs = _dedupe_ordered([str(item["raw_ref"]) for item in selected])
        return {
            "agent_id": agent_id,
            "session_id": session_id,
            "focus": focus,
            "project_root": str(Path(project_root).expanduser().resolve())
            if project_root
            else None,
            "results": [self._active_recall_public_item(item) for item in selected],
            "raw_refs": raw_refs,
            "diagnostics": {
                "policy": "lifecycle_recency",
                "query_required": False,
                "answer_model_used": False,
                "candidate_count": len(candidates),
                "semantic_focus_used": False,
                "semantic_count": 0,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            },
        }

    def _active_recall_candidates(
        self,
        agent_id: str,
        *,
        include_content: bool,
        limit: int,
    ) -> list[dict[str, Any]]:
        if include_content:
            rows = self.long_term.conn.execute(
                """
                SELECT l.item_id, l.source_type, l.source_id, l.content_id,
                       l.preview, l.metadata, l.indexed_at, c.content
                FROM long_term_items l
                JOIN content_store c ON c.content_id = l.content_id
                WHERE l.agent_id = ?
                ORDER BY l.indexed_at DESC
                LIMIT ?
                """,
                (agent_id, limit),
            ).fetchall()
        else:
            rows = self.long_term.conn.execute(
                """
                SELECT item_id, source_type, source_id, content_id,
                       preview, metadata, indexed_at, NULL
                FROM long_term_items
                WHERE agent_id = ?
                ORDER BY indexed_at DESC
                LIMIT ?
                """,
                (agent_id, limit),
            ).fetchall()
        candidates = []
        for row in rows:
            metadata = _json_dict(row[5])
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
    def _active_recall_score(
        item: dict[str, Any],
        *,
        session_id: str | None,
    ) -> tuple[float, list[str]]:
        metadata = dict(item.get("metadata") or {})
        source_type = str(item.get("source_type") or "memory")
        score = 1.0
        reasons = [active_recall_reason(source_type)]
        if source_type == "correction" or metadata.get(MetadataField.KIND) == "correction":
            score += 4.0
            reasons.append("correction")
        elif source_type == "contradiction":
            score += 1.5
            reasons.append("contradiction")

        confidence = str(metadata.get(MetadataField.CONFIDENCE) or "")
        if confidence == MemoryConfidence.HIGH:
            score += 0.5
            reasons.append("high_confidence")
        elif confidence == MemoryConfidence.LOW:
            score -= 3.0
            reasons.append("low_confidence")

        if metadata.get(MetadataField.CONTROLLER_IGNORED) is True:
            score -= 10.0
            reasons.append("controller_ignored")
        if metadata.get(MetadataField.VALID_TO) or metadata.get(MetadataField.SUPERSEDED_BY):
            score -= 4.0
            reasons.append("superseded_or_stale")
        if metadata.get(MetadataField.SUPERSEDES) or metadata.get(MetadataField.CONTRADICTS):
            score += 0.35
            reasons.append("contradiction_tracking")
        if session_id and metadata.get(MetadataField.SESSION_ID) == session_id:
            score += 0.35
            reasons.append("same_session")

        freshness = metadata.get(MetadataField.FRESHNESS)
        if freshness not in (None, "", "unknown"):
            score += 0.15
            reasons.append("freshness_available")

        return score, _dedupe_ordered(reasons)

    @staticmethod
    def _active_recall_public_item(item: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "item_id": item["item_id"],
            "raw_ref": item["raw_ref"],
            "source_type": item["source_type"],
            "source_id": item["source_id"],
            "content_id": item["content_id"],
            "preview": item["preview"],
            "metadata": item["metadata"],
            "active_recall_score": item["active_recall_score"],
            "reasons": item["reasons"],
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
