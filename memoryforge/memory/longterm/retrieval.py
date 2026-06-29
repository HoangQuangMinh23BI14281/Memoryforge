"""Long-term memory retrieval streams and fusion."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

from memoryforge.memory.longterm.models import (
    LongTermRecallResult,
    MetadataField,
)
from memoryforge.memory.longterm.utils import tokenize_query

_ANSWER_EVIDENCE_SNIPPET_CHARS = 1_100

if TYPE_CHECKING:
    from memoryforge.search.vector import VectorIndex


class LongTermRetrievalMixin:
    if TYPE_CHECKING:
        conn: sqlite3.Connection
        _has_fts: bool
        vector: VectorIndex

    def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        *,
        include_content: bool = False,
        session_id: str | None = None,
    ) -> list[LongTermRecallResult]:
        limit = max(1, top_k)
        vector_results = self._vector_search(agent_id, query, limit * 4)
        bm25_results = self._bm25_search(agent_id, query, limit * 4)
        streams = {
            "bm25": bm25_results,
            "vector": vector_results,
        }
        fused_ids, stream_details = self._rrf(streams)
        selected_ids = self._select_ensemble_ids(fused_ids, streams, limit * 2)
        rows = self._fetch_items(selected_ids, include_content)
        results: list[LongTermRecallResult] = []
        fused_scores = dict(fused_ids)
        selection_details = self._selection_details(
            selected_ids=selected_ids,
            streams=streams,
            fused_scores=fused_scores,
        )
        for item_id in selected_ids:
            row = rows.get(item_id)
            if row is None:
                continue
            score = fused_scores.get(item_id, 0.0)
            final_streams = dict(stream_details.get(item_id, {}))
            final_streams["fusion"] = {"score": score}
            final_streams["selection"] = selection_details.get(item_id, {})
            results.append(
                LongTermRecallResult(
                    item_id=item_id,
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    content_id=row["content_id"],
                    raw_ref=f"{row['source_type']}:{row['source_id']}",
                    preview=row["preview"],
                    score=score,
                    streams=final_streams,
                    metadata=json.loads(row["metadata"] or "{}"),
                    content=row.get("content"),
                )
            )
        return results[:limit]

    def get_source(self, agent_id: str, item_id: str) -> dict[str, Any] | None:
        rows = self._fetch_items([item_id], include_content=True)
        row = rows.get(item_id)
        if row is None or row["agent_id"] != agent_id:
            return None
        return {
            "item_id": row["item_id"],
            "source_type": row["source_type"],
            "source_id": row["source_id"],
            "content_id": row["content_id"],
            "raw_ref": f"{row['source_type']}:{row['source_id']}",
            "content": row.get("content"),
            "metadata": json.loads(row["metadata"] or "{}"),
        }

    def format_recall_block(
        self,
        results: list[LongTermRecallResult],
        *,
        content_policy: str = "auto",
        query: str = "",
    ) -> str:
        if not results:
            return "[MemoryForge long-term recall]\nNo matching long-term memory."
        lines = ["[MemoryForge long-term recall: refs point to immutable raw content]"]
        for index, result in enumerate(results, start=1):
            stream_names = ", ".join(sorted(result.streams)) or "index"
            content = _recall_text(result, content_policy, query=query)
            lines.append(
                f"{index}. {result.raw_ref} ({stream_names}, score={result.score:.4f})\n"
                f"   {content}"
            )
        return "\n".join(lines)

    def recall_injection_diagnostics(
        self,
        results: list[LongTermRecallResult],
        *,
        content_policy: str = "auto",
        query: str = "",
    ) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        for result in results:
            source_text = result.content if result.content is not None else result.preview
            injected_text = _recall_text(result, content_policy, query=query)
            effective_policy = _effective_recall_text_policy(
                result,
                content_policy,
                injected_text=injected_text,
                source_text=source_text,
            )
            diagnostics.append(
                {
                    "raw_ref": result.raw_ref,
                    "requested_policy": content_policy,
                    "effective_policy": effective_policy,
                    "stream_champion": _is_stream_champion(result),
                    "source_chars": len(source_text),
                    "injected_chars": len(injected_text),
                    "truncated": len(injected_text) < len(source_text),
                    "streams": sorted(result.streams),
                }
            )
        return diagnostics

    def count(self, agent_id: str | None = None) -> int:
        if agent_id is None:
            row = self.conn.execute("SELECT COUNT(*) FROM long_term_items").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM long_term_items WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def _bm25_search(self, agent_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        tokens = tokenize_query(query)
        if self._has_fts and tokens:
            fts_query = " OR ".join(tokens)
            try:
                rows = self.conn.execute(
                    """
                    SELECT source_id, rank
                    FROM search_fts
                    WHERE scope = 'long_term' AND agent_id = ? AND search_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (agent_id, fts_query, limit),
                ).fetchall()
                return [(str(row[0]), max(0.0, -float(row[1]))) for row in rows]
            except sqlite3.OperationalError:
                pass
        return self._like_search(agent_id, tokens, limit)

    def _like_search(
        self,
        agent_id: str,
        tokens: list[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        clauses = ["scope = 'long_term'", "agent_id = ?"]
        params: list[Any] = [agent_id]
        rows = self.conn.execute(
            f"""
            SELECT source_id, content
            FROM search_fts
            WHERE {" AND ".join(clauses)}
            LIMIT 2000
            """,
            params,
        ).fetchall()
        scored: list[tuple[str, float]] = []
        for item_id, content in rows:
            text = str(content).lower()
            score = sum(text.count(token) for token in tokens)
            if score:
                scored.append((str(item_id), float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _vector_search(self, agent_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        content_results = self.vector.search(query, limit * 4)
        if not content_results:
            return []
        content_scores = {content_id: score for content_id, score in content_results}
        placeholders = ",".join("?" for _content_id in content_scores)
        rows = self.conn.execute(
            f"""
            SELECT item_id, content_id
            FROM long_term_items
            WHERE agent_id = ? AND content_id IN ({placeholders})
            """,
            (agent_id, *content_scores.keys()),
        ).fetchall()
        ranked = [(str(row[0]), float(content_scores[str(row[1])])) for row in rows]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def _fetch_items(self, item_ids: list[str], include_content: bool) -> dict[str, dict[str, Any]]:
        if not item_ids:
            return {}
        placeholders = ",".join("?" for _item_id in item_ids)
        if include_content:
            rows = self.conn.execute(
                f"""
                SELECT l.item_id, l.agent_id, l.source_type, l.source_id, l.content_id,
                       l.preview, l.metadata, c.content
                FROM long_term_items l
                JOIN content_store c ON c.content_id = l.content_id
                WHERE l.item_id IN ({placeholders})
                """,
                item_ids,
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"""
                SELECT item_id, agent_id, source_type, source_id, content_id,
                       preview, metadata, NULL
                FROM long_term_items
                WHERE item_id IN ({placeholders})
                """,
                item_ids,
            ).fetchall()
        return {
            str(row[0]): {
                "item_id": str(row[0]),
                "agent_id": str(row[1]),
                "source_type": str(row[2]),
                "source_id": str(row[3]),
                "content_id": str(row[4]),
                "preview": str(row[5] or ""),
                "metadata": str(row[6] or "{}"),
                "content": row[7],
            }
            for row in rows
        }

    @staticmethod
    def _rrf(
        streams: dict[str, list[tuple[str, float]]],
        weights: dict[str, float] | None = None,
        rrf_k: float = 60.0,
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, dict[str, float | int]]]]:
        resolved_weights = weights or {
            "bm25": 0.5,
            "vector": 0.5,
        }
        active_streams = {
            stream_name: list(results)
            for stream_name, results in streams.items()
            if results and resolved_weights.get(stream_name, 0.0) > 0
        }
        if not active_streams:
            return [], {}
        total_weight = sum(resolved_weights.get(stream_name, 0.0) for stream_name in active_streams)
        scores: dict[str, float] = {}
        details: dict[str, dict[str, dict[str, float | int]]] = {}
        for stream_name, results in active_streams.items():
            weight = resolved_weights.get(stream_name, 0.0) / total_weight
            for rank, (item_id, raw_score) in enumerate(results, start=1):
                contribution = weight / (rrf_k + rank)
                scores[item_id] = scores.get(item_id, 0.0) + contribution
                details.setdefault(item_id, {})[stream_name] = {
                    "rank": rank,
                    "score": float(raw_score),
                    "contribution": contribution,
                }
        return sorted(scores.items(), key=lambda item: item[1], reverse=True), details

    @staticmethod
    def _select_ensemble_ids(
        fused_ids: list[tuple[str, float]],
        streams: dict[str, list[tuple[str, float]]],
        limit: int,
    ) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()

        for stream_name in sorted(streams):
            stream_results = streams.get(stream_name) or []
            if stream_results:
                item_id = stream_results[0][0]
                if item_id not in seen:
                    selected.append(item_id)
                    seen.add(item_id)
                    if len(selected) >= limit:
                        return selected

        for item_id, _score in fused_ids:
            if item_id in seen:
                continue
            selected.append(item_id)
            seen.add(item_id)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _selection_details(
        *,
        selected_ids: list[str],
        streams: dict[str, list[tuple[str, float]]],
        fused_scores: dict[str, float],
    ) -> dict[str, dict[str, float | int]]:
        stream_champions = {
            stream_name: results[0][0]
            for stream_name, results in streams.items()
            if results
        }
        details: dict[str, dict[str, float | int]] = {}
        for selected_rank, item_id in enumerate(selected_ids, start=1):
            item_details: dict[str, float | int] = {
                "selected_rank": selected_rank,
                "fused_score": fused_scores.get(item_id, 0.0),
            }
            for stream_name, champion_id in stream_champions.items():
                if item_id == champion_id:
                    item_details[f"{stream_name}_champion"] = 1
            details[item_id] = item_details
        return details


def _recall_text(result: LongTermRecallResult, content_policy: str, *, query: str = "") -> str:
    if content_policy == "full":
        return result.content if result.content is not None else result.preview
    if content_policy == "champion" and _is_stream_champion(result):
        return result.content if result.content is not None else result.preview
    if content_policy == "legacy":
        return result.content if result.content is not None else result.preview
    if content_policy == "preview":
        return result.preview
    if content_policy == "auto":
        if _is_stream_champion(result):
            return result.content if result.content is not None else result.preview
        answer_range_snippet = _answer_range_snippet(result)
        if answer_range_snippet:
            return answer_range_snippet
        return result.preview
    if content_policy == "snippet":
        answer_range_snippet = _answer_range_snippet(result)
        if answer_range_snippet:
            return answer_range_snippet
        return result.preview
    return result.preview


def _effective_recall_text_policy(
    result: LongTermRecallResult,
    content_policy: str,
    *,
    injected_text: str,
    source_text: str,
) -> str:
    if injected_text == source_text and result.content is not None:
        return "full"
    if content_policy == "preview":
        return "preview"
    if content_policy in {"snippet", "auto"} and injected_text != result.preview:
        return "snippet"
    if content_policy == "champion" and _is_stream_champion(result) and result.content is not None:
        return "full"
    return "preview"


def _is_stream_champion(result: LongTermRecallResult) -> bool:
    return any(int(detail.get("rank", 0) or 0) == 1 for detail in result.streams.values())


def _snippet_max_chars(result: LongTermRecallResult) -> int:
    metadata = result.metadata or {}
    if metadata.get(MetadataField.ANSWER_EVIDENCE) is True:
        return _ANSWER_EVIDENCE_SNIPPET_CHARS
    return 720


def _answer_range_snippet(result: LongTermRecallResult) -> str:
    ranges = _answer_evidence_ranges(result)
    if not ranges:
        return ""
    return _range_focused_snippet(
        result,
        ranges=ranges,
        max_chars=_snippet_max_chars(result),
    )


def _answer_evidence_ranges(result: LongTermRecallResult) -> list[tuple[int, int]]:
    metadata = result.metadata or {}
    raw_ranges = metadata.get(MetadataField.ANSWER_EVIDENCE_RANGES)
    if not isinstance(raw_ranges, list):
        return []
    ranges: list[tuple[int, int]] = []
    for raw_range in raw_ranges:
        if not isinstance(raw_range, dict):
            continue
        start = _optional_int(raw_range.get("start"))
        end = _optional_int(raw_range.get("end"))
        if start is None or end is None or start >= end:
            continue
        ranges.append((start, end))
    return ranges


def _range_focused_snippet(
    result: LongTermRecallResult,
    *,
    ranges: list[tuple[int, int]],
    max_chars: int,
) -> str:
    text = result.content or result.preview
    if len(text) <= max_chars:
        return text
    start, end = ranges[0]
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    center = (start + end) // 2
    snippet_start = max(0, center - max_chars // 2)
    snippet_end = min(len(text), snippet_start + max_chars)
    snippet_start = max(0, snippet_end - max_chars)
    return _trimmed_snippet(text, snippet_start, snippet_end)


def _trimmed_snippet(text: str, start: int, end: int) -> str:
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"... {snippet}"
    if end < len(text):
        snippet = f"{snippet} ..."
    return snippet


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
