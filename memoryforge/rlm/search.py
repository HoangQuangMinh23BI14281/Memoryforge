"""Search over RLM chunks."""

from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING, Any

from memoryforge._core import rrf_fusion
from memoryforge.rlm.common import is_transient_sqlite_error, tokenize

if TYPE_CHECKING:
    from memoryforge.search.vector import VectorIndex


class RLMSearchMixin:
    if TYPE_CHECKING:
        conn: sqlite3.Connection
        _has_fts: bool

        def get_chunk(
            self, chunk_id: str, *, include_content: bool = True
        ) -> dict[str, Any] | None: ...

        def _vector_index(self) -> VectorIndex | None: ...

    def search(
        self,
        agent_id: str,
        query: str,
        *,
        buffer_id: str | None = None,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Search RLM chunks and return chunk IDs plus previews, not full content."""

        mode = mode.lower()
        bm25_results = self._search_bm25(agent_id, query, buffer_id, limit * 3)
        vector_results = self._search_vector(agent_id, query, buffer_id, limit * 3)

        if mode == "bm25":
            ranked = bm25_results
        elif mode in {"semantic", "vector"}:
            ranked = vector_results
        else:
            ranked = rrf_fusion(
                {"bm25": bm25_results, "vector": vector_results},
                {"bm25": 1.0, "vector": 1.0},
            )

        results = []
        for chunk_id, score in ranked:
            chunk = self.get_chunk(chunk_id, include_content=False)
            if chunk is None:
                continue
            chunk["score"] = score
            results.append(chunk)
            if len(results) >= limit:
                break
        return results

    def _search_bm25(
        self,
        agent_id: str,
        query: str,
        buffer_id: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)
        if self._has_fts:
            try:
                clauses = ["scope = 'rlm_chunk'", "agent_id = ?", "search_fts MATCH ?"]
                params: list[Any] = [agent_id, fts_query]
                if buffer_id:
                    clauses.append("buffer_id = ?")
                    params.append(buffer_id)
                rows = self.conn.execute(
                    f"""
                    SELECT source_id, rank
                    FROM search_fts
                    WHERE {" AND ".join(clauses)}
                    ORDER BY rank
                    LIMIT ?
                    """,
                    [*params, limit],
                ).fetchall()
                return [(str(row[0]), max(0.0, -float(row[1]))) for row in rows]
            except sqlite3.OperationalError:
                pass
        return self._search_like(agent_id, query, buffer_id, limit)

    def _search_like(
        self,
        agent_id: str,
        query: str,
        buffer_id: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        clauses = ["scope = 'rlm_chunk'", "agent_id = ?"]
        params: list[Any] = [agent_id]
        if buffer_id:
            clauses.append("buffer_id = ?")
            params.append(buffer_id)
        rows: list[tuple[Any, Any]] | None = None
        fallback_used = False
        for attempt in range(5):
            try:
                rows = self.conn.execute(
                    f"""
                    SELECT source_id, content
                    FROM search_fts
                    WHERE {" AND ".join(clauses)}
                    LIMIT 5000
                    """,
                    params,
                ).fetchall()
                break
            except sqlite3.OperationalError as exc:
                if attempt < 4 and is_transient_sqlite_error(exc):
                    time.sleep(0.05 * (attempt + 1))
                    continue
                rows = self._searchable_chunk_rows(agent_id, buffer_id)
                fallback_used = True
                break
        if rows is None and not fallback_used:
            rows = self._searchable_chunk_rows(agent_id, buffer_id)
        if rows is None:
            rows = []
        scored = []
        for chunk_id, content in rows:
            score = sum(str(content).lower().count(token) for token in tokens)
            if score:
                scored.append((str(chunk_id), float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _searchable_chunk_rows(
        self,
        agent_id: str,
        buffer_id: str | None,
    ) -> list[tuple[str, str]]:
        clauses = ["c.agent_id = ?"]
        params: list[Any] = [agent_id]
        if buffer_id:
            clauses.append("c.buffer_id = ?")
            params.append(buffer_id)
        try:
            return [
                (str(row[0]), str(row[1]))
                for row in self.conn.execute(
                    f"""
                    SELECT c.chunk_id, s.content
                    FROM rlm_chunks c
                    JOIN content_store s ON s.content_id = c.content_id
                    WHERE {" AND ".join(clauses)}
                    ORDER BY c.buffer_id, c.chunk_index
                    LIMIT 5000
                    """,
                    params,
                ).fetchall()
            ]
        except sqlite3.DatabaseError:
            return []

    def _search_vector(
        self,
        agent_id: str,
        query: str,
        buffer_id: str | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        vector = self._vector_index()
        if vector is None:
            return []
        content_results = vector.search(query, limit * 4)
        if not content_results:
            return []
        content_scores = {content_id: score for content_id, score in content_results}
        placeholders = ",".join("?" for _ in content_scores)
        clauses = [f"content_id IN ({placeholders})", "agent_id = ?"]
        params: list[Any] = [*content_scores.keys(), agent_id]
        if buffer_id:
            clauses.append("buffer_id = ?")
            params.append(buffer_id)
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, content_id
            FROM rlm_chunks
            WHERE {" AND ".join(clauses)}
            """,
            params,
        ).fetchall()
        ranked = [(str(row[0]), float(content_scores[str(row[1])])) for row in rows]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]
