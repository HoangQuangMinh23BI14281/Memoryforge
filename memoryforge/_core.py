"""Pure-Python/SQLite core primitives for MemoryForge."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

from memoryforge.search.fts import ensure_search_fts


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


def _hash_content(content: str) -> str:
    return hashlib.blake2b(content.encode("utf-8"), digest_size=32).hexdigest()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000):016x}_{uuid.uuid4().hex[:8]}"


def _tokenize_query(query: str) -> list[str]:
    """Tokenize query for FTS and LIKE fallback search."""
    tokens = re.findall(r"[\w]+", query.lower())

    normalized = []
    for token in tokens:
        if len(token) > 4:
            if token.endswith("ing"):
                normalized.append(token[:-3])
            elif token.endswith("ed"):
                normalized.append(token[:-2])
            elif token.endswith("s") and not token.endswith("ss"):
                normalized.append(token[:-1])
        normalized.append(token)

    return list(set(normalized))


class ContentHashTable:
    """SQLite-backed content-addressed store with reference counting."""

    def __init__(self, db_path: str):
        self.conn = _connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS content_store (
                content_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                token_count INTEGER,
                reference_count INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_hash ON content_store(content_hash)"
        )
        self._ensure_column("content_store", "reference_count", "INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("content_store", "last_accessed", "REAL NOT NULL DEFAULT 0")
        self.conn.execute(
            "UPDATE content_store SET last_accessed = created_at WHERE last_accessed = 0"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reference_count ON content_store(reference_count)"
        )
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self.conn.commit()

    def store(self, content: str) -> tuple[str, bool]:
        content_hash = _hash_content(content)
        now = time.time()

        row = self.conn.execute(
            "SELECT content_id, reference_count FROM content_store WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()

        if row:
            content_id = str(row[0])
            self.conn.execute(
                """
                UPDATE content_store
                SET reference_count = reference_count + 1, last_accessed = ?
                WHERE content_id = ?
                """,
                (now, content_id),
            )
            self.conn.commit()
            return content_id, False

        content_id = _new_id("cnt")
        try:
            self.conn.execute(
                """
                INSERT INTO content_store
                (content_id, content, content_hash, token_count, reference_count, created_at, last_accessed)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                """,
                (content_id, content, content_hash, len(content) // 4, now, now),
            )
            self.conn.commit()
            return content_id, True
        except sqlite3.IntegrityError:
            self.conn.rollback()
            row = self.conn.execute(
                "SELECT content_id FROM content_store WHERE content_hash = ? LIMIT 1",
                (content_hash,),
            ).fetchone()
            if row:
                content_id = str(row[0])
                self.conn.execute(
                    """
                    UPDATE content_store
                    SET reference_count = reference_count + 1, last_accessed = ?
                    WHERE content_id = ?
                    """,
                    (now, content_id),
                )
                self.conn.commit()
                return content_id, False
            raise

    def release(self, content_id: str) -> bool:
        row = self.conn.execute(
            "SELECT reference_count FROM content_store WHERE content_id = ?",
            (content_id,),
        ).fetchone()

        if not row:
            import sys

            print(
                f"Warning: Attempted to release non-existent content_id: {content_id}",
                file=sys.stderr,
            )
            return False

        ref_count = int(row[0])

        if ref_count <= 0:
            print(f"Warning: Content {content_id} already has zero references", file=sys.stderr)
            return False

        if ref_count == 1:
            self.conn.execute("DELETE FROM content_store WHERE content_id = ?", (content_id,))
            self.conn.commit()
            return True
        else:
            self.conn.execute(
                "UPDATE content_store SET reference_count = reference_count - 1 WHERE content_id = ?",
                (content_id,),
            )
            self.conn.commit()
            return False

    def cleanup_unreferenced(self, older_than_days: int = 30) -> int:
        cutoff = time.time() - (older_than_days * 86400)
        cursor = self.conn.execute(
            """
            DELETE FROM content_store
            WHERE reference_count = 0 AND last_accessed < ?
            RETURNING content_id
            """,
            (cutoff,),
        )
        deleted_ids = cursor.fetchall()
        cursor.close()
        deleted = len(deleted_ids)
        self.conn.commit()
        return deleted

    def retrieve(self, content_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT content FROM content_store WHERE content_id = ?",
            (content_id,),
        ).fetchone()
        if row:
            # Update last_accessed timestamp on retrieval
            self.conn.execute(
                "UPDATE content_store SET last_accessed = ? WHERE content_id = ?",
                (time.time(), content_id),
            )
            self.conn.commit()
            return str(row[0])
        return self.retrieve_by_hash(content_id)

    def retrieve_by_hash(self, content_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT content FROM content_store WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if row and row[0] is not None:
            return str(row[0])
        return None

    def get_by_id(self, content_id: str) -> str | None:
        return self.retrieve(content_id)


class BM25Index:
    """SQLite FTS5-backed BM25 index with a LIKE fallback."""

    def __init__(self, db_path: str):
        self.conn = _connect(db_path)
        self._has_fts = ensure_search_fts(self.conn)

    def index_turn(
        self,
        agent_id: str,
        session_id: str,
        role: str,
        content: str,
        content_id: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO search_fts
            (content, scope, agent_id, source_type, source_id, content_id, session_id, role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content,
                "conversation",
                agent_id,
                "message",
                content_id,
                content_id,
                session_id,
                role,
            ),
        )
        self.conn.commit()

    def search(self, agent_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        if self._has_fts:
            tokens = _tokenize_query(query)
            if tokens:
                fts_query = " OR ".join(tokens)
                try:
                    rows = self.conn.execute(
                        """
                        SELECT content_id, rank
                        FROM search_fts
                        WHERE scope = 'conversation' AND agent_id = ? AND search_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (agent_id, fts_query, limit),
                    ).fetchall()
                    return [(str(row[0]), max(0.0, -float(row[1]))) for row in rows]
                except sqlite3.OperationalError:
                    pass
        return self._search_like(agent_id, query, limit)

    def _search_like(self, agent_id: str, query: str, limit: int) -> list[tuple[str, float]]:
        tokens = _tokenize_query(query)
        rows = self.conn.execute(
            """
            SELECT content_id, content
            FROM search_fts
            WHERE scope = 'conversation' AND agent_id = ?
            LIMIT 1000
            """,
            (agent_id,),
        ).fetchall()
        scored = []
        for content_id, content in rows:
            text = str(content).lower()
            score = sum(text.count(token) for token in tokens)
            if score:
                scored.append((str(content_id), float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


def rrf_fusion(
    streams: Mapping[str, Iterable[tuple[str, float]]],
    weights: Mapping[str, float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion for any number of named retrieval streams."""

    active_streams = {name: list(results) for name, results in streams.items()}
    total_weight = sum(weights.get(name, 0.0) for name in active_streams)
    if total_weight <= 0:
        raise ValueError("Sum of weights must be positive")

    weighted_streams: list[tuple[float, list[tuple[str, float]]]] = []
    for name, results in active_streams.items():
        weight = weights.get(name, 0.0) / total_weight
        weighted_streams.append((weight, results))

    fused_scores: dict[str, float] = {}
    for weight, results in weighted_streams:
        for rank, (item_id, _score) in enumerate(results, start=1):
            rrf_component = weight / (k + rank)
            fused_scores[item_id] = fused_scores.get(item_id, 0.0) + rrf_component

    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)


def normalize_scores(scores: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    items = list(scores)
    if not items:
        return []
    values = [score for _item_id, score in items]
    min_score = min(values)
    max_score = max(values)
    if abs(max_score - min_score) < 1e-9:
        return [(item_id, 1.0) for item_id, _score in items]
    return [(item_id, (score - min_score) / (max_score - min_score)) for item_id, score in items]


class HnswIndex:
    """Small in-process vector index with an HNSW-compatible API."""

    def __init__(
        self,
        dimensions: int = 256,
        connectivity: int = 16,
        expansion_add: int = 128,
        expansion_search: int = 64,
    ):
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self.dimensions = dimensions
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self._vectors: dict[int, list[float]] = {}

    @staticmethod
    def is_available() -> bool:
        return False

    def add(self, item_id: int, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(vector)}")
        self._vectors[int(item_id)] = [float(value) for value in vector]

    def search(self, query: list[float], limit: int) -> list[tuple[int, float]]:
        if len(query) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(query)}")
        scored = [
            (item_id, self._cosine(query, vector)) for item_id, vector in self._vectors.items()
        ]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def remove(self, item_id: int) -> bool:
        return self._vectors.pop(int(item_id), None) is not None

    def len(self) -> int:
        return len(self._vectors)

    def is_empty(self) -> bool:
        return not self._vectors

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        if denominator <= 0:
            return 0.0
        return sum(lvalue * rvalue for lvalue, rvalue in zip(left, right)) / denominator
