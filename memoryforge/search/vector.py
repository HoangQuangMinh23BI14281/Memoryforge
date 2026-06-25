"""Local vector search with optional FastEmbed and lexical failover."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Protocol

from memoryforge._core import HnswIndex

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


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


class _Embedder(Protocol):
    dimensions: int
    model_name: str

    def encode(self, text: str) -> list[float]: ...

    def encode_many(self, texts: list[str]) -> list[list[float]]: ...


def _coerce_embedding(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    if embedding and isinstance(embedding[0], list):
        embedding = embedding[0]
    return [float(value) for value in embedding]


def _coerce_embedding_batch(embeddings: Any) -> list[list[float]]:
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    return [_coerce_embedding(embedding) for embedding in embeddings]


def _content_hash(content: str) -> str:
    return hashlib.blake2b(content.encode("utf-8"), digest_size=32).hexdigest()


def _fastembed_model_name(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_FASTEMBED_MODEL
    if model_name == "all-MiniLM-L6-v2":
        return "sentence-transformers/all-MiniLM-L6-v2"
    return model_name


class _FastEmbedder:
    """FastEmbed wrapper with dimensions discovered from the loaded model."""

    def __init__(self, model_name: str | None):
        resolved_model = _fastembed_model_name(model_name)
        try:
            fastembed = importlib.import_module("fastembed")
        except Exception as exc:
            raise RuntimeError(
                "FastEmbed vector backend requires the optional 'fastembed' dependency"
            ) from exc

        self._model = fastembed.TextEmbedding(model_name=resolved_model)
        self.model_name = resolved_model
        self.dimensions = len(self.encode("MemoryForge embedding dimension probe"))

    def encode(self, text: str) -> list[float]:
        embeddings = iter(self.encode_many([text]))
        try:
            return next(embeddings)
        except StopIteration as exc:
            raise RuntimeError("FastEmbed returned no embeddings") from exc

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.embed([text or " " for text in texts])
        encoded = _coerce_embedding_batch(list(embeddings))
        if len(encoded) != len(texts):
            raise RuntimeError(
                f"FastEmbed returned {len(encoded)} embeddings for {len(texts)} inputs"
            )
        return encoded


class _DisabledEmbedder:
    """No-op embedder used when semantic vector search is unavailable."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions
        self.model_name = "disabled"

    def encode(self, text: str) -> list[float]:
        raise RuntimeError("Vector search is disabled because no embedding model is available")

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Vector indexing is disabled because no embedding model is available")


class VectorIndex:
    """SQLite-persistent vector index.

    Uses a real configured embedding backend when available. If no semantic
    model can be loaded, vector search is disabled and callers should rely on
    lexical/BM25 retrieval. Hashing is intentionally not used for semantic
    search.
    """

    def __init__(
        self,
        db_path: str,
        model_name: str | None = None,
        backend: str | None = None,
        dimensions: int = 256,
    ):
        self.db_path = str(Path(db_path).expanduser())
        self.conn = _connect(self.db_path)
        self.model, self.embedding_backend = self._load_model(model_name, backend, dimensions)
        self.enabled = self.embedding_backend != "disabled"
        self.dimensions = int(self.model.dimensions)
        configured_model = model_name or os.environ.get("MEMORYFORGE_VECTOR_MODEL")
        model_label = getattr(self.model, "model_name", None) or configured_model or "disabled"
        self.model_name = str(model_label)
        self.model_key = f"{self.embedding_backend}:{self.model_name}"
        self.index = HnswIndex(self.dimensions)
        self.last_add_stats = {"requested": 0, "encoded": 0, "cached": 0}
        self._init_schema()
        self._load_vector_index()

    def _load_model(
        self, model_name: str | None, backend: str | None, dimensions: int
    ) -> tuple[_Embedder, str]:
        resolved_model = model_name or os.environ.get("MEMORYFORGE_VECTOR_MODEL")
        resolved_backend = (
            backend or os.environ.get("MEMORYFORGE_VECTOR_BACKEND") or "auto"
        ).lower()
        require_vector_model = os.environ.get("MEMORYFORGE_REQUIRE_VECTOR_MODEL") == "1"
        if resolved_backend in {"disabled", "none", "off"}:
            return _DisabledEmbedder(dimensions=dimensions), "disabled"
        if resolved_backend in {"hash", "hashing"}:
            raise RuntimeError("Hashing vector backend is not supported for semantic search")
        if resolved_backend not in {"auto", "fastembed"}:
            raise RuntimeError(
                f"Unsupported vector backend {resolved_backend!r}. "
                "MemoryForge supports 'fastembed' or lexical-only fallback."
            )
        if not resolved_model and resolved_backend == "auto":
            if require_vector_model:
                raise RuntimeError(
                    "A required vector model needs MEMORYFORGE_VECTOR_MODEL "
                    "or VectorIndex(model_name=...)"
                )
            return _DisabledEmbedder(dimensions=dimensions), "disabled"
        if not resolved_model:
            resolved_model = DEFAULT_FASTEMBED_MODEL

        try:
            return _FastEmbedder(resolved_model), "fastembed"
        except Exception as exc:
            if require_vector_model or resolved_backend == "fastembed":
                raise RuntimeError(
                    f"Failed to load required vector model {resolved_model!r} "
                    f"with FastEmbed: {exc}"
                ) from exc
            return _DisabledEmbedder(dimensions=dimensions), "disabled"

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vec_index (
                cache_key TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                numeric_id INTEGER NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._migrate_vec_index_cache_key()
        self._migrate_vec_index_content_hash_cache_keys()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vec_index_content_model "
            "ON vec_index(content_id, model)"
        )
        self.conn.commit()

    def _migrate_vec_index_cache_key(self) -> None:
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(vec_index)")]
        if "cache_key" in columns:
            return
        rows = self.conn.execute(
            """
            SELECT content_id, embedding, dimensions, model, created_at
            FROM vec_index
            """
        ).fetchall()
        self.conn.execute("ALTER TABLE vec_index RENAME TO vec_index_legacy")
        self.conn.execute(
            """
            CREATE TABLE vec_index (
                cache_key TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                numeric_id INTEGER NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO vec_index
            (cache_key, content_id, numeric_id, embedding, dimensions, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    self._cache_key(str(content_id), str(model)),
                    str(content_id),
                    self._numeric_id(self._cache_key(str(content_id), str(model))),
                    embedding,
                    int(dimensions),
                    str(model),
                    float(created_at),
                )
                for content_id, embedding, dimensions, model, created_at in rows
            ],
        )
        self.conn.execute("DROP TABLE vec_index_legacy")

    def _migrate_vec_index_content_hash_cache_keys(self) -> None:
        table = self.conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'content_store'
            """
        ).fetchone()
        if table is None:
            return
        rows = self.conn.execute(
            """
            SELECT v.cache_key, v.content_id, v.model, c.content_hash
            FROM vec_index v
            JOIN content_store c ON c.content_id = v.content_id
            """
        ).fetchall()
        with self.conn:
            for cache_key, content_id, model, content_hash in rows:
                expected = self._cache_key(str(content_hash), str(model))
                if str(cache_key) == expected:
                    continue
                existing = self.conn.execute(
                    "SELECT 1 FROM vec_index WHERE cache_key = ?",
                    (expected,),
                ).fetchone()
                if existing:
                    self.conn.execute("DELETE FROM vec_index WHERE cache_key = ?", (cache_key,))
                    continue
                self.conn.execute(
                    """
                    UPDATE vec_index
                    SET cache_key = ?, numeric_id = ?
                    WHERE cache_key = ?
                    """,
                    (expected, self._numeric_id(expected), cache_key),
                )

    def add(self, content_id: str, content: str) -> None:
        self.add_many([(content_id, content)])

    def add_many(self, items: list[tuple[str, str]]) -> int:
        """Add embeddings in a batch, skipping cached rows for this model identity."""

        deduped: dict[str, str] = {}
        for content_id, content in items:
            deduped[str(content_id)] = str(content)
        requested = len(deduped)
        if not deduped:
            self.last_add_stats = {"requested": 0, "encoded": 0, "cached": 0}
            return 0
        if not self.enabled:
            self.last_add_stats = {
                "requested": requested,
                "encoded": 0,
                "cached": 0,
            }
            return 0

        existing = self._existing_embeddings(list(deduped.items()))
        to_encode: list[tuple[str, str]] = []
        for content_id, content in deduped.items():
            row = existing.get(content_id)
            if row is None:
                to_encode.append((content_id, content))
                continue
            numeric_id, embedding_bytes, dimensions, model = row
            if int(dimensions) == self.dimensions and str(model) == self.model_key:
                embedding = self._unpack(bytes(embedding_bytes), int(dimensions))
                self.index.add(int(numeric_id), embedding)
            else:
                to_encode.append((content_id, content))

        if not to_encode:
            self.last_add_stats = {
                "requested": requested,
                "encoded": 0,
                "cached": requested,
            }
            return 0

        embeddings = self._encode_many([content for _content_id, content in to_encode])
        rows = []
        now = time.time()
        for (content_id, _content), embedding in zip(to_encode, embeddings):
            cache_key = self._embedding_cache_key(content_id, _content)
            numeric_id = self._numeric_id(cache_key)
            rows.append(
                (
                    cache_key,
                    content_id,
                    numeric_id,
                    self._pack(embedding),
                    len(embedding),
                    self.model_key,
                    now,
                )
            )
            self.index.add(numeric_id, embedding)

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO vec_index
            (cache_key, content_id, numeric_id, embedding, dimensions, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        self.last_add_stats = {
            "requested": requested,
            "encoded": len(rows),
            "cached": requested - len(rows),
        }
        return len(rows)

    def search(
        self, query: str, limit: int = 20, agent_id: str | None = None
    ) -> list[tuple[str, float]]:
        if not self.enabled:
            return []
        query_embedding = self._encode(query)
        vector_results = self.index.search(query_embedding, limit * 4 if agent_id else limit)
        if vector_results:
            content_ids = self._content_ids_for_numeric(
                [item_id for item_id, _score in vector_results]
            )
            results = [
                (content_ids[item_id], score)
                for item_id, score in vector_results
                if item_id in content_ids
            ]
            if agent_id:
                results = self._filter_by_agent(results, agent_id)
            return results[:limit]
        return self._search_brute_force(query_embedding, limit, agent_id)

    def _search_brute_force(
        self, query_embedding: list[float], limit: int, agent_id: str | None = None
    ) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        rows = self.conn.execute(
            """
            SELECT content_id, embedding, dimensions
            FROM vec_index
            WHERE dimensions = ? AND model = ?
            """,
            (len(query_embedding), self.model_key),
        ).fetchall()
        for content_id, embedding_bytes, dimensions in rows:
            embedding = self._unpack(embedding_bytes, int(dimensions))
            score = self._cosine(query_embedding, embedding)
            if score > 0:
                results.append((str(content_id), score))
        results.sort(key=lambda item: item[1], reverse=True)
        if agent_id:
            results = self._filter_by_agent(results, agent_id)
        return results[:limit]

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM vec_index").fetchone()
        return int(row[0]) if row else 0

    def backend(self) -> str:
        if not self.enabled:
            return "disabled"
        return f"local-vector:{self.embedding_backend}"

    def _load_vector_index(self) -> None:
        if not self.enabled:
            return
        rows = self.conn.execute(
            """
            SELECT content_id, numeric_id, embedding, dimensions, model
            FROM vec_index
            """
        ).fetchall()
        skipped = 0
        for content_id, numeric_id, embedding_bytes, dimensions, model in rows:
            if int(dimensions) != self.dimensions or str(model) != self.model_key:
                skipped += 1
                continue
            embedding = self._unpack(embedding_bytes, int(dimensions))
            self.index.add(int(numeric_id), embedding)
        if skipped > 0:
            import sys

            print(
                f"Warning: Skipped {skipped} embeddings with mismatched dimensions (expected {self.dimensions})",
                file=sys.stderr,
            )

    def _content_ids_for_numeric(self, numeric_ids: list[int]) -> dict[int, str]:
        if not numeric_ids:
            return {}
        placeholders = ",".join("?" for _ in numeric_ids)
        rows = self.conn.execute(
            f"SELECT numeric_id, content_id FROM vec_index WHERE numeric_id IN ({placeholders})",
            numeric_ids,
        ).fetchall()
        return {int(row[0]): str(row[1]) for row in rows}

    def _existing_embeddings(
        self, items: list[tuple[str, str]]
    ) -> dict[str, tuple[int, bytes, int, str]]:
        if not items:
            return {}
        cache_keys_by_content_id = {
            content_id: self._embedding_cache_key(content_id, content)
            for content_id, content in items
        }
        cache_placeholders = ",".join("?" for _ in cache_keys_by_content_id)
        rows = self.conn.execute(
            f"""
            SELECT cache_key, numeric_id, embedding, dimensions, model
            FROM vec_index
            WHERE cache_key IN ({cache_placeholders}) AND model = ?
            """,
            [*cache_keys_by_content_id.values(), self.model_key],
        ).fetchall()
        rows_by_cache_key = {
            str(row[0]): (int(row[1]), bytes(row[2]), int(row[3]), str(row[4]))
            for row in rows
        }
        existing: dict[str, tuple[int, bytes, int, str]] = {}
        for content_id, cache_key in cache_keys_by_content_id.items():
            row = rows_by_cache_key.get(cache_key)
            if row is not None:
                existing[content_id] = row

        missing_content_ids = [
            content_id for content_id, _content in items if content_id not in existing
        ]
        if not missing_content_ids:
            return existing

        content_placeholders = ",".join("?" for _ in missing_content_ids)
        legacy_rows = self.conn.execute(
            f"""
            SELECT content_id, numeric_id, embedding, dimensions, model
            FROM vec_index
            WHERE content_id IN ({content_placeholders}) AND model = ?
            """,
            [*missing_content_ids, self.model_key],
        ).fetchall()
        for row in legacy_rows:
            existing[str(row[0])] = (int(row[1]), bytes(row[2]), int(row[3]), str(row[4]))
        return existing

    @staticmethod
    def _cache_key(content_hash: str, model_key: str) -> str:
        return f"{model_key}\x1f{content_hash}"

    def _embedding_cache_key(self, content_id: str, content: str) -> str:
        return self._cache_key(_content_hash(content), self.model_key)

    @staticmethod
    def _numeric_id(content_id: str) -> int:
        digest = hashlib.blake2b(content_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)

    def _encode(self, text: str) -> list[float]:
        return self.model.encode(text)

    def _encode_many(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode_many(texts)
        for embedding in embeddings:
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, got {len(embedding)}"
                )
        return embeddings

    @staticmethod
    def _pack(embedding: list[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _unpack(embedding_bytes: bytes, dimensions: int) -> list[float]:
        return list(struct.unpack(f"{dimensions}f", embedding_bytes))

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        denominator = left_norm * right_norm
        if denominator <= 0:
            return 0.0
        return sum(lvalue * rvalue for lvalue, rvalue in zip(left, right)) / denominator

    def _filter_by_agent(
        self, results: list[tuple[str, float]], agent_id: str
    ) -> list[tuple[str, float]]:
        if not results:
            return []
        content_ids = [content_id for content_id, _ in results]
        placeholders = ",".join("?" for _ in content_ids)
        rows = self.conn.execute(
            f"SELECT l.content_id FROM long_term_items l WHERE l.agent_id = ? AND l.content_id IN ({placeholders})",
            (agent_id, *content_ids),
        ).fetchall()
        allowed = {str(row[0]) for row in rows}
        return [(content_id, score) for content_id, score in results if content_id in allowed]
