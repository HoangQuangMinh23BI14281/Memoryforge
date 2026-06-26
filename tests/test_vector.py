import sqlite3
import sys
import types

from memoryforge._core import ContentHashTable, HnswIndex
from memoryforge.search import VectorIndex


def _install_fake_fastembed(monkeypatch, dimensions=3):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for text in texts:
                vector = [0.0] * dimensions
                lowered = text.lower()
                if "auth" in lowered or "token" in lowered:
                    vector[0] = 1.0
                if dimensions > 1 and ("sqlite" in lowered or "database" in lowered):
                    vector[1] = 1.0
                if dimensions > 2 and ("layout" in lowered or "button" in lowered):
                    vector[2] = 1.0
                if not any(vector):
                    vector[0] = 1.0
                yield vector

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)


def test_vector_index_without_model_disables_vector_search(tmp_path):
    index = VectorIndex(str(tmp_path / "memory.db"))
    index.add("doc-auth", "jwt authentication token secret")
    index.add("doc-db", "sqlite local database storage")

    results = index.search("authentication token", limit=2)
    assert results == []
    assert index.count() == 0
    assert index.model_name == "disabled"
    assert index.embedding_backend == "disabled"
    assert index.backend() == "disabled"


def test_vector_index_compatibility_api():
    index = HnswIndex(3)
    index.add(1, [1.0, 0.0, 0.0])
    index.add(2, [0.0, 1.0, 0.0])

    assert index.search([1.0, 0.0, 0.0], 2)[0][0] == 1


def test_vector_index_fastembed_backend_uses_model_dimension(tmp_path, monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for text in texts:
                vector = [0.0] * 384
                lowered = text.lower()
                if "auth" in lowered or "token" in lowered:
                    vector[0] = 1.0
                if "sqlite" in lowered or "database" in lowered:
                    vector[1] = 1.0
                yield vector

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    index = VectorIndex(
        str(tmp_path / "memory.db"),
        model_name="BAAI/bge-small-en-v1.5",
        backend="fastembed",
    )
    index.add("doc-auth", "jwt authentication token secret")
    index.add("doc-db", "sqlite local database storage")

    results = index.search("authentication token", limit=2)

    assert index.dimensions == 384
    assert index.model_name == "BAAI/bge-small-en-v1.5"
    assert index.embedding_backend == "fastembed"
    assert index.backend() == "local-vector:fastembed"
    assert results[0][0] == "doc-auth"


def test_vector_index_add_many_batches_and_skips_cached_embeddings(tmp_path, monkeypatch):
    seen = {"calls": []}

    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            seen["calls"].append(list(texts))
            for text in texts:
                vector = [0.0] * 384
                vector[0] = float(len(text) or 1)
                yield vector

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    index = VectorIndex(
        str(tmp_path / "memory.db"),
        model_name="BAAI/bge-small-en-v1.5",
        backend="fastembed",
    )

    encoded = index.add_many(
        [
            ("doc-auth", "jwt authentication token secret"),
            ("doc-db", "sqlite local database storage"),
        ]
    )
    call_count_after_first_batch = len(seen["calls"])
    cached = index.add_many(
        [
            ("doc-auth", "jwt authentication token secret"),
            ("doc-db", "sqlite local database storage"),
        ]
    )

    assert encoded == 2
    assert cached == 0
    assert index.last_add_stats == {"requested": 2, "encoded": 0, "cached": 2}
    assert seen["calls"][1] == [
        "jwt authentication token secret",
        "sqlite local database storage",
    ]
    assert len(seen["calls"]) == call_count_after_first_batch


def test_vector_index_cache_key_uses_content_hash_when_available(tmp_path, monkeypatch):
    _install_fake_fastembed(monkeypatch)
    db_path = str(tmp_path / "memory.db")
    content_store = ContentHashTable(db_path)
    content_id, _is_new = content_store.store("same durable text")
    content_hash = content_store.conn.execute(
        "SELECT content_hash FROM content_store WHERE content_id = ?",
        (content_id,),
    ).fetchone()[0]

    index = VectorIndex(db_path, model_name="model-a", backend="fastembed")
    index.add(content_id, "same durable text")
    row = index.conn.execute("SELECT cache_key, content_id, model FROM vec_index").fetchone()

    assert row == (f"{index.model_key}\x1f{content_hash}", content_id, index.model_key)


def test_vector_index_reuses_same_content_hash_for_different_content_ids(tmp_path, monkeypatch):
    _install_fake_fastembed(monkeypatch)
    db_path = str(tmp_path / "memory.db")
    index = VectorIndex(db_path, model_name="model-a", backend="fastembed")

    encoded = index.add_many([("first-content-id", "same durable text")])
    cached = index.add_many([("second-content-id", "same durable text")])

    assert encoded == 1
    assert cached == 0
    assert index.last_add_stats == {"requested": 1, "encoded": 0, "cached": 1}
    assert index.count() == 1


def test_vector_index_does_not_load_embeddings_from_other_model_identity(
    tmp_path, monkeypatch
):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for text in texts:
                vector = [0.0] * 384
                if self.model_name == "model-a":
                    vector[0] = 1.0
                elif "query" in text:
                    vector[0] = 1.0
                yield vector

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    db_path = str(tmp_path / "memory.db")
    first = VectorIndex(db_path, model_name="model-a", backend="fastembed")
    first.add("doc-auth", "jwt authentication token secret")
    second = VectorIndex(db_path, model_name="model-b", backend="fastembed")

    assert first.search("query", limit=1)
    assert second.search("query", limit=1) == []


def test_vector_index_keeps_same_content_for_multiple_model_identities(tmp_path, monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for _text in texts:
                vector = [0.0] * 384
                vector[0] = 1.0
                yield vector

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    db_path = str(tmp_path / "memory.db")
    first = VectorIndex(db_path, model_name="model-a", backend="fastembed")
    first.add("shared-content", "same durable text")
    second = VectorIndex(db_path, model_name="model-b", backend="fastembed")
    second.add("shared-content", "same durable text")

    first_reopened = VectorIndex(db_path, model_name="model-a", backend="fastembed")
    second_reopened = VectorIndex(db_path, model_name="model-b", backend="fastembed")
    rows = first_reopened.conn.execute(
        "SELECT content_id, model FROM vec_index ORDER BY model"
    ).fetchall()

    assert rows == [
        ("shared-content", "fastembed:model-a"),
        ("shared-content", "fastembed:model-b"),
    ]
    assert first_reopened.search("query", limit=1)[0][0] == "shared-content"
    assert second_reopened.search("query", limit=1)[0][0] == "shared-content"


def test_vector_index_migrates_legacy_content_id_primary_key(tmp_path):
    db_path = str(tmp_path / "memory.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE vec_index (
            content_id TEXT PRIMARY KEY,
            numeric_id INTEGER NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            dimensions INTEGER NOT NULL,
            model TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO vec_index
        (content_id, numeric_id, embedding, dimensions, model, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-content",
            1,
            VectorIndex._pack([1.0, 0.0, 0.0]),
            3,
            "hashing:hashing-3",
            1.0,
        ),
    )
    conn.commit()
    conn.close()

    index = VectorIndex(db_path, dimensions=3)
    columns = {row[1] for row in index.conn.execute("PRAGMA table_info(vec_index)")}
    rows = index.conn.execute(
        "SELECT cache_key, content_id, model FROM vec_index"
    ).fetchall()

    assert "cache_key" in columns
    assert rows == [("hashing:hashing-3\x1flegacy-content", "legacy-content", "hashing:hashing-3")]
    assert index.count() == 1


def test_vector_index_fastembed_all_minilm_alias(tmp_path, monkeypatch):
    seen = {}

    class FakeTextEmbedding:
        def __init__(self, model_name):
            seen["model_name"] = model_name

        def embed(self, texts):
            for _text in texts:
                yield [0.0] * 384

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)

    VectorIndex(str(tmp_path / "memory.db"), model_name="all-MiniLM-L6-v2", backend="fastembed")

    assert seen["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
