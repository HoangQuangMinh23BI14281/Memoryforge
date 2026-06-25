from memoryforge import MemoryForge
from memoryforge.rlm.chunking import ChunkingStrategy, ContentType


def test_chunking_strategy_routes_conversation():
    chunks = ChunkingStrategy().chunk([{"role": "user", "content": "hello"}])

    assert len(chunks) == 1
    assert chunks[0].content_type == ContentType.CONVERSATION
    assert chunks[0].metadata["role"] == "user"


def test_chunking_strategy_routes_document_file(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\nAlice uses SQLite.\n", encoding="utf-8")

    chunks = ChunkingStrategy().chunk(path)

    assert chunks[0].content.startswith("# Notes")
    assert all(chunk.content_type == ContentType.DOCS for chunk in chunks)


def test_memoryforge_exposes_chunking_api(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\nAlice uses SQLite.\n", encoding="utf-8")

    chunks = MemoryForge(":memory:").chunk_content(path)

    assert chunks[0]["content"].startswith("# Notes")
    assert all(chunk["content_type"] == "docs" for chunk in chunks)
