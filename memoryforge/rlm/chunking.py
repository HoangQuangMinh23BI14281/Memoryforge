"""RLM chunking primitives and engine chunk storage mixin."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryforge._core import ContentHashTable
from memoryforge.lcm.conversation import ConversationChunker
from memoryforge.rlm.common import hash_text, new_id, preview
from memoryforge.rlm.types import ChunkDraft

if TYPE_CHECKING:
    from memoryforge.search.vector import VectorIndex


class ContentType(Enum):
    CONVERSATION = "conversation"
    DOCS = "docs"
    CONFIG = "config"


@dataclass
class Chunk:
    content: str
    content_type: ContentType
    metadata: dict[str, Any]


class ChunkingStrategy:
    def __init__(
        self,
        conversation_chunker: ConversationChunker | None = None,
    ):
        self.conversation_chunker = conversation_chunker or ConversationChunker()

    def detect_content_type(self, value: str | Path | list[dict[str, Any]]) -> ContentType:
        if isinstance(value, list):
            return ContentType.CONVERSATION
        path = Path(value)
        if path.suffix in {".md", ".rst", ".txt"}:
            return ContentType.DOCS
        if path.suffix in {".toml", ".json", ".yaml", ".yml", ".ini"}:
            return ContentType.CONFIG
        return ContentType.DOCS

    def chunk(self, value: str | Path | list[dict[str, Any]]) -> list[Chunk]:
        if isinstance(value, list):
            return [
                Chunk(
                    content=turn.content,
                    content_type=ContentType.CONVERSATION,
                    metadata={"role": turn.role, "turn_index": turn.turn_index},
                )
                for turn in self.conversation_chunker.chunk_conversation(value)
            ]
        content_type = self.detect_content_type(value)
        path = Path(value)
        content = path.read_text(encoding="utf-8") if path.exists() else str(value)
        return [Chunk(content=content, content_type=content_type, metadata={"source": str(value)})]


class RLMContentChunker:
    """Lossless source reader and semantic text slicer for RLM buffers."""

    def read_source(self, value: str | Path, source_path: str | None) -> tuple[str, str | None]:
        if source_path:
            path = Path(source_path).expanduser().resolve()
            return path.read_text(encoding="utf-8", errors="replace"), str(path)
        if isinstance(value, Path):
            path = value.expanduser().resolve()
            return path.read_text(encoding="utf-8", errors="replace"), str(path)
        try:
            candidate = Path(value).expanduser()
            if candidate.exists() and candidate.is_file():
                path = candidate.resolve()
                return path.read_text(encoding="utf-8", errors="replace"), str(path)
        except (OSError, ValueError):
            pass
        return str(value), None

    def detect_content_type(self, source_path: str | None, source: str) -> str:
        if source_path:
            suffix = Path(source_path).suffix.lower()
            if suffix in {".toml", ".json", ".yaml", ".yml", ".ini"}:
                return ContentType.CONFIG.value
            if suffix in {".md", ".rst", ".txt"}:
                return ContentType.DOCS.value
        return ContentType.DOCS.value

    def chunk_text(
        self,
        source: str,
        content_type: str,
        chunk_size: int,
        overlap: int,
        *,
        base_char: int = 0,
        base_byte: int = 0,
        base_line: int = 1,
        strategy: str = "semantic",
    ) -> list[ChunkDraft]:
        if not source:
            return []
        chunk_size = max(1_000, chunk_size)
        overlap = max(0, min(overlap, chunk_size // 2))
        drafts: list[ChunkDraft] = []
        start = 0

        min_progress = max(100, chunk_size // 10)

        char_to_byte = [0]
        byte_pos = 0
        for char in source:
            byte_pos += len(char.encode("utf-8"))
            char_to_byte.append(byte_pos)

        while start < len(source):
            target_end = min(len(source), start + chunk_size)
            end = (
                self._semantic_boundary(source, start, target_end)
                if target_end < len(source)
                else target_end
            )

            if end <= start:
                end = min(start + min_progress, len(source))

            byte_start = base_byte + char_to_byte[start]
            byte_end = base_byte + char_to_byte[end]

            text = source[start:end]
            drafts.append(
                ChunkDraft(
                    content=text,
                    content_type=content_type,
                    strategy=strategy,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    char_start=base_char + start,
                    char_end=base_char + end,
                    start_line=base_line + source[:start].count("\n") + 1,
                    end_line=base_line + source[: end - 1].count("\n") + 1
                    if end > 0
                    else base_line,
                    has_overlap=bool(drafts),
                    metadata={"chunker": strategy},
                )
            )

            if end >= len(source):
                break

            next_start = end - overlap
            if next_start <= start:
                next_start = start + min_progress

            start = next_start

        return drafts

    @staticmethod
    def _semantic_boundary(source: str, start: int, target_end: int) -> int:
        if target_end - start < 400:
            return target_end

        lower_bound = start + max(200, (target_end - start) // 2)
        candidates = [
            source.rfind("\n\n", lower_bound, target_end),
            source.rfind("\n", lower_bound, target_end),
            source.rfind(". ", lower_bound, target_end),
        ]
        boundary = max(candidates)
        if boundary <= start:
            return target_end
        if source.startswith("\n\n", boundary):
            return boundary + 2
        elif source.startswith("\n", boundary):
            return boundary + 1
        elif source.startswith(". ", boundary):
            return boundary + 2
        return boundary + 1


class RLMChunkMixin:
    """Load, retrieve, and dispatch RLM chunks backed by the shared SQLite store."""

    if TYPE_CHECKING:
        conn: sqlite3.Connection
        chunker: RLMContentChunker

        def _vector_index(self) -> VectorIndex | None: ...

        def _content_store(self) -> ContentHashTable: ...

        def _retrieve_content(self, content_id: str) -> str | None: ...

        def search(
            self,
            agent_id: str,
            query: str,
            *,
            buffer_id: str | None = None,
            limit: int = 10,
            mode: str = "hybrid",
        ) -> list[dict[str, Any]]: ...

    def load(
        self,
        agent_id: str,
        value: str | Path,
        *,
        name: str | None = None,
        source_path: str | None = None,
        content_type: str | None = None,
        chunk_size: int = 12_000,
        overlap: int = 1_000,
    ) -> dict[str, Any]:
        """Load prompt/file content into RLM buffers and lossless chunks."""

        source, resolved_path = self.chunker.read_source(value, source_path)
        resolved_type = content_type or self.chunker.detect_content_type(resolved_path, source)
        strategy = "semantic"
        source_hash = hash_text(source)
        existing = self._find_existing_buffer(
            agent_id,
            content_hash=source_hash,
            content_type=resolved_type,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if existing is not None:
            existing["deduped"] = True
            return existing

        chunks = self.chunker.chunk_text(source, resolved_type, chunk_size, overlap)
        content_id, _is_new = self._content_store().store(source)
        buffer_id = new_id("rbuf")
        timestamp = time.time()
        resolved_name = name or (Path(resolved_path).name if resolved_path else buffer_id)

        self.conn.execute(
            """
            INSERT INTO rlm_buffers
            (buffer_id, agent_id, name, source_path, content_id, content_hash,
             content_type, strategy, size, line_count, chunk_count, metadata,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                buffer_id,
                agent_id,
                resolved_name,
                resolved_path,
                content_id,
                source_hash,
                resolved_type,
                strategy,
                len(source.encode("utf-8")),
                source.count("\n") + 1,
                len(chunks),
                json.dumps(
                    {
                        "lossless": True,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                    },
                    sort_keys=True,
                ),
                timestamp,
                timestamp,
            ),
        )
        self.conn.commit()

        try:
            self._insert_chunks(agent_id, buffer_id, chunks)
        except Exception:
            with self.conn:
                self.conn.execute(
                    "DELETE FROM search_fts WHERE scope = ? AND buffer_id = ?",
                    ("rlm_chunk", buffer_id),
                )
                self.conn.execute("DELETE FROM rlm_buffers WHERE buffer_id = ?", (buffer_id,))
            raise

        return {
            "buffer_id": buffer_id,
            "agent_id": agent_id,
            "name": resolved_name,
            "source_path": resolved_path,
            "content_id": content_id,
            "content_hash": source_hash,
            "content_type": resolved_type,
            "strategy": strategy,
            "chunk_count": len(chunks),
            "lossless": True,
            "deduped": False,
        }

    def get_chunk(self, chunk_id: str, *, include_content: bool = True) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT c.chunk_id, c.agent_id, c.buffer_id, b.name, b.source_path,
                   c.content_id, c.chunk_index, c.content_type, c.strategy,
                   c.byte_start, c.byte_end, c.char_start, c.char_end,
                   c.start_line, c.end_line, c.token_count, c.has_overlap,
                   c.metadata
            FROM rlm_chunks c
            JOIN rlm_buffers b ON b.buffer_id = c.buffer_id
            WHERE c.chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if not row:
            return None
        content = self._retrieve_content(row[5]) or ""
        payload = {
            "chunk_id": row[0],
            "ref": f"rlm_chunk:{row[0]}",
            "agent_id": row[1],
            "buffer_id": row[2],
            "buffer_name": row[3],
            "source_path": row[4],
            "content_id": row[5],
            "chunk_index": row[6],
            "content_type": row[7],
            "strategy": row[8],
            "byte_range": {"start": row[9], "end": row[10]},
            "char_range": {"start": row[11], "end": row[12]},
            "start_line": row[13],
            "end_line": row[14],
            "token_count": row[15],
            "has_overlap": bool(row[16]),
            "metadata": json.loads(row[17] or "{}"),
            "preview": preview(content),
            "lossless": True,
        }
        if include_content:
            payload["content"] = content
        return payload

    def dispatch(
        self,
        agent_id: str,
        *,
        buffer_id: str | None = None,
        query: str | None = None,
        limit: int = 20,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Return chunk-ID batches for external sub-agents; never calls a model."""

        retrieval_mode = "search" if query else "full_scan"
        chunks = (
            self.search(agent_id, query, buffer_id=buffer_id, limit=limit)
            if query
            else self.list_chunks(agent_id, buffer_id=buffer_id, limit=limit)
        )
        run_id = new_id("rrun")

        batches = []
        effective_batch_size = max(1, len(chunks)) if batch_size is None else max(1, batch_size)
        for batch_index, offset in enumerate(range(0, len(chunks), effective_batch_size)):
            items = chunks[offset : offset + effective_batch_size]
            batches.append(
                {
                    "batch_index": batch_index,
                    "refs": [item["ref"] for item in items],
                    "chunk_ids": [item["chunk_id"] for item in items],
                    "items": items,
                    "instruction": self._dispatch_instruction(retrieval_mode),
                }
            )

        return {
            "run_id": run_id,
            "agent_id": agent_id,
            "buffer_id": buffer_id,
            "query": query,
            "retrieval_mode": retrieval_mode,
            "chunk_count": len(chunks),
            "batch_count": len(batches),
            "batch_size": effective_batch_size,
            "lossless": True,
            "batches": batches,
        }

    def list_chunks(
        self,
        agent_id: str,
        *,
        buffer_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["c.agent_id = ?"]
        params: list[Any] = [agent_id]
        if buffer_id:
            clauses.append("c.buffer_id = ?")
            params.append(buffer_id)
        rows = self.conn.execute(
            f"""
            SELECT c.chunk_id
            FROM rlm_chunks c
            WHERE {" AND ".join(clauses)}
            ORDER BY c.buffer_id, c.chunk_index
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
        return [
            chunk for row in rows if (chunk := self.get_chunk(str(row[0]), include_content=False))
        ]

    def _insert_chunk(
        self,
        agent_id: str,
        buffer_id: str,
        index: int,
        draft: ChunkDraft,
    ) -> None:
        self._insert_chunks(agent_id, buffer_id, [draft], start_index=index)

    def _insert_chunks(
        self,
        agent_id: str,
        buffer_id: str,
        drafts: list[ChunkDraft],
        *,
        start_index: int = 0,
    ) -> None:
        prepared: list[dict[str, Any]] = []
        content_store = self._content_store()
        for offset, draft in enumerate(drafts):
            chunk_id = new_id("rchunk")
            content_id, _is_new = content_store.store(draft.content)
            prepared.append(
                {
                    "chunk_id": chunk_id,
                    "content_id": content_id,
                    "chunk_index": start_index + offset,
                    "draft": draft,
                    "created_at": time.time(),
                }
            )

        vector = self._vector_index()
        if vector is not None:
            try:
                vector.add_many(
                    [
                        (str(item["content_id"]), str(item["draft"].content))
                        for item in prepared
                    ]
                )
            except sqlite3.DatabaseError as exc:
                import sys

                print(
                    f"Warning: Vector indexing failed for buffer {buffer_id}: {exc}",
                    file=sys.stderr,
                )

        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO rlm_chunks
                (chunk_id, agent_id, buffer_id, content_id, chunk_index, content_type,
                 strategy, byte_start, byte_end, char_start, char_end, start_line,
                 end_line, token_count, has_overlap, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["chunk_id"],
                        agent_id,
                        buffer_id,
                        item["content_id"],
                        item["chunk_index"],
                        item["draft"].content_type,
                        item["draft"].strategy,
                        item["draft"].byte_start,
                        item["draft"].byte_end,
                        item["draft"].char_start,
                        item["draft"].char_end,
                        item["draft"].start_line,
                        item["draft"].end_line,
                        max(1, len(item["draft"].content) // 4),
                        1 if item["draft"].has_overlap else 0,
                        json.dumps(item["draft"].metadata or {}),
                        item["created_at"],
                    )
                    for item in prepared
                ],
            )
            self.conn.executemany(
                """
                INSERT INTO search_fts
                (content, scope, agent_id, source_type, source_id, content_id, buffer_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["draft"].content,
                        "rlm_chunk",
                        agent_id,
                        "rlm_chunk",
                        item["chunk_id"],
                        item["content_id"],
                        buffer_id,
                    )
                    for item in prepared
                ],
            )

    def _find_existing_buffer(
        self,
        agent_id: str,
        *,
        content_hash: str,
        content_type: str,
        strategy: str,
        chunk_size: int,
        overlap: int,
    ) -> dict[str, Any] | None:
        rows = self.conn.execute(
            """
            SELECT buffer_id, agent_id, name, source_path, content_id, content_hash,
                   content_type, strategy, chunk_count, metadata
            FROM rlm_buffers
            WHERE agent_id = ? AND content_hash = ? AND content_type = ? AND strategy = ?
            ORDER BY created_at DESC
            """,
            (agent_id, content_hash, content_type, strategy),
        ).fetchall()
        for row in rows:
            metadata = _metadata_json(row[9])
            if int(metadata.get("chunk_size", -1)) != chunk_size:
                continue
            if int(metadata.get("overlap", -1)) != overlap:
                continue
            return {
                "buffer_id": str(row[0]),
                "agent_id": str(row[1]),
                "name": str(row[2]),
                "source_path": row[3],
                "content_id": str(row[4]),
                "content_hash": str(row[5]),
                "content_type": str(row[6]),
                "strategy": str(row[7]),
                "chunk_count": int(row[8]),
                "lossless": True,
            }
        return None

    @staticmethod
    def _dispatch_instruction(retrieval_mode: str) -> str:
        mode_note = (
            "These refs were selected by query search."
            if retrieval_mode == "search"
            else "These refs are a sequential full-scan batch, not search-filtered."
        )
        return (
            f"Sub-agent: {mode_note} Fetch each chunk with "
            "`memoryforge rlm-chunk-get <id>`, analyze only cited chunks, "
            "then return findings with chunk refs."
        )


def _metadata_json(value: Any) -> dict[str, Any]:
    try:
        metadata = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return metadata if isinstance(metadata, dict) else {}
