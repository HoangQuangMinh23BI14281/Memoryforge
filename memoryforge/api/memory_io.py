"""Conversation, prompt, file, and chunk ingestion facade."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoryforge.lcm import CompactionRunResult
    from memoryforge.lcm.conversation import ConversationStore
    from memoryforge.memory.longterm.store import LongTermMemoryIndex
    from memoryforge.rlm.chunking import ChunkingStrategy
    from memoryforge.rlm.engine import RLMEngine


class MemoryIOMixin:
    if TYPE_CHECKING:

        @property
        def conversations(self) -> ConversationStore: ...

        @property
        def long_term(self) -> LongTermMemoryIndex: ...

        @property
        def chunking(self) -> ChunkingStrategy: ...

        @property
        def rlm(self) -> RLMEngine: ...

        def rlm_load(
            self,
            agent_id: str,
            value: str | Path,
            name: str | None = None,
            source_path: str | None = None,
            chunk_size: int = 12_000,
            overlap: int = 1_000,
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]: ...

        def lcm_compact_if_needed(
            self,
            agent_id: str,
            session_id: str,
            **kwargs: Any,
        ) -> CompactionRunResult: ...

        def _index_rlm_load_result(
            self,
            *,
            agent_id: str,
            result: dict[str, Any],
            chunk_size: int,
            overlap: int,
            rlm_deduped: bool,
        ) -> None: ...

    def store_conversation(
        self,
        agent_id: str,
        turns: list[dict[str, Any]],
        session_id: str | None = None,
        event_date: float | None = None,
    ) -> list[str]:
        """Store conversation turns into raw store, LCM, and LTM."""
        resolved_session_id = session_id or self._new_session_id()

        try:
            turn_ids = self.conversations.store_session(
                agent_id, turns, resolved_session_id, event_date
            )
            turn_metadata_by_id = _turn_metadata_by_message_id(turns, turn_ids)
            long_term_item_ids = self.long_term.index_messages(
                agent_id,
                turn_ids,
                metadata_by_message_id=turn_metadata_by_id,
            )
            self._merge_turn_metadata(agent_id, turns, long_term_item_ids)
            return turn_ids

        except Exception as e:
            raise RuntimeError(
                f"Failed to store conversation for session {resolved_session_id}: {e}"
            ) from e

    def store_session(
        self,
        agent_id: str,
        session_id: str,
        turns: list[dict[str, Any]],
        event_date: float | None = None,
    ) -> list[str]:
        return self.store_conversation(agent_id, turns, session_id, event_date)

    def search(self, agent_id: str, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.conversations.search(agent_id, query, top_k)

    def search_ensemble(self, agent_id: str, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.recall_long_term(agent_id, query, top_k, include_content=True)

    def ingest_prompt(
        self,
        agent_id: str,
        prompt: str,
        session_id: str | None = None,
        project_root: str | None = None,
    ) -> dict[str, Any]:
        resolved_session_id = session_id or self._new_session_id()
        turns = [{"role": "user", "content": prompt}]
        file_refs = self._resolve_prompt_files(prompt, project_root)
        turn_ids = self.store_conversation(agent_id, turns, session_id=resolved_session_id)
        file_ingests = [
            self.ingest_file(
                agent_id=agent_id,
                path=Path(str(file_ref["path"])),
                name=str(file_ref["relative_path"]),
            )
            for file_ref in file_refs
        ]
        file_item_ids = [
            item_id
            for file_ingest in file_ingests
            for item_id in file_ingest.get("long_term_item_ids", [])
        ]
        compaction = self.lcm_compact_if_needed(
            agent_id,
            resolved_session_id,
            defer_soft=True,
        )
        return {
            "turn_ids": turn_ids,
            "session_id": resolved_session_id,
            "files_ingested": len(file_ingests),
            "file_ingests": file_ingests,
            "long_term_file_item_ids": file_item_ids,
            "lcm_compaction": {
                "triggered": compaction.triggered,
                "rounds": compaction.rounds,
                "before_tokens": compaction.before_tokens,
                "after_tokens": compaction.after_tokens,
                "delta_tokens": compaction.delta_tokens,
                "expanded": compaction.expanded,
                "effective": compaction.effective,
                "deferred": compaction.deferred,
                "reason": compaction.reason,
                "summary_node_ids": compaction.summary_node_ids,
            },
        }

    def ingest_file(
        self,
        agent_id: str,
        path: str | Path,
        *,
        name: str | None = None,
        chunk_size: int = 12_000,
        overlap: int = 1_000,
    ) -> dict[str, Any]:
        """Ingest a file through RLM chunks and LTM without appending it to LCM."""

        file_path = Path(path).expanduser().resolve()
        result = self.rlm.load(
            agent_id=agent_id,
            value=file_path,
            name=name or file_path.name,
            source_path=str(file_path),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        result["rlm_worker"] = {
            "enabled": False,
            "skipped": "ingest_file_indexing_only",
        }
        self._index_rlm_load_result(
            agent_id=agent_id,
            result=result,
            chunk_size=chunk_size,
            overlap=overlap,
            rlm_deduped=bool(result.get("deduped")),
        )
        result["ingest_path"] = str(file_path)
        return result

    def recall_long_term(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        *,
        include_content: bool = False,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            result.to_dict()
            for result in self.long_term.recall(
                agent_id=agent_id,
                query=query,
                top_k=top_k,
                include_content=include_content,
                session_id=session_id,
            )
        ]

    def long_term_source(self, agent_id: str, item_id: str) -> dict[str, Any] | None:
        return self.long_term.get_source(agent_id, item_id)

    def chunk_content(self, value: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "content": chunk.content,
                "content_type": chunk.content_type.value,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunking.chunk(value)
        ]

    @staticmethod
    def _resolve_prompt_files(
        prompt: str,
        project_root: str | None,
        max_file_bytes: int = 1_000_000,
    ) -> list[dict[str, str | int]]:
        if not project_root:
            return []
        root = Path(project_root).expanduser().resolve()
        file_refs: list[dict[str, str | int]] = []
        seen_paths: set[Path] = set()
        for token in prompt.replace("\n", " ").split():
            clean = token.strip("`'\".,;:()[]{}")
            if Path(clean).suffix.lower() not in {
                ".md",
                ".txt",
                ".py",
                ".toml",
                ".json",
                ".yaml",
                ".yml",
            }:
                continue
            candidate = (
                (root / clean).resolve() if not Path(clean).is_absolute() else Path(clean).resolve()
            )
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if not candidate.is_file() or candidate.stat().st_size > max_file_bytes:
                continue
            if candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            file_refs.append(
                {
                    "path": str(candidate),
                    "relative_path": candidate.relative_to(root).as_posix(),
                    "size_bytes": candidate.stat().st_size,
                }
            )
        return file_refs

    @staticmethod
    def _new_session_id() -> str:
        return f"ses_{int(time.time() * 1000):016x}_{uuid.uuid4().hex[:8]}"

    def _merge_turn_metadata(
        self,
        agent_id: str,
        turns: list[dict[str, Any]],
        item_ids: list[str],
    ) -> None:
        updates: list[tuple[str, dict[str, Any]]] = []
        for item_id, turn in zip(item_ids, turns, strict=False):
            metadata = turn.get("metadata")
            if isinstance(metadata, dict):
                updates.append((item_id, dict(metadata)))
        self.long_term.merge_items_metadata(agent_id, updates)


def _turn_metadata_by_message_id(
    turns: list[dict[str, Any]],
    message_ids: list[str],
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for message_id, turn in zip(message_ids, turns, strict=False):
        metadata = turn.get("metadata")
        if isinstance(metadata, dict):
            mapped[str(message_id)] = dict(metadata)
    return mapped

