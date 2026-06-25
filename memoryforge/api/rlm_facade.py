"""Public RLM facade methods for MemoryForge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from memoryforge.memory.longterm.store import LongTermMemoryIndex
from memoryforge.rlm.engine import RLMEngine


class RLMFacadeMixin:
    if TYPE_CHECKING:
        db_path: str
        _rlm: RLMEngine | None

        @property
        def rlm(self) -> RLMEngine: ...

        @property
        def long_term(self) -> LongTermMemoryIndex: ...

    def _rlm_existing(self) -> tuple[RLMEngine, bool]:
        if self._rlm is not None:
            return self._rlm, False

        return RLMEngine(self.db_path, ensure_schema=False), True

    def rlm_load(
        self,
        agent_id: str,
        value: str | Path,
        name: str | None = None,
        source_path: str | None = None,
        chunk_size: int = 12_000,
        overlap: int = 1_000,
        runner: str | None = "auto",
        model: str | None = None,
        base_url: str | None = None,
        project_root: str | None = None,
        timeout_s: float = 900.0,
        batch_size: int | None = None,
        max_workers: int | None = None,
        max_retries: int = 0,
        allow_partial: bool = False,
        synthesize: bool = True,
        recursive: bool = True,
        max_recursive_rounds: int = 2,
        recursive_token_limit: int | None = None,
    ) -> dict[str, Any]:
        result = self.rlm.load(
            agent_id=agent_id,
            value=value,
            name=name,
            source_path=source_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        rlm_deduped = bool(result.get("deduped"))
        configured_batch_size = _resolve_optional_int(batch_size, "MEMORYFORGE_RLM_BATCH_SIZE")
        resolved_max_workers = _resolve_positive_int(
            max_workers, "MEMORYFORGE_RLM_MAX_WORKERS", default=1
        )
        chunk_count = int(result.get("chunk_count") or 0)
        if rlm_deduped:
            result["rlm_worker"] = {
                "enabled": True,
                "skipped": "deduped_existing_buffer",
                "batch_size": _effective_batch_size(configured_batch_size, chunk_count),
                "max_workers": resolved_max_workers,
            }
        elif chunk_count > 0:
            worker_result = self.rlm.run(
                agent_id=agent_id,
                buffer_id=str(result["buffer_id"]),
                query=None,
                limit=chunk_count,
                batch_size=configured_batch_size,
                runner=runner,
                model=model,
                base_url=base_url,
                project_root=project_root,
                timeout_s=timeout_s,
                max_workers=resolved_max_workers,
                max_retries=max_retries,
                allow_partial=allow_partial,
                synthesize=synthesize,
                recursive=recursive,
                max_recursive_rounds=max_recursive_rounds,
                recursive_token_limit=recursive_token_limit,
            )
            result["rlm_worker"] = _worker_manifest(worker_result)
        else:
            result["rlm_worker"] = {
                "enabled": True,
                "skipped": "empty_buffer",
                "batch_size": _effective_batch_size(configured_batch_size, chunk_count),
                "max_workers": resolved_max_workers,
            }
        self._index_rlm_load_result(
            agent_id=agent_id,
            result=result,
            chunk_size=chunk_size,
            overlap=overlap,
            rlm_deduped=rlm_deduped,
        )
        return cast(dict[str, Any], result)

    def _index_rlm_load_result(
        self,
        *,
        agent_id: str,
        result: dict[str, Any],
        chunk_size: int,
        overlap: int,
        rlm_deduped: bool,
    ) -> None:
        long_term = self.long_term
        existing_item_ids = {
            str(row[0])
            for row in long_term.conn.execute(
                "SELECT item_id FROM long_term_items WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
        }
        indexed_item_ids = long_term.index_rlm_buffer(agent_id, str(result["buffer_id"]))
        result["long_term_item_ids"] = indexed_item_ids
        result["source_type"] = "rlm_chunk"
        ltm_deduped = bool(indexed_item_ids) and (
            any(item_id in existing_item_ids for item_id in indexed_item_ids)
            or len(indexed_item_ids) != int(result.get("chunk_count") or 0)
        )
        vector_stats = dict(getattr(long_term.vector, "last_add_stats", {}) or {})
        if indexed_item_ids and all(item_id in existing_item_ids for item_id in indexed_item_ids):
            vector_stats = {
                "requested": len(indexed_item_ids),
                "encoded": 0,
                "cached": len(indexed_item_ids),
            }
        vector_manifest = {
            "embedding_backend": long_term.vector.embedding_backend,
            "vector_model": long_term.vector.model_name,
            "model_key": long_term.vector.model_key,
            "dimensions": long_term.vector.dimensions,
            "add_stats": vector_stats,
        }
        result["rlm_deduped"] = rlm_deduped
        result["ltm_deduped"] = ltm_deduped
        result["deduped"] = rlm_deduped or ltm_deduped
        result["ingestion_manifest"] = {
            "agent_id": agent_id,
            "buffer_id": result["buffer_id"],
            "content_hash": result["content_hash"],
            "content_id": result["content_id"],
            "content_type": result["content_type"],
            "strategy": result["strategy"],
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_count": result["chunk_count"],
            "long_term_item_count": len(indexed_item_ids),
            "rlm_deduped": rlm_deduped,
            "ltm_deduped": ltm_deduped,
            "vector": vector_manifest,
        }

    def rlm_search(
        self,
        agent_id: str,
        query: str,
        buffer_id: str | None = None,
        limit: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        engine, owned = self._rlm_existing()
        try:
            return cast(
                list[dict[str, Any]],
                engine.search(
                    agent_id=agent_id,
                    query=query,
                    buffer_id=buffer_id,
                    limit=limit,
                    mode=mode,
                ),
            )
        finally:
            if owned:
                engine.close()

    def rlm_chunk_get(self, chunk_id: str) -> dict[str, Any] | None:
        engine, owned = self._rlm_existing()
        try:
            return cast(dict[str, Any] | None, engine.get_chunk(chunk_id, include_content=True))
        finally:
            if owned:
                engine.close()

    def rlm_dispatch(
        self,
        agent_id: str,
        buffer_id: str | None = None,
        query: str | None = None,
        limit: int = 20,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        configured_batch_size = _resolve_optional_int(batch_size, "MEMORYFORGE_RLM_BATCH_SIZE")
        engine, owned = self._rlm_existing()
        try:
            return cast(
                dict[str, Any],
                engine.dispatch(
                    agent_id=agent_id,
                    buffer_id=buffer_id,
                    query=query,
                    limit=limit,
                    batch_size=configured_batch_size,
                ),
            )
        finally:
            if owned:
                engine.close()

    def rlm_record_result(
        self,
        agent_id: str,
        run_id: str,
        chunk_ids: list[str],
        analysis: str,
        batch_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        engine, owned = self._rlm_existing()
        try:
            return cast(
                dict[str, Any],
                engine.record_result(
                    agent_id=agent_id,
                    run_id=run_id,
                    chunk_ids=chunk_ids,
                    analysis=analysis,
                    batch_index=batch_index,
                    metadata=metadata,
                ),
            )
        finally:
            if owned:
                engine.close()

    def rlm_aggregate(
        self,
        agent_id: str,
        run_id: str,
        summary: str | None = None,
    ) -> dict[str, Any]:
        engine, owned = self._rlm_existing()
        try:
            return cast(
                dict[str, Any], engine.aggregate(agent_id=agent_id, run_id=run_id, summary=summary)
            )
        finally:
            if owned:
                engine.close()

    def rlm_run(
        self,
        agent_id: str,
        value: str | Path | None = None,
        name: str | None = None,
        buffer_id: str | None = None,
        query: str | None = None,
        limit: int = 20,
        batch_size: int | None = None,
        chunk_size: int = 12_000,
        overlap: int = 1_000,
        runner: str | None = "auto",
        model: str | None = None,
        base_url: str | None = None,
        project_root: str | None = None,
        timeout_s: float = 900.0,
        max_workers: int = 1,
        max_retries: int = 0,
        allow_partial: bool = False,
        synthesize: bool = True,
        recursive: bool = True,
        max_recursive_rounds: int = 2,
        recursive_token_limit: int | None = None,
    ) -> dict[str, Any]:
        configured_batch_size = _resolve_optional_int(batch_size, "MEMORYFORGE_RLM_BATCH_SIZE")
        result = self.rlm.run(
            agent_id=agent_id,
            value=value,
            name=name,
            buffer_id=buffer_id,
            query=query,
            limit=limit,
            batch_size=configured_batch_size,
            chunk_size=chunk_size,
            overlap=overlap,
            runner=runner,
            model=model,
            base_url=base_url,
            project_root=project_root,
            timeout_s=timeout_s,
            max_workers=max_workers,
            max_retries=max_retries,
            allow_partial=allow_partial,
            synthesize=synthesize,
            recursive=recursive,
            max_recursive_rounds=max_recursive_rounds,
            recursive_token_limit=recursive_token_limit,
        )
        indexed_item_ids: list[str] = []
        loaded_buffers = result.get("loaded_buffers") or []
        if not loaded_buffers and result.get("loaded"):
            loaded_buffers = [result["loaded"]]
        for loaded_buffer in loaded_buffers:
            buffer_id_to_index = (
                loaded_buffer.get("buffer_id") if isinstance(loaded_buffer, dict) else None
            )
            if buffer_id_to_index:
                indexed_item_ids.extend(
                    self.long_term.index_rlm_buffer(agent_id, str(buffer_id_to_index))
                )
        if not indexed_item_ids and buffer_id:
            indexed_item_ids.extend(self.long_term.index_rlm_buffer(agent_id, str(buffer_id)))
        result["long_term_item_ids"] = indexed_item_ids
        return cast(dict[str, Any], result)


def _resolve_positive_int(value: int | None, env_name: str, *, default: int) -> int:
    candidate: object = value if value is not None else os.environ.get(env_name)
    if candidate in {None, ""}:
        return default
    try:
        resolved = int(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {candidate!r}") from exc
    if resolved < 1:
        raise ValueError(f"{env_name} must be at least 1, got {resolved}")
    return resolved


def _resolve_optional_int(value: int | None, env_name: str) -> int | None:
    candidate: object = value if value is not None else os.environ.get(env_name)
    if candidate in {None, ""}:
        return None
    try:
        return int(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {candidate!r}") from exc


def _effective_batch_size(batch_size: int | None, chunk_count: int) -> int:
    if batch_size is None:
        return max(1, chunk_count)
    return max(1, batch_size)


def _worker_manifest(worker_result: dict[str, Any]) -> dict[str, Any]:
    plan = worker_result.get("plan")
    records = worker_result.get("records")
    failures = worker_result.get("failures")
    return {
        "enabled": True,
        "runner": worker_result.get("runner"),
        "model": worker_result.get("model"),
        "batch_size": plan.get("batch_size") if isinstance(plan, dict) else None,
        "batch_count": plan.get("batch_count") if isinstance(plan, dict) else None,
        "max_workers": plan.get("max_workers") if isinstance(plan, dict) else None,
        "max_retries": plan.get("max_retries") if isinstance(plan, dict) else None,
        "allow_partial": plan.get("allow_partial") if isinstance(plan, dict) else None,
        "record_count": len(records) if isinstance(records, list) else 0,
        "failure_count": len(failures) if isinstance(failures, list) else 0,
        "run_metrics": worker_result.get("run_metrics"),
        "recursion": worker_result.get("recursion"),
        "lossless": bool(worker_result.get("lossless", True)),
    }
