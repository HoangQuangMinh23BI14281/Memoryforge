"""Tool handlers shared by MCP and tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from memoryforge.api import MemoryForge
from memoryforge.init import ensure_project_initialized
from memoryforge.init.autoload import index_project_markdown
from memoryforge.lcm import ContextBudget

T = TypeVar("T")


def ensure_project_memory_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(arguments.get("project_root") or ".")).expanduser().resolve()
    return ensure_project_initialized(
        str(root),
        agent_id=str(arguments.get("agent_id") or "default"),
        configure_codex=False,
        auto_index=bool(arguments.get("auto_index", False)),
        install_hooks=False,
    )


def autoload_markdown_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(arguments.get("project_root") or ".")).expanduser().resolve()
    return index_project_markdown(
        db_path=db_path,
        agent_id=str(arguments.get("agent_id") or "default"),
        project_root=str(root),
        chunk_size=int(arguments.get("chunk_size", 12_000)),
        overlap=int(arguments.get("overlap", 1_000)),
        max_files=int(arguments.get("max_files", 200)),
        max_file_bytes=int(arguments.get("max_file_bytes", 1_000_000)),
    )



def _safe_close(mf: MemoryForge) -> None:
    try:
        mf.close()
    except Exception:
        pass


def _with_memory_forge(
    db_path: str,
    handler: Callable[[MemoryForge, dict[str, Any]], T],
    arguments: dict[str, Any],
) -> T:
    mf = MemoryForge(db_path=db_path)
    try:
        result = handler(mf, arguments)
    except Exception:
        _safe_close(mf)
        raise
    else:
        mf.close()
        return result


def store_conversation_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        turn_ids = mf.store_conversation(
            agent_id=args["agent_id"],
            turns=args.get("turns") or args.get("messages") or [],
            session_id=args.get("session_id"),
        )
        return {"turn_ids": turn_ids, "count": len(turn_ids)}

    return _with_memory_forge(db_path, handler, arguments)


def recall_conversation_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": mf.search(
                agent_id=args["agent_id"],
                query=args["query"],
                top_k=int(args.get("top_k", 10)),
            )
        }

    return _with_memory_forge(db_path, handler, arguments)


def recall_memory_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": mf.recall_long_term(
                agent_id=args["agent_id"],
                query=args["query"],
                top_k=int(args.get("top_k", args.get("limit", 10))),
                include_content=bool(args.get("include_content", False)),
                session_id=args.get("session_id"),
            )
        }

    return _with_memory_forge(db_path, handler, arguments)


def record_correction_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return mf.record_correction(
            agent_id=args["agent_id"],
            corrected_fact=args["corrected_fact"],
            wrong_item_id=args.get("wrong_item_id"),
            wrong_raw_ref=args.get("wrong_raw_ref"),
            session_id=args.get("session_id"),
            source=str(args.get("source") or "user"),
        )

    return _with_memory_forge(db_path, handler, arguments)


def record_contradiction_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return mf.record_contradiction(
            agent_id=args["agent_id"],
            statement=args["statement"],
            conflicting_item_ids=[str(item) for item in args.get("conflicting_item_ids") or []],
            conflicting_raw_refs=[str(ref) for ref in args.get("conflicting_raw_refs") or []],
            session_id=args.get("session_id"),
            source=str(args.get("source") or "user"),
        )

    return _with_memory_forge(db_path, handler, arguments)


def find_contradictions_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return mf.find_contradictions(
            agent_id=args["agent_id"],
            query=args.get("query"),
            limit=int(args.get("limit", args.get("top_k", 10))),
            include_content=bool(args.get("include_content", False)),
        )

    return _with_memory_forge(db_path, handler, arguments)


def build_context_bundle_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        budget = _context_budget_from_args(args)
        return mf.build_core_context_bundle(
            agent_id=args["agent_id"],
            session_id=args["session_id"],
            query=args["query"],
            system_prompt=str(args.get("system_prompt") or ""),
            budget=budget,
            top_k=int(args.get("top_k", args.get("limit", 5))),
            include_content=bool(args.get("include_content", False)),
            recall_content_policy=str(args.get("recall_content_policy") or "snippet"),
            long_term_token_budget=_optional_int_arg(args, "long_term_token_budget"),
        ).to_dict()

    return _with_memory_forge(db_path, handler, arguments)


def build_runtime_context_bundle_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        budget = _context_budget_from_args(args)
        return mf.build_runtime_context_bundle(
            agent_id=args["agent_id"],
            session_id=args["session_id"],
            query=args["query"],
            project_root=args["project_root"],
            runtime=str(args.get("runtime") or "auto"),
            system_prompt=str(args.get("system_prompt") or ""),
            budget=budget,
            top_k=int(args.get("top_k", args.get("limit", 5))),
            include_content=bool(args.get("include_content", False)),
            recall_content_policy=str(args.get("recall_content_policy") or "snippet"),
            long_term_token_budget=_optional_int_arg(args, "long_term_token_budget"),
        )

    return _with_memory_forge(db_path, handler, arguments)


def active_recall_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return mf.active_recall(
            agent_id=args["agent_id"],
            session_id=args.get("session_id"),
            focus=args.get("focus"),
            project_root=args.get("project_root"),
            limit=int(args.get("limit", args.get("top_k", 8))),
            include_content=bool(args.get("include_content", False)),
        )

    return _with_memory_forge(db_path, handler, arguments)


def ingest_file_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return mf.ingest_file(
            agent_id=args["agent_id"],
            path=Path(str(args["path"])),
            name=args.get("name"),
            chunk_size=int(args.get("chunk_size", 12_000)),
            overlap=int(args.get("overlap", 1_000)),
        )

    return _with_memory_forge(db_path, handler, arguments)


def _context_budget_from_args(args: dict[str, Any]) -> ContextBudget | None:
    budget_args = {
        "model_context_limit": args.get("model_context_limit"),
        "reserved_output_tokens": args.get("reserved_output_tokens"),
        "compaction_buffer": args.get("compaction_buffer"),
        "soft_threshold_fraction": args.get("soft_threshold_fraction"),
    }
    if all(value is None for value in budget_args.values()):
        return None
    return ContextBudget(
        model_context_limit=int(budget_args["model_context_limit"] or 200_000),
        reserved_output_tokens=int(budget_args["reserved_output_tokens"] or 4_000),
        compaction_buffer=int(budget_args["compaction_buffer"] or 2_000),
        soft_threshold_fraction=float(budget_args["soft_threshold_fraction"] or 0.6),
    )


def _optional_int_arg(args: dict[str, Any], key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    return int(value)


def rlm_load_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        value = args.get("path") or args.get("content") or ""
        if args.get("path"):
            candidate = Path(str(value))
            if candidate.is_absolute():
                candidate = candidate.resolve()
                if ".." in candidate.parts:
                    raise ValueError("Path traversal detected")
            load_value: str | Path = candidate
        else:
            load_value = str(value)
        return mf.rlm_load(
            agent_id=args["agent_id"],
            value=load_value,
            name=args.get("name"),
            chunk_size=int(args.get("chunk_size", 12_000)),
            overlap=int(args.get("overlap", 1_000)),
        )

    return _with_memory_forge(db_path, handler, arguments)


def rlm_search_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": mf.rlm_search(
                agent_id=args["agent_id"],
                query=args["query"],
                buffer_id=args.get("buffer_id"),
                limit=int(args.get("top_k", args.get("limit", 10))),
                mode=args.get("mode", "hybrid"),
            )
        }

    return _with_memory_forge(db_path, handler, arguments)


def rlm_chunk_get_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return {"chunk": mf.rlm_chunk_get(args["chunk_id"])}

    return _with_memory_forge(db_path, handler, arguments)


def rlm_dispatch_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return mf.rlm_dispatch(
            agent_id=args["agent_id"],
            buffer_id=args.get("buffer_id"),
            query=args.get("query"),
            limit=int(args.get("limit", 20)),
            batch_size=_optional_int_arg(args, "batch_size"),
        )

    return _with_memory_forge(db_path, handler, arguments)


def rlm_record_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        chunk_ids = args["chunk_ids"]
        if not isinstance(chunk_ids, list):
            raise ValueError(f"chunk_ids must be an array, got {type(chunk_ids).__name__}")
        return mf.rlm_record_result(
            agent_id=args["agent_id"],
            run_id=args["run_id"],
            chunk_ids=chunk_ids,
            analysis=args["analysis"],
            batch_index=args.get("batch_index"),
        )

    return _with_memory_forge(db_path, handler, arguments)


def rlm_aggregate_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        return mf.rlm_aggregate(
            agent_id=args["agent_id"],
            run_id=args["run_id"],
            summary=args.get("summary"),
        )

    return _with_memory_forge(db_path, handler, arguments)


def rlm_run_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        value = args.get("path") or args.get("content")
        load_value: str | Path | None = None
        if value is not None:
            if args.get("path"):
                candidate = Path(str(value))
                if candidate.is_absolute():
                    candidate = candidate.resolve()
                    if ".." in candidate.parts:
                        raise ValueError("Path traversal detected")
                load_value = candidate
            else:
                load_value = str(value)

        timeout = float(args.get("timeout", 900.0))
        if timeout < 0 or timeout > 3600:
            raise ValueError(f"timeout must be between 0 and 3600 seconds, got {timeout}")
        max_workers = int(args.get("max_workers", 1))
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        max_retries = int(args.get("max_retries", 0))
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")

        recursive_token_limit = args.get("recursive_token_limit")
        if recursive_token_limit is not None:
            recursive_token_limit = int(recursive_token_limit)
            if recursive_token_limit < 0:
                raise ValueError(
                    f"recursive_token_limit must be non-negative, got {recursive_token_limit}"
                )

        return mf.rlm_run(
            agent_id=args["agent_id"],
            value=load_value,
            name=args.get("name"),
            buffer_id=args.get("buffer_id"),
            query=args.get("query"),
            limit=int(args.get("limit", 20)),
            batch_size=_optional_int_arg(args, "batch_size"),
            chunk_size=int(args.get("chunk_size", 12_000)),
            overlap=int(args.get("overlap", 1_000)),
            runner=args.get("runner", "auto"),
            model=args.get("model"),
            base_url=args.get("base_url"),
            project_root=args.get("project_root"),
            timeout_s=timeout,
            max_workers=max_workers,
            max_retries=max_retries,
            allow_partial=bool(args.get("allow_partial", False)),
            synthesize=bool(args.get("synthesize", True)),
            recursive=bool(args.get("recursive", True)),
            max_recursive_rounds=int(args.get("max_recursive_rounds", 2)),
            recursive_token_limit=recursive_token_limit,
        )

    return _with_memory_forge(db_path, handler, arguments)
