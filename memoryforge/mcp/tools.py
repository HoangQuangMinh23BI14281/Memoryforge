"""Public MemoryForge MCP tool handlers.

Keep this module intentionally small. Codex CLI needs factual recall, bounded
context assembly, and one explicit RLM planning tool. Completed-turn capture is
owned by the mandatory project hook runner, not by ad-hoc MCP calls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from memoryforge.api import MemoryForge
from memoryforge.init.autoload import index_project_markdown, plan_project_markdown_analysis
from memoryforge.lcm import ContextBudget

T = TypeVar("T")
_MCP_RECALL_SNIPPET_CHARS = 1600


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


def recall_memory_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    def handler(mf: MemoryForge, args: dict[str, Any]) -> dict[str, Any]:
        results = mf.recall_long_term(
            agent_id=args["agent_id"],
            query=args["query"],
            top_k=int(args.get("top_k", args.get("limit", 10))),
            include_content=bool(args.get("include_content", False)),
            session_id=args.get("session_id"),
        )
        return {"results": _compact_recall_results_for_mcp(results)}

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


def index_analyze_tool(db_path: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Index Markdown and return host-subagent RLM analysis plans.

    This mirrors `memoryforge index --analyze`: MemoryForge owns chunking,
    LTM indexing, and batch planning. The active Codex host owns subagent
    execution. No model process is spawned from this MCP tool.
    """

    project_root = str(arguments.get("project_root") or ".")
    agent_id = str(arguments["agent_id"])
    chunk_size = _int_arg(arguments, "chunk_size", 12_000)
    overlap = _int_arg(arguments, "overlap", 1_000)
    max_files = _int_arg(arguments, "max_files", 200)
    max_file_bytes = _int_arg(arguments, "max_file_bytes", 1_000_000)
    analyze_max_files = _int_arg(arguments, "analyze_max_files", 5)
    analyze_min_bytes = _int_arg(arguments, "analyze_min_bytes", 20_000)
    limit = _int_arg(arguments, "limit", 10_000)
    batch_size = arguments.get("batch_size")
    force = bool(arguments.get("force", False))

    raw = index_project_markdown(
        db_path=db_path,
        agent_id=agent_id,
        project_root=project_root,
        chunk_size=chunk_size,
        overlap=overlap,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
    )
    analyze = plan_project_markdown_analysis(
        db_path=db_path,
        agent_id=agent_id,
        project_root=project_root,
        chunk_size=chunk_size,
        overlap=overlap,
        max_files=analyze_max_files,
        max_file_bytes=max_file_bytes,
        min_file_bytes=analyze_min_bytes,
        limit=limit,
        batch_size=int(batch_size) if batch_size is not None else None,
        force=force,
    )
    return {
        "enabled": True,
        "mode": "raw+host_subagent_plan",
        "external_model_calls": False,
        "raw": raw,
        "analyze": analyze,
    }


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


def _int_arg(args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key)
    if value is None:
        return default
    return int(value)


def _optional_int_arg(args: dict[str, Any], key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    return int(value)


def _compact_recall_results_for_mcp(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        content = item.get("content")
        if isinstance(content, str):
            text = content.lstrip("\ufeff")
            if len(text) > _MCP_RECALL_SNIPPET_CHARS:
                item["content"] = text[:_MCP_RECALL_SNIPPET_CHARS].rstrip() + " ..."
                item["content_truncated"] = True
                item["content_chars"] = len(text)
            else:
                item["content"] = text
                item["content_truncated"] = False
                item["content_chars"] = len(text)
        preview = item.get("preview")
        if isinstance(preview, str):
            item["preview"] = preview.lstrip("\ufeff")
        compacted.append(item)
    return compacted
