"""CLI command handlers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from memoryforge.api import MemoryForge
from memoryforge.init import handle_hook_event, init_project
from memoryforge.init.autoload import index_project_markdown, plan_project_markdown_analysis


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_command(args: argparse.Namespace) -> int:
    if args.command == "mcp-server":
        from memoryforge.mcp.server import run_server

        run_server()
        return 0
    if args.command == "init":
        auto_index = bool(getattr(args, "index", False)) and not bool(
            getattr(args, "no_index", False)
        )
        _print_json(
            init_project(
                project_root=args.path,
                db_path=args.db,
                agent_id=args.agent_id,
                configure_codex=bool(getattr(args, "configure_codex", False)),
                auto_index=auto_index,
                install_hooks=bool(getattr(args, "install_hooks", False)),
                force=args.force,
            )
        )
        return 0
    if args.command == "hook":
        import sys

        try:
            stdin_text = "" if sys.stdin.closed or sys.stdin.isatty() else sys.stdin.read()
        except OSError:
            stdin_text = ""
        try:
            payload = handle_hook_event(
                event=args.event,
                db_path=args.db,
                agent_id=args.agent_id,
                project_root=args.project_root,
                stdin_text=stdin_text,
                source=args.source,
                runtime=args.runtime,
            )
        except Exception as exc:
            if os.environ.get("MEMORYFORGE_HOOK_STRICT") == "1":
                raise
            payload = {
                "event": args.event,
                "error": type(exc).__name__,
                "message": str(exc),
                "skipped": "hook failure ignored",
            }
        _print_json(payload)
        return 0
    if args.command == "index":
        project_root = Path(args.path).expanduser().resolve()
        db_path = _resolve_project_db_path(args.db, project_root)
        raw_result = index_project_markdown(
            db_path=db_path,
            agent_id=args.agent_id,
            project_root=str(project_root),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_files=args.max_files,
            max_file_bytes=args.max_file_bytes,
        )
        analyze_result = {"enabled": False, "skipped": "pass --analyze to prepare host sub-agent batches"}
        if args.analyze:
            analyze_result = plan_project_markdown_analysis(
                db_path=db_path,
                agent_id=args.agent_id,
                project_root=str(project_root),
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                max_files=args.analyze_max_files,
                max_file_bytes=args.max_file_bytes,
                min_file_bytes=args.analyze_min_bytes,
                limit=args.limit,
                batch_size=args.batch_size,
                force=args.force,
            )
            ignored = [
                name
                for name, value in {
                    "runner": args.runner if args.runner != "host" else None,
                    "model": args.model,
                    "base_url": args.base_url,
                    "timeout": args.timeout if args.timeout != 900.0 else None,
                    "max_workers": args.max_workers if args.max_workers != 1 else None,
                    "max_retries": args.max_retries if args.max_retries != 0 else None,
                    "allow_partial": args.allow_partial or None,
                    "no_synthesis": args.no_synthesis or None,
                    "no_recursive": args.no_recursive or None,
                    "max_recursive_rounds": (
                        args.max_recursive_rounds if args.max_recursive_rounds != 2 else None
                    ),
                    "recursive_token_limit": args.recursive_token_limit,
                }.items()
                if value is not None
            ]
            if ignored:
                analyze_result["ignored_external_runner_options"] = ignored
        _print_json({"enabled": True, "mode": "raw", "raw": raw_result, "analyze": analyze_result})
        return 0
    mf = MemoryForge(db_path=args.db)
    try:
        if args.command == "store-session":
            payload = json.loads(Path(args.session_file).read_text(encoding="utf-8"))
            turns_payload = (
                payload.get("turns") or payload.get("messages")
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(turns_payload, list) or not all(
                isinstance(turn, dict) for turn in turns_payload
            ):
                raise SystemExit("store-session expects a JSON list, turns, or messages array")
            turns: list[dict[str, Any]] = turns_payload
            session_id = args.session_id or (
                payload.get("session_id") if isinstance(payload, dict) else None
            )
            _print_json(
                {
                    "turn_ids": mf.store_conversation(
                        agent_id=args.agent_id,
                        turns=turns,
                        session_id=session_id,
                    )
                }
            )
        elif args.command == "search":
            if args.ensemble:
                _print_json(mf.search_ensemble(args.agent_id, args.query, args.limit))
            else:
                _print_json(mf.search(args.agent_id, args.query, args.limit))
        elif args.command == "recall-memory":
            _print_json(
                mf.recall_long_term(
                    args.agent_id,
                    args.query,
                    args.limit,
                    include_content=args.include_content,
                    session_id=args.session_id,
                )
            )
        elif args.command == "active-recall":
            _print_json(
                mf.active_recall(
                    args.agent_id,
                    session_id=args.session_id,
                    focus=args.focus,
                    project_root=args.project_root,
                    limit=args.limit,
                    include_content=args.include_content,
                )
            )
        elif args.command == "record-contradiction":
            _print_json(
                mf.record_contradiction(
                    args.agent_id,
                    args.statement,
                    conflicting_item_ids=args.conflicting_item_id,
                    conflicting_raw_refs=args.conflicting_raw_ref,
                    session_id=args.session_id,
                    source=args.source,
                )
            )
        elif args.command == "find-contradictions":
            _print_json(
                mf.find_contradictions(
                    args.agent_id,
                    query=args.query,
                    limit=args.limit,
                    include_content=args.include_content,
                )
            )
        elif args.command == "runtime-context":
            from memoryforge.lcm import ContextBudget

            _print_json(
                mf.build_runtime_context_bundle(
                    agent_id=args.agent_id,
                    session_id=args.session_id,
                    query=args.query,
                    project_root=args.project_root,
                    runtime=args.runtime,
                    budget=ContextBudget(
                        model_context_limit=args.context_limit,
                        reserved_output_tokens=args.reserved_output,
                        compaction_buffer=args.compaction_buffer,
                    ),
                    top_k=args.top_k,
                    include_content=args.include_content,
                    recall_content_policy=args.recall_content_policy,
                    long_term_token_budget=args.long_term_token_budget,
                )
            )
        elif args.command == "long-term-source":
            _print_json(mf.long_term_source(args.agent_id, args.item_id))
        elif args.command == "lcm-context":
            from memoryforge.lcm import ContextBudget

            budget = ContextBudget(
                model_context_limit=args.context_limit,
                reserved_output_tokens=args.reserved_output,
                compaction_buffer=args.compaction_buffer,
            )
            recall_results = []
            if args.recall_query:
                if not args.agent_id:
                    raise SystemExit("--agent-id is required with --recall-query")
                augmented = mf.lcm_build_context_with_recall(
                    args.session_id,
                    args.agent_id,
                    args.recall_query,
                    budget=budget,
                    top_k=args.recall_limit,
                )
                context = augmented["context"]
                recall_results = augmented["long_term_recall"]
            else:
                context = mf.lcm_build_context(args.session_id, budget=budget)
            _print_json(
                {
                    "token_estimate": context.token_estimate,
                    "hard_limit": context.budget.hard_limit,
                    "soft_limit": context.budget.soft_limit,
                    "has_summary": context.has_summary,
                    "truncated": context.truncated,
                    "summary_node_ids": context.summary_node_ids,
                    "raw_message_ids": context.raw_message_ids,
                    "long_term_recall": recall_results,
                    "messages": [message.__dict__ for message in context.messages],
                }
            )
        elif args.command == "lcm-sessions":
            from memoryforge.lcm import ContextBudget

            budget = ContextBudget(
                model_context_limit=args.context_limit,
                reserved_output_tokens=args.reserved_output,
                compaction_buffer=args.compaction_buffer,
            )
            _print_json(_lcm_sessions_payload(mf, agent_id=args.agent_id, limit=args.limit, budget=budget))
        elif args.command == "lcm-messages":
            _print_json(
                _lcm_messages_payload(
                    mf,
                    session_id=args.session_id,
                    agent_id=args.agent_id,
                    limit=args.limit,
                    include_content=args.include_content,
                    include_summaries=args.include_summaries,
                )
            )
        elif args.command == "lcm-summary":
            _print_json(
                _lcm_summary_payload(
                    mf,
                    session_id=args.session_id,
                    limit=args.limit,
                    include_content=args.include_content,
                    include_superseded=args.all,
                )
            )
        elif args.command == "lcm-compact":
            from memoryforge.lcm import ContextBudget

            result = mf.lcm_compact_if_needed(
                args.agent_id,
                args.session_id,
                budget=ContextBudget(
                    model_context_limit=args.context_limit,
                    reserved_output_tokens=args.reserved_output,
                    compaction_buffer=args.compaction_buffer,
                ),
                force=args.force,
                defer_soft=args.defer_soft,
                runner=args.runner,
                model=args.model,
                project_root=args.project_root,
                base_url=args.base_url,
            )
            _print_json(
                {
                    "triggered": result.triggered,
                    "rounds": result.rounds,
                    "before_tokens": result.before_tokens,
                    "after_tokens": result.after_tokens,
                    "delta_tokens": result.delta_tokens,
                    "expanded": result.expanded,
                    "effective": result.effective,
                    "deferred": result.deferred,
                    "reason": result.reason,
                    "summary_node_ids": result.summary_node_ids,
                    "pruned": result.pruned.__dict__,
                    "decision": result.decision.__dict__,
                }
            )
        elif args.command == "lcm-maintain":
            from memoryforge.lcm import ContextBudget

            _print_json(
                mf.lcm_compact_due(
                    args.agent_id,
                    budget=ContextBudget(
                        model_context_limit=args.context_limit,
                        reserved_output_tokens=args.reserved_output,
                        compaction_buffer=args.compaction_buffer,
                    ),
                    hard_only=args.hard_only,
                    limit=args.limit,
                    max_rounds=args.max_rounds,
                    runner=args.runner,
                    model=args.model,
                    project_root=args.project_root,
                    base_url=args.base_url,
                )
            )
        elif args.command == "ingest-file":
            _print_json(
                mf.ingest_file(
                    agent_id=args.agent_id,
                    path=Path(args.path),
                    name=args.name,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                )
            )
        elif args.command == "rlm-load":
            input_path = Path(args.input)
            value = input_path if input_path.exists() else args.input
            _print_json(
                mf.rlm_load(
                    agent_id=args.agent_id,
                    value=value,
                    name=args.name,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                )
            )
        elif args.command == "rlm-search":
            _print_json(
                mf.rlm_search(
                    agent_id=args.agent_id,
                    query=args.query,
                    buffer_id=args.buffer_id,
                    limit=args.limit,
                    mode=args.mode,
                )
            )
        elif args.command == "rlm-chunk-get":
            _print_json(mf.rlm_chunk_get(args.chunk_id))
        elif args.command == "dispatch":
            _print_json(
                mf.rlm_dispatch(
                    agent_id=args.agent_id,
                    buffer_id=args.buffer_id,
                    query=args.query,
                    limit=args.limit,
                    batch_size=args.batch_size,
                )
            )
        elif args.command == "context-get":
            ref_kind, _separator, ref_value = args.ref.partition(":")
            if ref_kind == "rlm_chunk":
                _print_json(mf.rlm_chunk_get(ref_value))
            else:
                raise SystemExit(f"Unsupported ref kind for context-get: {ref_kind}")
        elif args.command == "rlm-record":
            import sys

            analysis = (
                sys.stdin.read()
                if args.analysis_file == "-"
                else Path(args.analysis_file).read_text(encoding="utf-8")
            )
            _print_json(
                mf.rlm_record_result(
                    agent_id=args.agent_id,
                    run_id=args.run_id,
                    chunk_ids=args.chunk_id,
                    analysis=analysis,
                    batch_index=args.batch_index,
                )
            )
        elif args.command == "aggregate":
            summary = (
                Path(args.summary_file).read_text(encoding="utf-8") if args.summary_file else None
            )
            _print_json(
                mf.rlm_aggregate(
                    args.agent_id,
                    args.run_id,
                    summary=summary,
                    expected_batch_count=args.expected_batches,
                )
            )
        elif args.command == "rlm-run":
            value = None
            if args.input:
                input_path = Path(args.input)
                value = input_path if input_path.exists() else args.input
            _print_json(
                mf.rlm_run(
                    agent_id=args.agent_id,
                    value=value,
                    name=args.name,
                    buffer_id=args.buffer_id,
                    query=args.query,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    runner=args.runner,
                    model=args.model,
                    base_url=args.base_url,
                    project_root=args.project_root,
                    timeout_s=args.timeout,
                    max_workers=args.max_workers,
                    max_retries=args.max_retries,
                    allow_partial=args.allow_partial,
                    synthesize=not args.no_synthesis,
                    recursive=not args.no_recursive,
                    max_recursive_rounds=args.max_recursive_rounds,
                    recursive_token_limit=args.recursive_token_limit,
                )
            )
        elif args.command == "chunk":
            if args.conversation_json:
                input_path = Path(args.input)
                raw = input_path.read_text(encoding="utf-8") if input_path.exists() else args.input
                value = json.loads(raw)
            else:
                input_path = Path(args.input)
                value = input_path if input_path.exists() else args.input
            _print_json(mf.chunk_content(value))
        elif args.command == "benchmark":
            from memoryforge.benchmark.adapter import BenchmarkCase
            from memoryforge.benchmark.secondbrain import run_second_brain_benchmark

            if args.dataset == "second-brain":
                _print_json(
                    run_second_brain_benchmark(
                        args.db,
                        agent_id=args.agent_id,
                        mode=args.mode,
                    )
                )
            else:
                _print_json(
                    {
                        "dataset": args.dataset,
                        "mode": args.mode,
                        "cases": [BenchmarkCase("synthetic", "ping").__dict__],
                    }
                )
    finally:
        mf.close()
    return 0


def _resolve_project_db_path(db_arg: str, project_root: Path) -> str:
    if db_arg != "~/.memoryforge/memory.db":
        return db_arg
    config_path = project_root / ".memoryforge" / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            configured = config.get("db_path")
            if configured:
                return str(configured)
        except (OSError, json.JSONDecodeError):
            pass
    return str(project_root / ".memoryforge" / "memory.db")


def _lcm_sessions_payload(
    mf: MemoryForge,
    *,
    agent_id: str | None,
    limit: int,
    budget: Any,
) -> dict[str, Any]:
    params: list[Any] = []
    where = ""
    if agent_id:
        where = "WHERE s.agent = ?"
        params.append(agent_id)
    rows = mf.lcm_store.conn.execute(
        f"""
        SELECT
            s.id,
            s.agent,
            s.created_at,
            s.updated_at,
            (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id AND m.is_summary = 0),
            (SELECT COALESCE(SUM(tokens_total), 0)
             FROM messages m WHERE m.session_id = s.id AND m.is_summary = 0),
            (SELECT COUNT(*) FROM summary_nodes sn
             WHERE sn.session_id = s.id AND sn.superseded = 0),
            (SELECT COUNT(*) FROM context_items ci WHERE ci.session_id = s.id)
        FROM sessions s
        {where}
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        [*params, max(1, int(limit))],
    ).fetchall()
    sessions: list[dict[str, Any]] = []
    for row in rows:
        context = mf.lcm_build_context(str(row[0]), budget=budget)
        sessions.append(
            {
                "session_id": row[0],
                "agent_id": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": row[4],
                "stored_tokens": row[5],
                "active_summary_count": row[6],
                "context_item_count": row[7],
                "active_context_tokens": context.token_estimate,
                "active_context_messages": len(context.messages),
                "has_summary": context.has_summary,
                "truncated": context.truncated,
                "soft_limit": context.budget.soft_limit,
                "hard_limit": context.budget.hard_limit,
            }
        )
    return {"sessions": sessions, "count": len(sessions)}


def _lcm_messages_payload(
    mf: MemoryForge,
    *,
    session_id: str,
    agent_id: str | None,
    limit: int,
    include_content: bool,
    include_summaries: bool,
) -> dict[str, Any]:
    messages = mf.lcm_store.get_messages(session_id, include_summaries=include_summaries)
    if agent_id:
        messages = [message for message in messages if message.agent_id == agent_id]
    selected = messages[-max(1, int(limit)) :]
    return {
        "session_id": session_id,
        "count": len(selected),
        "total_matching": len(messages),
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "agent_id": message.agent_id,
                "created_at": message.created_at,
                "is_summary": message.is_summary,
                "token_estimate": message.token_estimate,
                "parts": [
                    {
                        "id": part.id,
                        "part_type": part.part_type,
                        "content_id": part.content_id,
                        "token_estimate": part.token_estimate,
                        "tool_name": part.tool_name,
                        "tool_call_id": part.tool_call_id,
                        "tool_state": part.tool_state,
                        "compacted_at": part.compacted_at,
                        **(
                            {"content": part.content}
                            if include_content
                            else {"preview": _preview(part.content)}
                        ),
                    }
                    for part in message.parts
                ],
            }
            for message in selected
        ],
    }


def _lcm_summary_payload(
    mf: MemoryForge,
    *,
    session_id: str,
    limit: int,
    include_content: bool,
    include_superseded: bool,
) -> dict[str, Any]:
    where = "session_id = ?"
    params: list[Any] = [session_id]
    if not include_superseded:
        where += " AND superseded = 0"
    rows = mf.lcm_engine.dag.conn.execute(
        f"""
        SELECT id, level, kind, span_start_message_id, span_end_message_id,
               token_count, created_at, parent_node_ids, superseded, file_ids,
               source_refs, content
        FROM summary_nodes
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        [*params, max(1, int(limit))],
    ).fetchall()
    nodes = []
    for row in rows:
        nodes.append(
            {
                "id": row[0],
                "level": row[1],
                "kind": row[2],
                "span_start": row[3],
                "span_end": row[4],
                "token_count": row[5],
                "created_at": row[6],
                "parent_node_ids": _json_list(row[7]),
                "superseded": bool(row[8]),
                "file_ids": _json_list(row[9]),
                "source_refs": _json_list(row[10]),
                **(
                    {"content": row[11]}
                    if include_content
                    else {"preview": _preview(str(row[11] or ""))}
                ),
            }
        )
    return {"session_id": session_id, "count": len(nodes), "summary_nodes": nodes}


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
