"""CLI command handlers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memoryforge.api import MemoryForge
from memoryforge.init import handle_hook_event, init_project


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_command(args: argparse.Namespace) -> int:
    if args.command == "mcp-server":
        from memoryforge.mcp.server import run_server

        run_server()
        return 0
    if args.command == "init":
        _print_json(
            init_project(
                project_root=args.path,
                db_path=args.db,
                agent_id=args.agent_id,
                configure_codex=not args.no_codex,
                force=args.force,
            )
        )
        return 0
    if args.command == "hook":
        import sys

        _print_json(
            handle_hook_event(
                event=args.event,
                db_path=args.db,
                agent_id=args.agent_id,
                project_root=args.project_root,
                stdin_text=sys.stdin.read(),
            )
        )
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
            _print_json(mf.rlm_aggregate(args.agent_id, args.run_id, summary=summary))
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
