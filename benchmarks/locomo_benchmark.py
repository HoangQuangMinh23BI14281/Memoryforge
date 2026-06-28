#!/usr/bin/env python3
"""Run LoCoMo through MemoryForge benchmark paths."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from memoryforge.agents import SubAgentOperator, SubAgentTask
from memoryforge.api import MemoryForge
from memoryforge.benchmark import LoComoAdapter
from memoryforge.benchmark.adapter import BenchmarkCase
from memoryforge.lcm import ContextBudget

DEFAULT_VECTOR_BACKEND = "fastembed"
DEFAULT_VECTOR_MODEL = "BAAI/bge-small-en-v1.5"
LOCOMO_MODE_CONTRACTS: dict[str, dict[str, bool | str]] = {
    "adapter-search": {
        "pipeline": "conversation-search",
        "ingests": True,
        "builds_context_bundle": False,
        "uses_core_answer_runner": False,
        "uses_rlm_worker": False,
        "uses_lcm_worker": False,
        "produces_prediction": True,
        "requires_runner": False,
    },
    "ingest-only": {
        "pipeline": "rlm-ltm",
        "ingests": True,
        "builds_context_bundle": False,
        "uses_core_answer_runner": False,
        "uses_rlm_worker": True,
        "uses_lcm_worker": False,
        "produces_prediction": False,
        "requires_runner": False,
    },
    "context-only": {
        "pipeline": "ltm-lcm",
        "ingests": False,
        "builds_context_bundle": True,
        "uses_core_answer_runner": False,
        "uses_rlm_worker": False,
        "uses_lcm_worker": False,
        "produces_prediction": False,
        "requires_runner": False,
    },
    "rlm-worker": {
        "pipeline": "rlm-worker",
        "ingests": True,
        "builds_context_bundle": False,
        "uses_core_answer_runner": False,
        "uses_rlm_worker": True,
        "uses_lcm_worker": False,
        "produces_prediction": False,
        "requires_runner": True,
    },
    "core-answer": {
        "pipeline": "rlm-ltm-lcm-core-answer",
        "ingests": True,
        "builds_context_bundle": True,
        "uses_core_answer_runner": True,
        "uses_rlm_worker": True,
        "uses_lcm_worker": False,
        "produces_prediction": True,
        "requires_runner": True,
    },
}
RUNNER_MODES = {
    mode
    for mode, contract in LOCOMO_MODE_CONTRACTS.items()
    if bool(contract["requires_runner"])
}
MODEL_PROMPT_AUDIT_FIELDS = {
    "active_recall",
    "long_term_recall",
    "summary_nodes",
    "raw_refs",
    "token_estimate",
    "budget",
    "provenance",
    "diagnostics",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--db", default="~/.memoryforge/memory.db")
    parser.add_argument("--agent-id", default="benchmark")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=sorted(LOCOMO_MODE_CONTRACTS),
        default="adapter-search",
    )
    parser.add_argument("--context-limit", type=int, default=16_000)
    parser.add_argument("--chunk-size", type=int, default=4_000)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument(
        "--recall-content-policy",
        choices=["snippet", "champion", "full", "auto", "preview"],
        default="snippet",
    )
    parser.add_argument("--long-term-token-budget", type=int, default=None)
    parser.add_argument("--runner", default=os.environ.get("MEMORYFORGE_SUBAGENT_RUNNER", "codex"))
    parser.add_argument(
        "--model", default=os.environ.get("MEMORYFORGE_MODEL") or os.environ.get("OPENAI_MODEL")
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MEMORYFORGE_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--rlm-max-workers",
        type=int,
        default=int(os.environ.get("MEMORYFORGE_RLM_MAX_WORKERS", "1")),
        help="Maximum parallel RLM analysis sub-agents in rlm-worker/core-answer modes.",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--jsonl-output", default="")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--vector-backend", default=os.environ.get("MEMORYFORGE_VECTOR_BACKEND") or DEFAULT_VECTOR_BACKEND)
    parser.add_argument("--vector-model", default=os.environ.get("MEMORYFORGE_VECTOR_MODEL") or DEFAULT_VECTOR_MODEL)
    parser.add_argument("--require-vector-model", action="store_true")
    args = parser.parse_args()

    if args.rlm_max_workers < 1:
        raise SystemExit("--rlm-max-workers must be at least 1")

    requires_runner = args.mode in RUNNER_MODES
    if requires_runner and (
        args.runner == "mock" or os.environ.get("MEMORYFORGE_SUBAGENT_RUNNER") == "mock"
    ):
        raise SystemExit(f"Refusing to run {args.mode} benchmark with mock runner")
    if requires_runner and not args.model:
        raise SystemExit(
            f"{args.mode} benchmark requires --model or MEMORYFORGE_MODEL/OPENAI_MODEL"
        )

    if args.vector_backend:
        os.environ["MEMORYFORGE_VECTOR_BACKEND"] = args.vector_backend
    if args.vector_model:
        os.environ["MEMORYFORGE_VECTOR_MODEL"] = args.vector_model
    if args.require_vector_model or args.mode in {"rlm-worker", "core-answer"}:
        os.environ["MEMORYFORGE_REQUIRE_VECTOR_MODEL"] = "1"

    for output in (args.output, args.jsonl_output):
        if args.clean_output and output:
            Path(output).expanduser().unlink(missing_ok=True)

    adapter = LoComoAdapter()
    cases = adapter.load_cases(args.dataset)[: args.limit]
    mf = MemoryForge(args.db)
    operator = (
        SubAgentOperator(
            runner=args.runner,
            model=args.model,
            base_url=args.base_url,
            project_root=args.project_root,
            timeout_s=args.timeout,
        )
        if requires_runner
        else None
    )
    actual_vector_model = ""
    actual_vector_backend = ""
    ingested_keys: set[str] = set()
    results: list[dict[str, Any]] = []
    jsonl_handle = None
    if args.jsonl_output:
        jsonl_path = Path(args.jsonl_output).expanduser()
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    try:
        for case in cases:
            if args.mode in {"adapter-search", "context-only"}:
                result = _run_legacy_case(
                    mf=mf,
                    adapter=adapter,
                    case=case,
                    agent_id=args.agent_id,
                    mode=args.mode,
                    top_k=args.top_k,
                    context_limit=args.context_limit,
                    recall_content_policy=args.recall_content_policy,
                    long_term_token_budget=args.long_term_token_budget,
                    ingested_keys=ingested_keys,
                )
            else:
                result = _run_rlm_case(
                    mf=mf,
                    operator=operator,
                    adapter=adapter,
                    case=case,
                    agent_id=args.agent_id,
                    mode=args.mode,
                    top_k=args.top_k,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                    context_limit=args.context_limit,
                    recall_content_policy=args.recall_content_policy,
                    long_term_token_budget=args.long_term_token_budget,
                    rlm_max_workers=args.rlm_max_workers,
                )
            results.append(result)
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        actual_vector_model = mf.long_term.vector.model_name
        actual_vector_backend = mf.long_term.vector.embedding_backend
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        mf.close()

    summary = {
        "total": len(results),
        "correct": sum(result["correct"] is True for result in results),
        "incorrect": sum(result["correct"] is False for result in results),
        "unknown": sum(result["correct"] is None for result in results),
        "mode": args.mode,
        "mode_contract": _mode_contract(args.mode),
        "benchmark": adapter.name,
        "top_k": args.top_k,
        "context_limit": args.context_limit if _mode_contract(args.mode)["builds_context_bundle"] else None,
        "chunk_size": args.chunk_size if _mode_contract(args.mode)["uses_rlm_worker"] else None,
        "overlap": args.overlap if _mode_contract(args.mode)["uses_rlm_worker"] else None,
        "runner": args.runner if requires_runner else None,
        "model": args.model if requires_runner else None,
        "recall_content_policy": args.recall_content_policy
        if _mode_contract(args.mode)["builds_context_bundle"]
        else None,
        "ingested_sessions": len(ingested_keys),
        "vector_model": actual_vector_model,
        "vector_backend": actual_vector_backend,
        "requested_vector_model": args.vector_model or None,
        "requested_vector_backend": args.vector_backend or None,
        "rlm_max_workers": args.rlm_max_workers if _mode_contract(args.mode)["uses_rlm_worker"] else None,
        "performance": _performance_summary(results),
        "db_counts": _db_counts(args.db),
    }
    payload = {"summary": summary, "results": results}
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_legacy_case(
    *,
    mf: MemoryForge,
    adapter: LoComoAdapter,
    case: BenchmarkCase,
    agent_id: str,
    mode: str,
    top_k: int,
    context_limit: int,
    recall_content_policy: str,
    long_term_token_budget: int | None,
    ingested_keys: set[str],
) -> dict[str, Any]:
    key = adapter.ingestion_key(case)
    ingestion_started = time.perf_counter()
    message_ids: list[str] = []
    ingested_new = False
    deduped = key in ingested_keys or _case_already_ingested(
        mf,
        adapter,
        case,
        agent_id=agent_id,
    )
    if mode == "context-only" and not deduped:
        raise RuntimeError(
            "context-only mode requires pre-ingested LoCoMo sessions; run "
            "adapter-search first or ingest conversation sessions with the same --db and --agent-id"
        )
    if mode == "adapter-search" and not deduped:
        message_ids = adapter.ingest_case(mf, case, agent_id=agent_id)
        ingested_new = True
    ingested_keys.add(key)
    ingestion_ms = (time.perf_counter() - ingestion_started) * 1000.0
    if mode == "context-only":
        return _run_context_only_case(
            mf=mf,
            adapter=adapter,
            case=case,
            agent_id=agent_id,
            top_k=top_k,
            context_limit=context_limit,
            recall_content_policy=recall_content_policy,
            long_term_token_budget=long_term_token_budget,
        )

    result = adapter.run_case(
        mf,
        case,
        agent_id=agent_id,
        top_k=top_k,
    ).__dict__
    result["diagnostics"] = {
        "mode": mode,
        "pipeline": _mode_contract(mode)["pipeline"],
        "mode_contract": _mode_contract(mode),
        "ingestion": {
            "performed": ingested_new,
            "deduped": deduped,
            "message_count": len(message_ids),
            "latency_ms": ingestion_ms,
        },
        "vector_model": mf.long_term.vector.model_name,
        "vector_backend": mf.long_term.vector.embedding_backend,
        "db_counts": _db_counts(mf.db_path),
    }
    return result


def _run_rlm_case(
    *,
    mf: MemoryForge,
    operator: SubAgentOperator | None,
    adapter: LoComoAdapter,
    case: BenchmarkCase,
    agent_id: str,
    mode: str,
    top_k: int,
    chunk_size: int,
    overlap: int,
    context_limit: int,
    recall_content_policy: str,
    long_term_token_budget: int | None,
    rlm_max_workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    mode_contract = _mode_contract(mode)
    source_text = _locomo_source_text(adapter, case)
    if mode == "rlm-worker":
        if operator is None:
            raise RuntimeError("rlm-worker mode requires a configured runner")
        rlm_result = mf.rlm_run(
            agent_id=agent_id,
            value=source_text,
            name=f"{case.case_id}.locomo",
            query=case.question,
            limit=top_k,
            batch_size=4,
            chunk_size=chunk_size,
            overlap=overlap,
            runner=operator.runner,
            model=operator.model,
            base_url=operator.base_url,
            project_root=operator.project_root,
            timeout_s=operator.timeout_s,
            max_workers=rlm_max_workers,
        )
        return {
            "case_id": case.case_id,
            "question": case.question,
            "expected": case.answer,
            "prediction": None,
            "correct": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "core_answer_runner": None,
            "diagnostics": {
                "mode": mode,
                "pipeline": mode_contract["pipeline"],
                "mode_contract": mode_contract,
                "vector_model": mf.long_term.vector.model_name,
                "vector_backend": mf.long_term.vector.embedding_backend,
                "rlm_worker": rlm_result,
                "db_counts": _db_counts(mf.db_path),
            },
        }

    loaded = mf.rlm_load(
        agent_id,
        source_text,
        name=f"{case.case_id}.locomo",
        chunk_size=chunk_size,
        overlap=overlap,
        runner=operator.runner if operator is not None else "auto",
        model=operator.model if operator is not None else None,
        base_url=operator.base_url if operator is not None else None,
        project_root=operator.project_root if operator is not None else None,
        timeout_s=operator.timeout_s if operator is not None else 900.0,
        batch_size=4,
        max_workers=rlm_max_workers,
    )
    if mode == "ingest-only":
        return {
            "case_id": case.case_id,
            "question": case.question,
            "expected": case.answer,
            "prediction": None,
            "correct": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "core_answer_runner": None,
            "diagnostics": {
                "mode": mode,
                "pipeline": mode_contract["pipeline"],
                "mode_contract": mode_contract,
                "vector_model": mf.long_term.vector.model_name,
                "vector_backend": mf.long_term.vector.embedding_backend,
                "rlm_buffer_id": loaded["buffer_id"],
                "rlm_chunk_count": loaded["chunk_count"],
                "ltm_indexed_count": len(loaded.get("long_term_item_ids", [])),
                "ingestion_manifest": loaded.get("ingestion_manifest"),
                "rlm_worker": loaded.get("rlm_worker"),
                "deduped": bool(loaded.get("deduped")),
                "rlm_deduped": bool(loaded.get("rlm_deduped")),
                "ltm_deduped": bool(loaded.get("ltm_deduped")),
                "db_counts": _db_counts(mf.db_path),
            },
        }

    if operator is None:
        raise RuntimeError("core-answer mode requires a configured runner")
    session_id = _question_session_id(adapter.ingestion_key(case), case.case_id)
    question_message_id = _append_question_to_lcm(
        mf,
        agent_id,
        session_id,
        question=case.question,
    )
    context_bundle = mf.build_core_context_bundle(
        agent_id=agent_id,
        session_id=session_id,
        query=case.question,
        budget=ContextBudget(model_context_limit=context_limit),
        top_k=top_k,
        include_content=True,
        recall_content_policy=recall_content_policy,
        long_term_token_budget=long_term_token_budget,
    )
    model_payload = context_bundle.to_model_payload()
    answer_runner = _answer_with_core_answer_runner(
        operator,
        case_id=case.case_id,
        question=case.question,
        model_payload=model_payload,
    )
    prediction = answer_runner["answer"]
    diagnostics = {
        "mode": mode,
        "pipeline": mode_contract["pipeline"],
        "mode_contract": mode_contract,
        "vector_model": mf.long_term.vector.model_name,
        "vector_backend": mf.long_term.vector.embedding_backend,
        "rlm_buffer_id": loaded["buffer_id"],
        "rlm_chunk_count": loaded["chunk_count"],
        "ltm_indexed_count": len(loaded.get("long_term_item_ids", [])),
        "ingestion_manifest": loaded.get("ingestion_manifest"),
        "rlm_worker": loaded.get("rlm_worker"),
        "deduped": bool(loaded.get("deduped")),
        "rlm_deduped": bool(loaded.get("rlm_deduped")),
        "ltm_deduped": bool(loaded.get("ltm_deduped")),
        "question_message_id": question_message_id,
        "context": {
            "message_count": context_bundle.diagnostics["context"]["message_count"],
            "token_estimate": context_bundle.token_estimate,
            "has_summary": context_bundle.diagnostics["context"]["has_summary"],
            "truncated": context_bundle.diagnostics["context"]["truncated"],
            "summary_node_ids": context_bundle.diagnostics["context"]["summary_node_ids"],
            "sources": [message["source"] for message in model_payload["messages"]],
            "long_term_recall_count": len(context_bundle.long_term_recall),
            "raw_ref_count": len(context_bundle.raw_refs),
        },
        "context_bundle": {
            "bundle_only": context_bundle.diagnostics["bundle_only"],
            "answer_model_used": context_bundle.diagnostics["answer_model_used"],
            "budget": context_bundle.budget,
            "raw_refs": context_bundle.raw_refs,
            "provenance": context_bundle.provenance,
            "summary_nodes": context_bundle.summary_nodes,
            "retrieval": context_bundle.diagnostics["retrieval"],
            "latency_ms": context_bundle.diagnostics["latency_ms"],
        },
        "retrieval": _retrieval_diagnostics(mf, agent_id, case.question, top_k=top_k),
        "db_counts": _db_counts(mf.db_path),
    }
    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected": case.answer,
        "prediction": prediction,
        "correct": _answer_matches(case.answer, prediction),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "core_answer_runner": answer_runner,
        "diagnostics": diagnostics,
    }


def _run_context_only_case(
    *,
    mf: MemoryForge,
    adapter: LoComoAdapter,
    case: BenchmarkCase,
    agent_id: str,
    top_k: int,
    context_limit: int,
    recall_content_policy: str,
    long_term_token_budget: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    session_id = _question_session_id(adapter.ingestion_key(case), case.case_id)
    question_message_id = _append_question_to_lcm(
        mf,
        agent_id,
        session_id,
        question=case.question,
    )
    context_bundle = mf.build_core_context_bundle(
        agent_id=agent_id,
        session_id=session_id,
        query=case.question,
        budget=ContextBudget(model_context_limit=context_limit),
        top_k=top_k,
        include_content=True,
        recall_content_policy=recall_content_policy,
        long_term_token_budget=long_term_token_budget,
    )
    context_diagnostics = context_bundle.diagnostics["context"]
    diagnostics = {
        "mode": "context-only",
        "pipeline": _mode_contract("context-only")["pipeline"],
        "mode_contract": _mode_contract("context-only"),
        "ingestion": {
            "performed": False,
            "preingested": True,
            "message_count": 0,
            "latency_ms": 0.0,
        },
        "question_message_id": question_message_id,
        "context": {
            "message_count": context_diagnostics["message_count"],
            "token_estimate": context_bundle.token_estimate,
            "has_summary": context_diagnostics["has_summary"],
            "truncated": context_diagnostics["truncated"],
            "summary_node_ids": context_diagnostics["summary_node_ids"],
            "sources": [message["source"] for message in context_bundle.messages],
            "long_term_recall_count": len(context_bundle.long_term_recall),
            "raw_ref_count": len(context_bundle.raw_refs),
        },
        "context_bundle": {
            "bundle_only": context_bundle.diagnostics["bundle_only"],
            "answer_model_used": context_bundle.diagnostics["answer_model_used"],
            "budget": context_bundle.budget,
            "raw_refs": context_bundle.raw_refs,
            "provenance": context_bundle.provenance,
            "summary_nodes": context_bundle.summary_nodes,
            "retrieval": context_bundle.diagnostics["retrieval"],
            "latency_ms": context_bundle.diagnostics["latency_ms"],
        },
    }
    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected": case.answer,
        "prediction": None,
        "correct": None,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "core_answer_runner": None,
        "diagnostics": diagnostics,
    }


def _append_question_to_lcm(
    mf: MemoryForge,
    agent_id: str,
    session_id: str,
    *,
    question: str,
) -> str:
    content = "\n".join(
        [
            "MemoryForge benchmark question.",
            f"Question: {question}",
        ]
    )
    return mf.lcm_store.append_text_message(agent_id, session_id, "user", content)


def _case_already_ingested(
    mf: MemoryForge,
    adapter: LoComoAdapter,
    case: BenchmarkCase,
    *,
    agent_id: str,
) -> bool:
    sessions = adapter.prepare_sessions(case)
    if not sessions:
        return False
    session_ids = [str(session.get("session_id") or adapter.ingestion_key(case)) for session in sessions]
    return all(mf.conversations.get_session(agent_id, session_id) for session_id in session_ids)


def _locomo_source_text(adapter: LoComoAdapter, case: BenchmarkCase) -> str:
    sessions = adapter.prepare_sessions(case)
    if not sessions:
        return case.question
    sections: list[str] = [
        f"Question ID: {case.case_id}",
        f"Question: {case.question}",
        f"Expected answer: {case.answer or ''}",
        "This document contains the LoCoMo conversation history only.",
        "",
    ]
    for session in sessions:
        session_id = str(session.get("session_id") or adapter.ingestion_key(case))
        sections.append(f"## Session {session_id}")
        for turn in session.get("turns") or []:
            if not isinstance(turn, dict):
                continue
            sections.append(str(turn.get("content") or ""))
        sections.append("")
    return "\n".join(sections).strip()


def _retrieval_diagnostics(
    mf: MemoryForge,
    agent_id: str,
    question: str,
    *,
    top_k: int,
) -> dict[str, Any]:
    ltm_hits = mf.recall_long_term(agent_id, question, top_k=top_k, include_content=True)
    rlm_hits = mf.rlm_search(agent_id, question, limit=top_k)
    streams = [set(hit.get("streams", {})) for hit in ltm_hits]
    return {
        "ltm_count": len(ltm_hits),
        "ltm_bm25_count": sum("bm25" in stream for stream in streams),
        "ltm_vector_count": sum("vector" in stream for stream in streams),
        "rlm_count": len(rlm_hits),
        "ltm_top_refs": [hit.get("raw_ref") for hit in ltm_hits[:3]],
        "rlm_top_refs": [hit.get("ref") for hit in rlm_hits[:3]],
    }


def _answer_with_core_answer_runner(
    operator: SubAgentOperator,
    *,
    case_id: str,
    question: str,
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    context_messages = _model_messages_from_payload(model_payload)
    task = SubAgentTask(
        kind="benchmark.locomo.core_answer",
        system_prompt=(
            "Answer the benchmark question using only the provided MemoryForge LCM context. "
            "Return a concise final answer phrase first. If unsupported, say "
            '"I don\'t know". Output only the final answer phrase; '
            "do not include reasoning, markdown, source refs, or <think> tags."
        ),
        user_prompt=json.dumps(
            {
                "case_id": case_id,
                "question": question,
                "lcm_context_messages": context_messages,
            },
            ensure_ascii=False,
        ),
        max_tokens=128,
        temperature=0,
        metadata={"case_id": case_id, "benchmark": "locomo", "pipeline": "rlm-ltm-lcm-core-answer"},
    )
    response = operator.execute(task)
    if response.provider == "mock":
        raise RuntimeError("Refusing mock core answer runner result for LoCoMo benchmark")
    answer = _clean_core_answer(response.text)
    return {
        "answer": answer,
        "raw_answer": response.text,
        "provider": response.provider,
        "model": response.model,
        "elapsed_seconds": response.elapsed_seconds,
        "input_hash": response.input_hash,
        "cached": response.cached,
    }


def _model_messages_from_payload(model_payload: dict[str, Any]) -> list[dict[str, str]]:
    forbidden = MODEL_PROMPT_AUDIT_FIELDS & set(model_payload)
    if forbidden:
        fields = ", ".join(sorted(forbidden))
        raise ValueError(f"Core answer model payload contains non-renderable audit fields: {fields}")
    messages = model_payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Core answer model payload must contain a messages list")
    rendered: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Core answer model messages must be objects")
        rendered.append(
            {
                "role": str(message.get("role") or ""),
                "source": str(message.get("source") or ""),
                "source_id": str(message.get("source_id") or ""),
                "content": str(message.get("content") or ""),
            }
        )
    return rendered


def _clean_core_answer(text: str) -> str:
    cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", text or "").strip()
    return cleaned or text.strip()


def _answer_matches(expected: str | None, prediction: str) -> bool | None:
    if expected is None:
        return None
    if expected.lower() in prediction.lower():
        return True
    normalized_expected = _normalize_answer(expected)
    normalized_prediction = _normalize_answer(prediction)
    return bool(normalized_expected and normalized_expected in normalized_prediction)


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", lowered)
    tokens = re.findall(r"[a-z0-9]+", lowered)
    return " ".join(token for token in tokens if token not in {"a", "an", "the"})


def _question_session_id(ingestion_key: str, case_id: str) -> str:
    return f"locomo_question_{_safe_id(ingestion_key)}_{_safe_id(case_id)}"


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _mode_contract(mode: str) -> dict[str, Any]:
    try:
        contract = dict(LOCOMO_MODE_CONTRACTS[mode])
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark mode: {mode}") from exc
    contract.update(
        {
            "mode": mode,
            "answer_model_used": bool(contract["uses_core_answer_runner"]),
        }
    )
    return contract


def _db_counts(db_path: str) -> dict[str, int]:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        counts: dict[str, int] = {}
        for table in (
            "sessions",
            "messages",
            "context_items",
            "summary_nodes",
            "rlm_buffers",
            "rlm_chunks",
            "long_term_items",
            "vec_index",
            "search_fts",
        ):
            try:
                counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.DatabaseError:
                counts[table] = -1
        return counts
    finally:
        conn.close()


def _performance_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_latencies = [float(result.get("latency_ms") or 0.0) for result in results]
    query_latencies: list[float] = []
    context_build_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    injection_latencies: list[float] = []
    answer_latencies: list[float] = []
    setup_latencies: list[float] = []
    token_estimates: list[float] = []
    raw_ref_counts: list[float] = []
    long_term_counts: list[float] = []

    for result in results:
        diagnostics = result.get("diagnostics") or {}
        context = diagnostics.get("context") or {}
        context_bundle = diagnostics.get("context_bundle") or {}
        bundle_latency = context_bundle.get("latency_ms") or {}
        if bundle_latency:
            query_total = float(bundle_latency.get("total") or 0.0)
            query_latencies.append(query_total)
            context_build_latencies.append(float(bundle_latency.get("context_build") or 0.0))
            retrieval_latencies.append(float(bundle_latency.get("long_term_recall") or 0.0))
            injection_latencies.append(float(bundle_latency.get("recall_injection") or 0.0))
        answer_runner = result.get("core_answer_runner") or {}
        answer_ms = float(answer_runner.get("elapsed_seconds") or 0.0) * 1000.0
        if answer_ms > 0:
            answer_latencies.append(answer_ms)
        if bundle_latency:
            setup_latencies.append(
                max(0.0, float(result.get("latency_ms") or 0.0) - query_total - answer_ms)
            )
        else:
            ingestion = diagnostics.get("ingestion") or {}
            if ingestion.get("latency_ms") is not None:
                setup_latencies.append(float(ingestion.get("latency_ms") or 0.0))
        if context.get("token_estimate") is not None:
            token_estimates.append(float(context.get("token_estimate") or 0.0))
        if context.get("raw_ref_count") is not None:
            raw_ref_counts.append(float(context.get("raw_ref_count") or 0.0))
        if context.get("long_term_recall_count") is not None:
            long_term_counts.append(float(context.get("long_term_recall_count") or 0.0))

    answered = [result for result in results if result.get("correct") is not None]
    correct = sum(result.get("correct") is True for result in answered)
    return {
        "total_latency_ms": _metric(total_latencies),
        "memoryforge_query_latency_ms": _metric(query_latencies),
        "context_build_latency_ms": _metric(context_build_latencies),
        "retrieval_latency_ms": _metric(retrieval_latencies),
        "recall_injection_latency_ms": _metric(injection_latencies),
        "answer_latency_ms": _metric(answer_latencies),
        "ingest_or_setup_latency_ms": _metric(setup_latencies),
        "context_tokens": _metric(token_estimates),
        "raw_refs_per_answer": _metric(raw_ref_counts),
        "refs_included_per_answer": _metric(raw_ref_counts),
        "long_term_hits_per_answer": _metric(long_term_counts),
        "exact_score": (correct / len(answered)) if answered else None,
        "semantic_score": None,
        "semantic_score_available": False,
        "true_miss_count": sum(result.get("correct") is False for result in results),
        "targets": {
            "typical_context_2k_4k": bool(token_estimates)
            and _percentile(token_estimates, 0.5) <= 4_000,
            "context_assembly_sub_300ms": bool(query_latencies)
            and _percentile(query_latencies, 0.5) <= 300,
            "answer_latency_separated": bool(answer_latencies) or all(
                result.get("core_answer_runner") is None for result in results
            ),
        },
    }


def _metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())

