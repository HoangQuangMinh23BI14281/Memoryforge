#!/usr/bin/env python3
"""Run LoCoMo adapter smoke benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from memoryforge.api import MemoryForge
from memoryforge.benchmark import LoComoAdapter
from memoryforge.benchmark.adapter import BenchmarkCase
from memoryforge.lcm import ContextBudget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--db", default="~/.memoryforge/memory.db")
    parser.add_argument("--agent-id", default="benchmark")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["adapter-search", "context-only", "ingest-only"],
        default="adapter-search",
    )
    parser.add_argument("--context-limit", type=int, default=16_000)
    parser.add_argument(
        "--recall-content-policy",
        choices=["snippet", "champion", "full", "auto", "preview"],
        default="snippet",
    )
    parser.add_argument("--long-term-token-budget", type=int, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--jsonl-output", default="")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--vector-backend", default="")
    parser.add_argument("--vector-model", default="")
    parser.add_argument("--require-vector-model", action="store_true")
    args = parser.parse_args()

    if args.vector_backend:
        os.environ["MEMORYFORGE_VECTOR_BACKEND"] = args.vector_backend
    if args.vector_model:
        os.environ["MEMORYFORGE_VECTOR_MODEL"] = args.vector_model
    if args.require_vector_model:
        os.environ["MEMORYFORGE_REQUIRE_VECTOR_MODEL"] = "1"

    for output in (args.output, args.jsonl_output):
        if args.clean_output and output:
            Path(output).expanduser().unlink(missing_ok=True)

    adapter = LoComoAdapter()
    cases = adapter.load_cases(args.dataset)[: args.limit]
    mf = MemoryForge(args.db)
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
            key = adapter.ingestion_key(case)
            ingestion_started = time.perf_counter()
            message_ids: list[str] = []
            ingested_new = False
            deduped = key in ingested_keys or _case_already_ingested(
                mf,
                adapter,
                case,
                agent_id=args.agent_id,
            )
            if args.mode == "context-only" and not deduped:
                raise RuntimeError(
                    "context-only mode requires pre-ingested LoCoMo sessions; run "
                    "ingest-only with the same --db, --agent-id, dataset, and case selection first"
                )
            if args.mode != "context-only" and not deduped:
                message_ids = adapter.ingest_case(mf, case, agent_id=args.agent_id)
                ingested_new = True
            ingested_keys.add(key)
            ingestion_ms = (time.perf_counter() - ingestion_started) * 1000.0
            if args.mode == "ingest-only":
                result = {
                    "case_id": case.case_id,
                    "question": case.question,
                    "expected": case.answer,
                    "prediction": None,
                    "correct": None,
                    "latency_ms": ingestion_ms,
                    "core_answer_runner": None,
                    "diagnostics": {
                        "mode": args.mode,
                        "mode_contract": _mode_contract(args.mode),
                        "ingestion": {
                            "performed": ingested_new,
                            "deduped": deduped,
                            "message_count": len(message_ids),
                            "latency_ms": ingestion_ms,
                        },
                    },
                }
            elif args.mode == "context-only":
                result = _run_context_only_case(
                    mf=mf,
                    adapter=adapter,
                    case=case,
                    agent_id=args.agent_id,
                    top_k=args.top_k,
                    context_limit=args.context_limit,
                    recall_content_policy=args.recall_content_policy,
                    long_term_token_budget=args.long_term_token_budget,
                )
            else:
                result = adapter.run_case(
                    mf,
                    case,
                    agent_id=args.agent_id,
                    top_k=args.top_k,
                ).__dict__
                result["diagnostics"] = {
                    "mode": args.mode,
                    "mode_contract": _mode_contract(args.mode),
                    "ingestion": {
                        "performed": ingested_new,
                        "deduped": deduped,
                        "message_count": len(message_ids),
                        "latency_ms": ingestion_ms,
                    },
                }
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
        "context_limit": args.context_limit if args.mode == "context-only" else None,
        "recall_content_policy": args.recall_content_policy
        if args.mode == "context-only"
        else None,
        "ingested_sessions": len(ingested_keys),
        "vector_model": actual_vector_model,
        "vector_backend": actual_vector_backend,
        "requested_vector_model": args.vector_model or None,
        "requested_vector_backend": args.vector_backend or None,
        "performance": _performance_summary(results),
    }
    payload = {"summary": summary, "results": results}
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


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


def _question_session_id(ingestion_key: str, case_id: str) -> str:
    return f"locomo_question_{_safe_id(ingestion_key)}_{_safe_id(case_id)}"


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _mode_contract(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "ingests": mode in {"adapter-search", "ingest-only"},
        "builds_context_bundle": mode == "context-only",
        "uses_core_answer_runner": False,
        "uses_rlm_worker": False,
        "uses_lcm_worker": False,
        "produces_prediction": mode == "adapter-search",
        "answer_model_used": False,
    }


def _performance_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_latencies = [float(result.get("latency_ms") or 0.0) for result in results]
    setup_latencies: list[float] = []
    query_latencies: list[float] = []
    context_build_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    injection_latencies: list[float] = []
    answer_latencies: list[float] = []
    token_estimates: list[float] = []
    raw_ref_counts: list[float] = []
    long_term_counts: list[float] = []
    correct_values = [result.get("correct") for result in results if result.get("correct") is not None]
    for result in results:
        diagnostics = result.get("diagnostics") or {}
        ingestion = diagnostics.get("ingestion") or {}
        if ingestion.get("latency_ms") is not None:
            setup_latencies.append(float(ingestion.get("latency_ms") or 0.0))
        context = diagnostics.get("context") or {}
        if context.get("token_estimate") is not None:
            token_estimates.append(float(context.get("token_estimate") or 0.0))
        if context.get("raw_ref_count") is not None:
            raw_ref_counts.append(float(context.get("raw_ref_count") or 0.0))
        if context.get("long_term_recall_count") is not None:
            long_term_counts.append(float(context.get("long_term_recall_count") or 0.0))
        context_bundle = diagnostics.get("context_bundle") or {}
        bundle_latency = context_bundle.get("latency_ms") or {}
        if bundle_latency:
            query_latencies.append(float(bundle_latency.get("total") or 0.0))
            context_build_latencies.append(float(bundle_latency.get("context_build") or 0.0))
            retrieval_latencies.append(float(bundle_latency.get("long_term_recall") or 0.0))
            injection_latencies.append(float(bundle_latency.get("recall_injection") or 0.0))
    exact_score = None
    if correct_values:
        exact_score = sum(value is True for value in correct_values) / len(correct_values)
    return {
        "total_latency_ms": _metric(total_latencies),
        "ingest_or_setup_latency_ms": _metric(setup_latencies),
        "memoryforge_query_latency_ms": _metric(query_latencies),
        "context_build_latency_ms": _metric(context_build_latencies),
        "retrieval_latency_ms": _metric(retrieval_latencies),
        "recall_injection_latency_ms": _metric(injection_latencies),
        "answer_latency_ms": _metric(answer_latencies),
        "context_tokens": _metric(token_estimates),
        "raw_refs_per_answer": _metric(raw_ref_counts),
        "long_term_hits_per_answer": _metric(long_term_counts),
        "exact_score": exact_score,
        "answer_latency_separated": not answer_latencies,
    }


def _metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg": None, "min": None, "max": None}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


if __name__ == "__main__":
    raise SystemExit(main())
