#!/usr/bin/env python3
"""Run a deterministic multi-session MemoryForge stress benchmark."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from memoryforge.api import MemoryForge
from memoryforge.lcm import ContextBudget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="")
    parser.add_argument("--agent-id", default="stress")
    parser.add_argument("--sessions", type=int, default=50)
    parser.add_argument("--turns-per-session", type=int, default=12)
    parser.add_argument("--context-limit", type=int, default=16_000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.sessions < 1:
        raise SystemExit("--sessions must be at least 1")
    if args.turns_per_session < 1:
        raise SystemExit("--turns-per-session must be at least 1")

    if args.db:
        payload = _run(args)
    else:
        with tempfile.TemporaryDirectory() as tempdir:
            args.db = str(Path(tempdir) / "memory.db")
            payload = _run(args)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run(args: argparse.Namespace) -> dict[str, Any]:
    mf = MemoryForge(args.db)
    ingest_latencies: list[float] = []
    query_latencies: list[float] = []
    token_counts: list[float] = []
    raw_ref_counts: list[float] = []
    long_term_counts: list[float] = []
    started = time.perf_counter()
    try:
        doc_result = mf.rlm_load(
            agent_id=args.agent_id,
            value=_project_markdown(args.sessions),
            name="medium-project-plan.md",
            chunk_size=2_500,
            overlap=250,
        )
        for index in range(args.sessions):
            session_id = f"stress-session-{index:04d}"
            turns = _session_turns(index, args.turns_per_session)
            turn_started = time.perf_counter()
            mf.store_conversation(args.agent_id, turns, session_id=session_id)
            ingest_latencies.append((time.perf_counter() - turn_started) * 1000.0)
            if index % 5 == 0 or index == args.sessions - 1:
                query = (
                    f"Summarize current risks for milestone {index % 7} and "
                    f"owner team-{index % 4}."
                )
                query_started = time.perf_counter()
                bundle = mf.build_core_context_bundle(
                    agent_id=args.agent_id,
                    session_id=session_id,
                    query=query,
                    budget=ContextBudget(model_context_limit=args.context_limit),
                    top_k=args.top_k,
                    include_content=True,
                )
                query_latencies.append((time.perf_counter() - query_started) * 1000.0)
                token_counts.append(float(bundle.token_estimate))
                raw_ref_counts.append(float(len(bundle.raw_refs)))
                long_term_counts.append(float(len(bundle.long_term_recall)))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "summary": {
                "sessions": args.sessions,
                "turns_per_session": args.turns_per_session,
                "agent_id": args.agent_id,
                "elapsed_ms": elapsed_ms,
                "rlm_chunks": doc_result.get("chunk_count", 0),
                "db_counts": _db_counts(args.db),
                "performance": {
                    "ingest_latency_ms": _metric(ingest_latencies),
                    "query_latency_ms": _metric(query_latencies),
                    "context_tokens": _metric(token_counts),
                    "raw_refs_per_query": _metric(raw_ref_counts),
                    "long_term_hits_per_query": _metric(long_term_counts),
                },
            }
        }
    finally:
        mf.close()


def _project_markdown(session_count: int) -> str:
    sections = [
        "# MemoryForge Stress Project",
        "",
        "This document describes a medium project with repeated implementation decisions, "
        "risk tracking, benchmark notes, release constraints, and setup details.",
    ]
    for index in range(max(20, session_count)):
        sections.extend(
            [
                "",
                f"## Milestone {index}",
                f"- Owner: team-{index % 4}",
                f"- Main module: service-{index % 6}",
                f"- Decision: keep SQLite-only persistence for artifact-{index % 9}.",
                f"- Risk: benchmark drift in scenario-{index % 7}.",
                f"- Setup note: run uv sync before validating workflow-{index % 5}.",
            ]
        )
    return "\n".join(sections)


def _session_turns(index: int, turns_per_session: int) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for turn_index in range(turns_per_session):
        role = "user" if turn_index % 2 == 0 else "assistant"
        turns.append(
            {
                "role": role,
                "content": (
                    f"Session {index} turn {turn_index}: team-{index % 4} is working on "
                    f"service-{turn_index % 6}. Milestone {index % 7} depends on "
                    f"artifact-{(index + turn_index) % 9}. Keep setup via uv and record "
                    f"latency/cost notes for benchmark scenario-{turn_index % 5}."
                ),
                "metadata": {
                    "kind": "stress",
                    "session_index": index,
                    "turn_index": turn_index,
                    "team": f"team-{index % 4}",
                    "milestone": index % 7,
                },
            }
        )
    return turns


def _db_counts(db_path: str) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        counts: dict[str, int] = {}
        for table in (
            "sessions",
            "messages",
            "message_parts",
            "context_items",
            "summary_nodes",
            "rlm_buffers",
            "rlm_chunks",
            "long_term_items",
            "vec_index",
            "search_fts",
        ):
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = int(row[0] if row else 0)
            except sqlite3.DatabaseError:
                counts[table] = -1
        return counts
    finally:
        conn.close()


def _metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "avg": None, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "avg": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


if __name__ == "__main__":
    raise SystemExit(main())
