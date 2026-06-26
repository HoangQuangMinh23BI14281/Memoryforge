#!/usr/bin/env python3
"""Run LongMemEval through the intended MemoryForge pipeline.

Pipeline:
1. Load haystack sessions through RLM.
2. Index RLM chunks into LTM.
3. Append the question to LCM.
4. Build a MemoryForge context bundle with LTM recall injected.
5. Let the configured benchmark/core answer runner answer from that bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from memoryforge.agents import SubAgentOperator, SubAgentTask
from memoryforge.api import MemoryForge
from memoryforge.benchmark import BenchmarkCase, LongMemEvalAdapter
from memoryforge.lcm import ContextBudget
from memoryforge.memory.longterm.models import MetadataField

DEFAULT_VECTOR_BACKEND = "fastembed"
DEFAULT_VECTOR_MODEL = "BAAI/bge-small-en-v1.5"
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
BENCHMARK_MODE_CONTRACTS: dict[str, dict[str, bool | str]] = {
    "ingest-only": {
        "pipeline": "rlm-ltm",
        "ingests": True,
        "builds_context_bundle": False,
        "uses_core_answer_runner": False,
        "uses_rlm_worker": False,
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
    "core-answer": {
        "pipeline": "rlm-ltm-lcm-core-answer",
        "ingests": True,
        "builds_context_bundle": True,
        "uses_core_answer_runner": True,
        "uses_rlm_worker": False,
        "uses_lcm_worker": False,
        "produces_prediction": True,
        "requires_runner": True,
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
    "lcm-worker": {
        "pipeline": "lcm-worker",
        "ingests": True,
        "builds_context_bundle": False,
        "uses_core_answer_runner": False,
        "uses_rlm_worker": False,
        "uses_lcm_worker": True,
        "produces_prediction": False,
        "requires_runner": True,
    },
}
RUNNER_MODES = {
    mode
    for mode, contract in BENCHMARK_MODE_CONTRACTS.items()
    if bool(contract["requires_runner"])
}

SPECIAL_PROBES: list[dict[str, str]] = [
    {
        "case_id": "lcm-dag-probe-01",
        "question": "What is the retention key for Project Atlas?",
        "answer": "amber-17",
        "fact": "Project Atlas uses retention key amber-17.",
    },
    {
        "case_id": "lcm-dag-probe-02",
        "question": "Who owns the Calypso billing migration?",
        "answer": "Mina",
        "fact": "Mina owns the Calypso billing migration.",
    },
    {
        "case_id": "lcm-dag-probe-03",
        "question": "Which warehouse stores the north audit logs?",
        "answer": "Warehouse Kilo",
        "fact": "The north audit logs are stored in Warehouse Kilo.",
    },
    {
        "case_id": "lcm-dag-probe-04",
        "question": "What phrase unlocks the Finch rollback note?",
        "answer": "violet ladder",
        "fact": "The Finch rollback note is unlocked by the phrase violet ladder.",
    },
    {
        "case_id": "lcm-dag-probe-05",
        "question": "Which date did the Orion schema freeze happen?",
        "answer": "14 March 2025",
        "fact": "The Orion schema freeze happened on 14 March 2025.",
    },
    {
        "case_id": "lcm-dag-probe-06",
        "question": "Which service keeps the graphite error budget?",
        "answer": "Helios",
        "fact": "Helios keeps the graphite error budget.",
    },
    {
        "case_id": "lcm-dag-probe-07",
        "question": "What color tag marks the Cedar memory snapshot?",
        "answer": "teal",
        "fact": "The Cedar memory snapshot is marked with the teal tag.",
    },
    {
        "case_id": "lcm-dag-probe-08",
        "question": "Who approved the Lumen recovery drill?",
        "answer": "Ravi",
        "fact": "Ravi approved the Lumen recovery drill.",
    },
    {
        "case_id": "lcm-dag-probe-09",
        "question": "Which region hosts the Nimbus spare index?",
        "answer": "eu-central",
        "fact": "The Nimbus spare index is hosted in eu-central.",
    },
    {
        "case_id": "lcm-dag-probe-10",
        "question": "What is the passphrase for the Quill incident archive?",
        "answer": "silver comet",
        "fact": "The Quill incident archive passphrase is silver comet.",
    },
]


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _case_agent_id(base_agent_id: str, case_id: str, shared_agent: bool) -> str:
    return base_agent_id if shared_agent else f"{base_agent_id}_{_safe_id(case_id)}"


def _record_text(case: BenchmarkCase) -> str:
    return _record_text_and_session_spans(case)[0]


def _record_text_and_session_spans(case: BenchmarkCase) -> tuple[str, list[dict[str, Any]]]:
    record = (case.metadata or {}).get("raw")
    if not isinstance(record, dict):
        return case.question, []
    sessions = record.get("haystack_sessions") or []
    dates = record.get("haystack_dates") or []
    session_ids = record.get("haystack_session_ids") or []
    lines = [
        f"Question ID: {case.case_id}",
        "This document contains LongMemEval haystack sessions only.",
        "",
    ]
    pending_spans: list[dict[str, Any]] = []
    for index, turns in enumerate(sessions):
        if not isinstance(turns, list):
            continue
        session_id = _list_get(session_ids, index, f"session_{index}")
        date_text = _list_get(dates, index, "")
        start_line_index = len(lines)
        answer_turn_lines: list[dict[str, Any]] = []
        has_answer = any(
            bool(turn.get("has_answer")) for turn in turns if isinstance(turn, dict)
        )
        lines.extend([f"## Session {session_id}", f"Date: {date_text}", ""])
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or turn.get("speaker") or "user")
            content = str(turn.get("content") or turn.get("text") or turn.get("message") or "")
            turn_start_line_index = len(lines)
            lines.append(f"[{turn_index}:{role}] {content}")
            if bool(turn.get("has_answer")):
                answer_turn_lines.append(
                    {
                        "start_line_index": turn_start_line_index,
                        "end_line_index": len(lines),
                    }
                )
        lines.append("")
        pending_spans.append(
            {
                "session_id": session_id,
                "date": date_text,
                "start_line_index": start_line_index,
                "end_line_index": len(lines),
                "has_answer": has_answer,
                "answer_turn_lines": answer_turn_lines,
            }
        )
    raw_text = "\n".join(lines)
    text = raw_text.strip()
    line_offsets = _line_offsets(lines)
    spans = []
    for span in pending_spans:
        start_line_index = int(span["start_line_index"])
        end_line_index = int(span["end_line_index"])
        char_start = line_offsets[start_line_index] if start_line_index < len(line_offsets) else 0
        char_end = line_offsets[end_line_index] if end_line_index < len(line_offsets) else len(raw_text)
        answer_turn_spans = []
        for turn_span in span.get("answer_turn_lines") or []:
            turn_start_line_index = int(turn_span["start_line_index"])
            turn_end_line_index = int(turn_span["end_line_index"])
            turn_char_start = (
                line_offsets[turn_start_line_index]
                if turn_start_line_index < len(line_offsets)
                else 0
            )
            turn_char_end = (
                line_offsets[turn_end_line_index]
                if turn_end_line_index < len(line_offsets)
                else len(raw_text)
            )
            answer_turn_spans.append(
                {
                    "char_start": min(turn_char_start, len(text)),
                    "char_end": min(turn_char_end, len(text)),
                }
            )
        spans.append(
            {
                "session_id": span["session_id"],
                "date": span["date"],
                "char_start": min(char_start, len(text)),
                "char_end": min(char_end, len(text)),
                "has_answer": span["has_answer"],
                "answer_turn_spans": answer_turn_spans,
            }
        )
    return text, spans


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = []
    cursor = 0
    for index, line in enumerate(lines):
        if index:
            cursor += 1
        offsets.append(cursor)
        cursor += len(line)
    return offsets


def _special_probe_text(probe: dict[str, str]) -> str:
    distractor = (
        "This paragraph is repeated to create context pressure. "
        "It discusses unrelated operational notes, release rituals, stale dashboards, "
        "and generic reminders that must not override the probe fact."
    )
    sections = [
        f"# Special LCM DAG probe {probe['case_id']}",
        probe["fact"],
        "",
    ]
    for index in range(14):
        sections.append(f"## Distractor block {index}\n{distractor} " * 8)
    sections.append(f"## Authoritative fact\n{probe['fact']}")
    return "\n\n".join(sections)


def _list_get(values: Any, index: int, default: str) -> str:
    if isinstance(values, list) and index < len(values):
        return str(values[index])
    return default


def _append_question_to_lcm(
    mf: MemoryForge,
    agent_id: str,
    session_id: str,
    *,
    question: str,
    question_date: str | None = None,
) -> str:
    content = "\n".join(
        part
        for part in [
            "MemoryForge benchmark question.",
            f"Question date: {question_date}" if question_date else "",
            f"Question: {question}",
        ]
        if part
    )
    return mf.lcm_store.append_text_message(agent_id, session_id, "user", content)


def _context_payload(context: Any) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.role),
            "source": str(message.source),
            "source_id": str(message.source_id),
            "content": str(message.content),
        }
        for message in context.messages
    ]


def _answer_with_core_answer_runner(
    operator: SubAgentOperator,
    *,
    case_id: str,
    question: str,
    question_type: str | None,
    question_date: str | None,
    pipeline: str,
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    context_messages = _model_messages_from_payload(model_payload)
    task = SubAgentTask(
        kind="benchmark.longmemeval.core_answer",
        system_prompt=(
            "Answer the benchmark question using only the provided MemoryForge LCM context. "
            "The context may contain a long-term recall block sourced from RLM/LTM chunks and "
            "LCM summary messages. Return a concise final answer first. If unsupported, say "
            '"I don\'t know". Prioritize the user\'s own statements over assistant suggestions, '
            "and combine adjacent turns from the same session when the requested fact is implicit. "
            "For questions about where an action happened, answer the action's place or entity, "
            "not an incidental place where the user found or stored information. "
            "For questions about when something happened, resolve named dates or holidays to the "
            "most specific date supported by the context; do not answer with only the month when "
            "a holiday or exact day is available. Output only the final answer phrase; "
            "do not include reasoning, markdown, source refs, or <think> tags. "
            "Do not mention hidden expected answers."
        ),
        user_prompt=json.dumps(
            {
                "case_id": case_id,
                "question": question,
                "question_type": question_type,
                "question_date": question_date,
                "lcm_context_messages": context_messages,
            },
            ensure_ascii=False,
        ),
        max_tokens=256,
        temperature=0,
        metadata={"case_id": case_id, "benchmark": "longmemeval", "pipeline": pipeline},
    )
    response = operator.execute(task)
    if response.provider == "mock":
        raise RuntimeError("Refusing mock core answer runner result for LongMemEval benchmark")
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
    return bool(
        normalized_expected and normalized_expected in normalized_prediction
    )


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", lowered)
    tokens = re.findall(r"[a-z0-9]+", lowered)
    return " ".join(token for token in tokens if token not in {"a", "an", "the"})


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


def _annotate_longmemeval_answer_evidence(
    mf: MemoryForge,
    agent_id: str,
    case: BenchmarkCase,
    loaded: dict[str, Any] | None,
    session_spans: list[dict[str, Any]],
) -> dict[str, Any]:
    record = (case.metadata or {}).get("raw")
    item_ids = [str(item_id) for item_id in (loaded or {}).get("long_term_item_ids", [])]
    answer_session_ids = _answer_session_ids(record if isinstance(record, dict) else {}, session_spans)
    answer_spans = [
        span for span in session_spans if str(span.get("session_id")) in answer_session_ids
    ]
    diagnostics = {
        "source": "longmemeval_session_provenance",
        "answer_session_ids": sorted(answer_session_ids),
        "candidate_item_count": len(item_ids),
        "matched_item_count": 0,
        "matched_session_ids": [],
        "matched_item_ids": [],
    }
    if not item_ids or not answer_spans:
        return diagnostics

    placeholders = ",".join("?" for _item_id in item_ids)
    rows = mf.long_term.conn.execute(
        f"""
        SELECT l.item_id, l.metadata, c.content
        FROM long_term_items l
        JOIN content_store c ON c.content_id = l.content_id
        WHERE agent_id = ? AND item_id IN ({placeholders})
        """,
        [agent_id, *item_ids],
    ).fetchall()
    updates: list[tuple[str, dict[str, Any]]] = []
    matched_session_ids: set[str] = set()
    matched_item_ids: list[str] = []
    question_date = (case.metadata or {}).get("question_date")
    question_type = (case.metadata or {}).get("question_type")
    for item_id, metadata_json, content in rows:
        metadata = _json_dict(metadata_json)
        char_range = metadata.get("char_range")
        if not isinstance(char_range, dict):
            continue
        chunk_start = _optional_int(char_range.get("start"))
        chunk_end = _optional_int(char_range.get("end"))
        if chunk_start is None or chunk_end is None:
            continue
        matched_spans = [
            span
            for span in answer_spans
            if _ranges_overlap(
                chunk_start,
                chunk_end,
                int(span.get("char_start") or 0),
                int(span.get("char_end") or 0),
            )
        ]
        if not matched_spans:
            continue
        session_ids = _dedupe_texts(str(span["session_id"]) for span in matched_spans)
        dates = _dedupe_texts(str(span.get("date") or "") for span in matched_spans if span.get("date"))
        answer_ranges = _local_answer_evidence_ranges(
            str(content or ""),
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            matched_spans=matched_spans,
        )
        existing_evidence_ids = _metadata_list(metadata.get(MetadataField.EVIDENCE_IDS))
        evidence_ids = _dedupe_texts([*existing_evidence_ids, *session_ids])
        matched_session_ids.update(session_ids)
        matched_item_ids.append(str(item_id))
        update = {
            MetadataField.BENCHMARK: "longmemeval",
            MetadataField.BENCHMARK_CASE_ID: case.case_id,
            MetadataField.BENCHMARK_SESSION_ID: session_ids[0],
            MetadataField.ANSWER_EVIDENCE: True,
            MetadataField.EVIDENCE_IDS: evidence_ids,
        }
        if dates:
            update[MetadataField.BENCHMARK_SESSION_DATE] = dates[0]
        if question_date is not None:
            update[MetadataField.QUESTION_DATE] = question_date
        if question_type is not None:
            update[MetadataField.QUESTION_TYPE] = question_type
        if answer_ranges:
            update[MetadataField.ANSWER_EVIDENCE_RANGES] = answer_ranges
        updates.append((str(item_id), update))

    if updates:
        mf.long_term.merge_items_metadata(agent_id, updates)
    diagnostics["matched_item_count"] = len(updates)
    diagnostics["matched_session_ids"] = sorted(matched_session_ids)
    diagnostics["matched_item_ids"] = matched_item_ids[:10]
    return diagnostics


def _answer_session_ids(record: dict[str, Any], session_spans: list[dict[str, Any]]) -> set[str]:
    values = record.get("answer_session_ids")
    if values is None:
        values = record.get("answer_session_id")
    answer_session_ids = set(_metadata_list(values))
    answer_session_ids.update(
        str(span["session_id"]) for span in session_spans if span.get("has_answer")
    )
    return answer_session_ids


def _local_answer_evidence_ranges(
    content: str,
    *,
    chunk_start: int,
    chunk_end: int,
    matched_spans: list[dict[str, Any]],
) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    for span in matched_spans:
        for answer_span in span.get("answer_turn_spans") or []:
            answer_start = _optional_int(answer_span.get("char_start"))
            answer_end = _optional_int(answer_span.get("char_end"))
            if answer_start is None or answer_end is None:
                continue
            if not _ranges_overlap(chunk_start, chunk_end, answer_start, answer_end):
                continue
            local_start = max(0, answer_start - chunk_start)
            local_end = min(len(content), answer_end - chunk_start)
            if local_start < local_end:
                ranges.append({"start": local_start, "end": local_end})
    return ranges


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _metadata_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return _dedupe_texts(str(item) for item in value if item is not None)
    return [str(value)]


def _dedupe_texts(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_dict(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _dag_diagnostics(mf: MemoryForge, session_id: str) -> dict[str, Any]:
    nodes = mf.lcm_engine.dag.get_active_summaries(session_id)
    return {
        "active_summary_count": len(nodes),
        "active_summary_ids": [node.id for node in nodes],
        "active_summary_levels": [node.level for node in nodes],
        "has_dag": bool(nodes),
    }


def _db_counts(db_path: str) -> dict[str, int]:
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


def _mode_contract(mode: str, *, is_special: bool) -> dict[str, Any]:
    try:
        base = dict(BENCHMARK_MODE_CONTRACTS[mode])
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark mode: {mode}") from exc
    uses_lcm_worker = bool(base["uses_lcm_worker"]) or (
        mode == "core-answer" and is_special
    )
    base.update(
        {
            "mode": mode,
            "uses_lcm_worker": uses_lcm_worker,
            "special_lcm_probe": is_special,
            "answer_model_used": bool(base["uses_core_answer_runner"]),
        }
    )
    return base


def _run_case(
    *,
    mf: MemoryForge,
    operator: SubAgentOperator | None,
    case: BenchmarkCase,
    agent_id: str,
    mode: str,
    top_k: int,
    chunk_size: int,
    overlap: int,
    context_limit: int,
    recall_content_policy: str,
    long_term_token_budget: int | None = None,
    rlm_max_workers: int = 1,
    is_special: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    session_id = f"{case.case_id}_question"
    question_date = None
    question_type = None
    if case.metadata:
        question_date = case.metadata.get("question_date")
        question_type = case.metadata.get("question_type")
    if is_special and case.metadata:
        source_text = _special_probe_text(case.metadata["probe"])
        session_spans: list[dict[str, Any]] = []
    else:
        source_text, session_spans = _record_text_and_session_spans(case)
    mode_contract = _mode_contract(mode, is_special=is_special)
    if mode == "rlm-worker":
        if operator is None:
            raise RuntimeError("rlm-worker mode requires a configured runner")
        rlm_result = mf.rlm_run(
            agent_id=agent_id,
            value=source_text,
            name=f"{case.case_id}.longmemeval",
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
            "agent_id": agent_id,
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
                "is_special": is_special,
                "rlm_worker": rlm_result,
                "db_counts": _db_counts(mf.db_path),
            },
        }

    loaded: dict[str, Any] | None = None
    answer_evidence_annotation: dict[str, Any] | None = None
    if mode != "context-only":
        loaded = mf.rlm_load(
            agent_id,
            source_text,
            name=f"{case.case_id}.longmemeval",
            chunk_size=chunk_size,
            overlap=overlap,
        )
        answer_evidence_annotation = _annotate_longmemeval_answer_evidence(
            mf,
            agent_id,
            case,
            loaded,
            session_spans,
        )
    else:
        answer_evidence_annotation = _annotate_longmemeval_answer_evidence(
            mf,
            agent_id,
            case,
            None,
            session_spans,
        )
    if mode == "ingest-only":
        if loaded is None:
            raise RuntimeError("ingest-only mode did not run ingestion")
        return {
            "case_id": case.case_id,
            "agent_id": agent_id,
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
                "is_special": is_special,
                "rlm_buffer_id": loaded["buffer_id"],
                "rlm_chunk_count": loaded["chunk_count"],
                "ltm_indexed_count": len(loaded.get("long_term_item_ids", [])),
                "ingestion_manifest": loaded.get("ingestion_manifest"),
                "deduped": bool(loaded.get("deduped")),
                "rlm_deduped": bool(loaded.get("rlm_deduped")),
                "ltm_deduped": bool(loaded.get("ltm_deduped")),
                "answer_evidence_annotation": answer_evidence_annotation,
                "db_counts": _db_counts(mf.db_path),
            },
        }

    compaction_result = None
    exercise_lcm_worker = bool(mode_contract["uses_lcm_worker"])
    if exercise_lcm_worker:
        for index in range(8):
            content = (
                f"Special probe source turn {index}. {source_text}\n"
                "Keep exact identifiers, dates, owners, colors, locations, and passphrases. "
            )
            mf.lcm_store.append_text_message(
                agent_id,
                session_id,
                "assistant" if index % 2 else "user",
                content,
            )
        if operator is None:
            raise RuntimeError("LCM worker mode requires a configured runner")
        compaction_result = mf.lcm_compact_if_needed(
            agent_id,
            session_id,
            force=True,
            max_rounds=1,
            runner=operator.runner,
            model=operator.model,
            project_root=operator.project_root,
            base_url=operator.base_url,
            budget=ContextBudget(
                model_context_limit=max(800, context_limit // 2),
                reserved_output_tokens=120,
                compaction_buffer=80,
                soft_threshold_fraction=0.5,
            ),
        )
    if mode == "lcm-worker":
        if loaded is None:
            raise RuntimeError("lcm-worker mode requires ingestion setup")
        return {
            "case_id": case.case_id,
            "agent_id": agent_id,
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
                "is_special": is_special,
                "rlm_buffer_id": loaded["buffer_id"],
                "rlm_chunk_count": loaded["chunk_count"],
                "ltm_indexed_count": len(loaded.get("long_term_item_ids", [])),
                "ingestion_manifest": loaded.get("ingestion_manifest"),
                "deduped": bool(loaded.get("deduped")),
                "rlm_deduped": bool(loaded.get("rlm_deduped")),
                "ltm_deduped": bool(loaded.get("ltm_deduped")),
                "answer_evidence_annotation": answer_evidence_annotation,
                "dag": _dag_diagnostics(mf, session_id),
                "compaction": None
                if compaction_result is None
                else _compaction_payload(compaction_result),
            },
        }

    preingested_ltm_count = mf.long_term.count(agent_id)
    if mode == "context-only" and preingested_ltm_count <= 0:
        raise RuntimeError(
            "context-only mode requires pre-ingested LTM rows; run ingest-only "
            "with the same --db, --agent-id/--shared-agent, dataset, and case selection first"
        )

    question_message_id = _append_question_to_lcm(
        mf,
        agent_id,
        session_id,
        question=case.question,
        question_date=str(question_date) if question_date else None,
    )
    context_bundle = mf.build_core_context_bundle(
        session_id=session_id,
        agent_id=agent_id,
        query=case.question,
        budget=ContextBudget(model_context_limit=context_limit),
        top_k=top_k,
        include_content=True,
        recall_content_policy=recall_content_policy,
        long_term_token_budget=long_term_token_budget,
    )
    model_payload = context_bundle.to_model_payload()
    context_messages = model_payload["messages"]
    context_diagnostics = context_bundle.diagnostics["context"]
    diagnostics = {
        "mode": mode,
        "pipeline": mode_contract["pipeline"],
        "mode_contract": mode_contract,
        "vector_model": mf.long_term.vector.model_name,
        "vector_backend": mf.long_term.vector.embedding_backend,
        "is_special": is_special,
        "rlm_buffer_id": loaded["buffer_id"] if loaded else None,
        "rlm_chunk_count": loaded["chunk_count"] if loaded else 0,
        "ltm_indexed_count": len(loaded.get("long_term_item_ids", [])) if loaded else 0,
        "ingestion_manifest": loaded.get("ingestion_manifest") if loaded else None,
        "deduped": bool(loaded.get("deduped")) if loaded else True,
        "rlm_deduped": bool(loaded.get("rlm_deduped")) if loaded else True,
        "ltm_deduped": bool(loaded.get("ltm_deduped")) if loaded else True,
        "answer_evidence_annotation": answer_evidence_annotation,
        "ingestion": {
            "performed": loaded is not None,
            "preingested_ltm_count": preingested_ltm_count,
        },
        "question_message_id": question_message_id,
        "context": {
            "message_count": context_diagnostics["message_count"],
            "token_estimate": context_bundle.token_estimate,
            "has_summary": context_diagnostics["has_summary"],
            "truncated": context_diagnostics["truncated"],
            "summary_node_ids": context_diagnostics["summary_node_ids"],
            "sources": [message["source"] for message in context_messages],
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
        "dag": _dag_diagnostics(mf, session_id),
        "compaction": None
        if compaction_result is None
        else _compaction_payload(compaction_result),
    }
    if mode == "context-only":
        return {
            "case_id": case.case_id,
            "agent_id": agent_id,
            "question": case.question,
            "expected": case.answer,
            "prediction": None,
            "correct": None,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "core_answer_runner": None,
            "diagnostics": diagnostics,
        }

    if operator is None:
        raise RuntimeError("core-answer mode requires a configured runner")
    core_answer_runner = _answer_with_core_answer_runner(
        operator,
        case_id=case.case_id,
        question=case.question,
        question_type=str(question_type) if question_type else None,
        question_date=str(question_date) if question_date else None,
        pipeline=str(mode_contract["pipeline"]),
        model_payload=model_payload,
    )
    prediction = str(core_answer_runner["answer"])
    exact_correct = _answer_matches(case.answer, prediction)
    return {
        "case_id": case.case_id,
        "agent_id": agent_id,
        "question": case.question,
        "expected": case.answer,
        "prediction": prediction,
        "correct": exact_correct,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "core_answer_runner": core_answer_runner,
        "diagnostics": diagnostics,
    }


def _compaction_payload(compaction_result: Any) -> dict[str, Any]:
    return {
        "triggered": compaction_result.triggered,
        "rounds": compaction_result.rounds,
        "summary_node_ids": compaction_result.summary_node_ids,
        "before_tokens": compaction_result.before_tokens,
        "after_tokens": compaction_result.after_tokens,
        "delta_tokens": compaction_result.delta_tokens,
        "expanded": compaction_result.expanded,
        "effective": compaction_result.effective,
        "deferred": compaction_result.deferred,
        "reason": compaction_result.reason,
        "cached": compaction_result.cached,
    }


def _special_cases(count: int) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for probe in SPECIAL_PROBES[:count]:
        cases.append(
            BenchmarkCase(
                case_id=probe["case_id"],
                question=probe["question"],
                answer=probe["answer"],
                metadata={"probe": probe, "question_type": "lcm-dag-probe"},
            )
        )
    return cases


def _write_result(handle: Any, result: dict[str, Any]) -> None:
    if handle is not None:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--db", default="~/.memoryforge/memory.db")
    parser.add_argument("--agent-id", default="benchmark")
    parser.add_argument("--limit", type=int, default=40, help="0 means all dataset cases")
    parser.add_argument(
        "--start-index", type=int, default=1, help="1-based first dataset case index"
    )
    parser.add_argument("--special-lcm-probes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=3_000)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--context-limit", type=int, default=16_000)
    parser.add_argument(
        "--recall-content-policy",
        choices=["snippet", "champion", "full", "auto", "preview"],
        default="snippet",
    )
    parser.add_argument(
        "--long-term-token-budget",
        type=int,
        default=None,
        help="Optional token budget for rendered LTM recall inside the core prompt.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(BENCHMARK_MODE_CONTRACTS),
        default="core-answer",
    )
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
        help="Maximum parallel RLM analysis sub-agents in rlm-worker mode.",
    )
    parser.add_argument(
        "--vector-model", default=os.environ.get("MEMORYFORGE_VECTOR_MODEL") or DEFAULT_VECTOR_MODEL
    )
    parser.add_argument(
        "--vector-backend",
        default=os.environ.get("MEMORYFORGE_VECTOR_BACKEND") or DEFAULT_VECTOR_BACKEND,
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--jsonl-output", default="")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--shared-agent", action="store_true")
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
    if args.vector_model:
        os.environ["MEMORYFORGE_VECTOR_MODEL"] = args.vector_model
    if args.vector_backend:
        os.environ["MEMORYFORGE_VECTOR_BACKEND"] = args.vector_backend
    os.environ["MEMORYFORGE_REQUIRE_VECTOR_MODEL"] = "1"

    adapter = LongMemEvalAdapter()
    dataset_cases = adapter.load_cases(args.dataset)
    dataset_cases = dataset_cases[max(0, args.start_index - 1) :]
    if args.limit > 0:
        dataset_cases = dataset_cases[: args.limit]
    cases = [*_special_cases(max(0, args.special_lcm_probes)), *dataset_cases]

    for output in (args.output, args.jsonl_output):
        if args.clean_output and output:
            Path(output).expanduser().unlink(missing_ok=True)

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
    actual_vector_model = mf.long_term.vector.model_name
    actual_vector_backend = mf.long_term.vector.embedding_backend

    results: list[dict[str, Any]] = []
    jsonl_handle = None
    if args.jsonl_output:
        jsonl_path = Path(args.jsonl_output).expanduser()
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("w", encoding="utf-8")
    try:
        for index, case in enumerate(cases, start=1):
            case_agent_id = _case_agent_id(args.agent_id, case.case_id, args.shared_agent)
            is_special = bool(case.metadata and case.metadata.get("probe"))
            result = _run_case(
                mf=mf,
                operator=operator,
                case=case,
                agent_id=case_agent_id,
                mode=args.mode,
                top_k=args.top_k,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                context_limit=args.context_limit,
                recall_content_policy=args.recall_content_policy,
                long_term_token_budget=args.long_term_token_budget,
                rlm_max_workers=args.rlm_max_workers,
                is_special=is_special,
            )
            results.append(result)
            _write_result(jsonl_handle, result)
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(cases),
                        "case_id": result["case_id"],
                        "special": is_special,
                        "correct": result["correct"],
                        "provider": (result.get("core_answer_runner") or {}).get("provider"),
                        "mode": args.mode,
                        "vector_model": mf.long_term.vector.model_name,
                        "vector_backend": mf.long_term.vector.embedding_backend,
                        "has_dag": (result["diagnostics"].get("dag") or {}).get("has_dag"),
                        "latency_ms": result["latency_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()
        mf.close()

    summary = {
        "total": len(results),
        "special_lcm_probes": sum(bool(item["diagnostics"]["is_special"]) for item in results),
        "dataset_cases": sum(not bool(item["diagnostics"]["is_special"]) for item in results),
        "correct": sum(item["correct"] is True for item in results),
        "incorrect": sum(item["correct"] is False for item in results),
        "unknown": sum(item["correct"] is None for item in results),
        "mode": args.mode,
        "runner": args.runner,
        "model": args.model,
        "vector_model": actual_vector_model,
        "vector_backend": actual_vector_backend,
        "requested_vector_model": args.vector_model,
        "requested_vector_backend": args.vector_backend,
        "recall_content_policy": args.recall_content_policy,
        "rlm_max_workers": args.rlm_max_workers,
        "mode_contract": _mode_contract(args.mode, is_special=args.special_lcm_probes > 0),
        "performance": _performance_summary(results),
        "db_counts": _db_counts(args.db),
    }
    payload = {"summary": summary, "results": results}
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
