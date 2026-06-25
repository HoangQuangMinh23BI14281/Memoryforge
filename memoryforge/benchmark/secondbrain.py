"""Deterministic second-brain behavior benchmark.

This benchmark exercises MemoryForge as memory infrastructure for a core model.
It does not ask a model to answer. Instead it verifies durable evidence records,
active recall, corrections, contradictions, and provenance for the active runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from memoryforge.api import MemoryForge
from memoryforge.lcm import ContextBudget
from memoryforge.memory.longterm.models import MetadataField

SECOND_BRAIN_MODES = {"ingest-only", "context-only"}

INGEST_DIMENSION_CHECKS: dict[str, tuple[str, ...]] = {
    "semantic_ledger_ingestion": ("ingested_raw_evidence",),
    "correction_metadata": ("ingested_correction",),
    "contradiction_metadata": ("ingested_contradiction",),
}

CONTEXT_DIMENSION_CHECKS: dict[str, tuple[str, ...]] = {
    "cross_session_continuity": ("cross_session_continuity",),
    "correction_learning": ("correction_learning",),
    "contradiction_detection": ("contradiction_detection",),
    "lifecycle_active_recall": ("active_recall_lifecycle",),
    "project_restart_quality": ("project_restart_quality",),
    "answer_provenance": ("provenance",),
    "context_budget_efficiency": ("context_budget",),
    "core_model_injected_memory": (
        "active_recall_injection",
        "core_model_boundary",
        "core_model_injected_memory_contract",
    ),
}


@dataclass(frozen=True)
class SecondBrainCheck:
    name: str
    passed: bool
    detail: str
    refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "refs": self.refs,
        }


def run_second_brain_benchmark(
    db_path: str,
    *,
    agent_id: str = "second-brain-benchmark",
    mode: str = "context-only",
) -> dict[str, Any]:
    """Run deterministic checks for behavior beyond agentic RAG."""

    if mode not in SECOND_BRAIN_MODES:
        raise ValueError(
            "second-brain benchmark supports ingest-only and context-only modes; "
            f"got {mode!r}"
        )
    started = time.perf_counter()
    session_a = "second-brain-session-a"
    session_b = "second-brain-session-b"
    checks: list[SecondBrainCheck] = []
    mf = MemoryForge(db_path)
    try:
        ingest_started = time.perf_counter()
        seeded = _seed_second_brain_memory(
            mf,
            agent_id=agent_id,
            session_a=session_a,
            session_b=session_b,
        )
        ingest_ms = (time.perf_counter() - ingest_started) * 1000.0

        checks.extend(
            [
                _check_ingested_raw_evidence(mf, agent_id),
                _check_ingested_correction(
                    mf,
                    agent_id,
                    seeded["correction"]["item_id"],
                    seeded["wrong_hit"]["item_id"],
                ),
                _check_ingested_contradiction(
                    mf,
                    agent_id,
                    seeded["contradiction"]["item_id"],
                    seeded["contested_owner_id"],
                ),
            ]
        )

        if mode == "ingest-only":
            passed = all(check.passed for check in checks)
            dimension_coverage = _dimension_coverage(checks, mode=mode)
            return {
                "dataset": "second-brain",
                "mode": mode,
                "agent_id": agent_id,
                "passed": passed,
                "score": sum(1 for check in checks if check.passed),
                "total": len(checks),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "checks": [check.to_dict() for check in checks],
                "diagnostics": {
                    "answer_model_used": False,
                    "context_built": False,
                    "dimension_coverage": dimension_coverage,
                    "all_required_dimensions_passed": all(
                        dimension["passed"] for dimension in dimension_coverage.values()
                    ),
                    "ingest_latency_ms": ingest_ms,
                    "db_counts": _db_counts(mf),
                },
            }

        active = mf.active_recall(
            agent_id,
            session_id=session_b,
            focus="Project Atlas benchmark latency schema preference",
            limit=8,
            include_content=True,
        )
        bundle = mf.build_core_context_bundle(
            agent_id=agent_id,
            session_id=session_b,
            query="What should I remember before continuing Project Atlas?",
            budget=ContextBudget(model_context_limit=16_000),
            top_k=5,
            include_content=False,
            recall_content_policy="champion",
        ).to_dict()
        recall = mf.recall_long_term(
            agent_id,
            "Project Atlas retention key",
            top_k=3,
            include_content=True,
        )
        contradictions = mf.find_contradictions(
            agent_id,
            query="Project Atlas owner",
            limit=5,
            include_content=False,
        )

        checks.extend(
            [
                _check_active_recall_lifecycle(active, seeded["correction"]["item_id"]),
                _check_cross_session_continuity(
                    active,
                    previous_session_id=session_a,
                    current_session_id=session_b,
                ),
                _check_project_restart_quality(
                    active,
                    bundle,
                    correction_item_id=seeded["correction"]["item_id"],
                    stale_item_id=seeded["wrong_hit"]["item_id"],
                ),
                _check_correction_recall(
                    active,
                    recall,
                    seeded["correction"]["item_id"],
                    seeded["wrong_hit"]["item_id"],
                ),
                _check_contradiction_detection(
                    contradictions,
                    seeded["contradiction"]["item_id"],
                    seeded["contested_owner_id"],
                ),
                _check_bundle_provenance(bundle),
                _check_active_recall_injection(bundle),
                _check_core_boundary(bundle),
                _check_core_model_injected_memory_contract(bundle),
                _check_context_budget(bundle, max_tokens=4_000),
            ]
        )
        passed = all(check.passed for check in checks)
        dimension_coverage = _dimension_coverage(checks, mode=mode)
        return {
            "dataset": "second-brain",
            "mode": mode,
            "agent_id": agent_id,
            "passed": passed,
            "score": sum(1 for check in checks if check.passed),
            "total": len(checks),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "checks": [check.to_dict() for check in checks],
            "diagnostics": {
                "answer_model_used": False,
                "context_built": True,
                "dimension_coverage": dimension_coverage,
                "all_required_dimensions_passed": all(
                    dimension["passed"] for dimension in dimension_coverage.values()
                ),
                "ingest_latency_ms": ingest_ms,
                "active_recall_count": len(active["results"]),
                "bundle_token_estimate": bundle["token_estimate"],
                "bundle_raw_ref_count": len(bundle["raw_refs"]),
                "db_counts": _db_counts(mf),
            },
        }
    finally:
        mf.close()


def _seed_second_brain_memory(
    mf: MemoryForge,
    *,
    agent_id: str,
    session_a: str,
    session_b: str,
) -> dict[str, Any]:
    mf.store_conversation(
        agent_id,
        [
            {
                "role": "user",
                "content": "SQLite is the local durable memory default.",
            },
            {
                "role": "assistant",
                "content": "Schema changes stay small.",
            },
            {
                "role": "assistant",
                "content": (
                    "Before release, capture context bundle latency evidence. "
                    "Project Atlas retention key is blue-12."
                ),
            },
        ],
        session_id=session_a,
    )
    wrong_hit = mf.recall_long_term(
        agent_id,
        "Project Atlas retention key blue-12",
        top_k=1,
        include_content=True,
    )[0]
    correction = mf.record_correction(
        agent_id,
        "Project Atlas retention key is amber-17.",
        wrong_item_id=wrong_hit["item_id"],
        session_id=session_a,
    )
    contested_owner_id = mf.long_term.index_raw_item(
        agent_id,
        "note",
        "atlas-owner-mina",
        "Project Atlas owner is Mina.",
        metadata={
            "confidence": "normal",
        },
    )
    contradiction = mf.record_contradiction(
        agent_id,
        "Project Atlas owner is Rina.",
        conflicting_item_ids=[contested_owner_id],
        session_id=session_a,
    )
    mf.store_conversation(
        agent_id,
        [
            {
                "role": "user",
                "content": "I reopened Project Atlas and need the current memory state.",
            }
        ],
        session_id=session_b,
    )
    return {
        "wrong_hit": wrong_hit,
        "correction": correction,
        "contested_owner_id": contested_owner_id,
        "contradiction": contradiction,
    }


def _db_counts(mf: MemoryForge) -> dict[str, int]:
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
            row = mf.long_term.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        except Exception:
            counts[table] = -1
        else:
            counts[table] = int(row[0]) if row else 0
    return counts


def _check_ingested_raw_evidence(mf: MemoryForge, agent_id: str) -> SecondBrainCheck:
    rows = mf.long_term.conn.execute(
        """
        SELECT source_type, source_id, metadata
        FROM long_term_items
        WHERE agent_id = ?
        """,
        (agent_id,),
    ).fetchall()
    refs: list[str] = []
    for source_type, source_id, metadata_json in rows:
        metadata = _json_dict(metadata_json)
        if metadata.get(MetadataField.RAW_REFS) and metadata.get(MetadataField.SOURCE_ORIGIN):
            refs.append(f"{source_type}:{source_id}")
    return SecondBrainCheck(
        name="ingested_raw_evidence",
        passed=bool(refs),
        detail="ingested raw evidence with provenance" if refs else "missing raw evidence",
        refs=refs,
    )


def _check_ingested_correction(
    mf: MemoryForge,
    agent_id: str,
    correction_item_id: str,
    wrong_item_id: str,
) -> SecondBrainCheck:
    correction = mf.long_term_source(agent_id, correction_item_id)
    wrong = mf.long_term_source(agent_id, wrong_item_id)
    correction_metadata = (correction or {}).get("metadata") or {}
    wrong_metadata = (wrong or {}).get("metadata") or {}
    passed = correction_metadata.get(MetadataField.KIND) == "correction"
    passed = passed and correction_metadata.get(MetadataField.CONFIDENCE) == "high"
    passed = passed and correction_item_id in (
        wrong_metadata.get(MetadataField.SUPERSEDED_BY) or []
    )
    return SecondBrainCheck(
        name="ingested_correction",
        passed=passed,
        detail="correction persisted with supersession metadata"
        if passed
        else "correction metadata incomplete",
        refs=[str((correction or {}).get("raw_ref") or "")] if correction else [],
    )


def _check_ingested_contradiction(
    mf: MemoryForge,
    agent_id: str,
    contradiction_item_id: str,
    contested_item_id: str,
) -> SecondBrainCheck:
    contradiction = mf.long_term_source(agent_id, contradiction_item_id)
    contested = mf.long_term_source(agent_id, contested_item_id)
    contradiction_metadata = (contradiction or {}).get("metadata") or {}
    contested_metadata = (contested or {}).get("metadata") or {}
    passed = contradiction_metadata.get("contradicts")
    passed = passed and contradiction_item_id in (contested_metadata.get("contradicted_by") or [])
    return SecondBrainCheck(
        name="ingested_contradiction",
        passed=bool(passed),
        detail="contradiction persisted with reverse metadata"
        if passed
        else "contradiction metadata incomplete",
        refs=[str((contradiction or {}).get("raw_ref") or "")] if contradiction else [],
    )


def _check_active_recall_lifecycle(
    active: dict[str, Any],
    correction_item_id: str,
) -> SecondBrainCheck:
    matches = [
        item
        for item in active.get("results") or []
        if str(item.get("item_id")) == correction_item_id
        and "correction" in set(str(reason) for reason in item.get("reasons") or [])
    ]
    diagnostics = active.get("diagnostics") or {}
    passed = bool(matches)
    passed = passed and diagnostics.get("query_required") is False
    passed = passed and diagnostics.get("answer_model_used") is False
    passed = passed and diagnostics.get("semantic_focus_used") is False
    return SecondBrainCheck(
        name="active_recall_lifecycle",
        passed=passed,
        detail="active recall surfaces explicit correction lifecycle memory"
        if passed
        else "active recall missed explicit correction lifecycle memory",
        refs=[str(item["raw_ref"]) for item in matches],
    )


def _check_cross_session_continuity(
    active: dict[str, Any],
    *,
    previous_session_id: str,
    current_session_id: str,
) -> SecondBrainCheck:
    matches = [
        item
        for item in active.get("results") or []
        if (item.get("metadata") or {}).get(MetadataField.SESSION_ID) == previous_session_id
    ]
    passed = bool(matches) and active.get("session_id") == current_session_id
    return SecondBrainCheck(
        name="cross_session_continuity",
        passed=passed,
        detail="active recall surfaced prior-session memory for the reopened session"
        if passed
        else "prior-session memory was not surfaced for the reopened session",
        refs=[str(item["raw_ref"]) for item in matches],
    )


def _check_project_restart_quality(
    active: dict[str, Any],
    bundle: dict[str, Any],
    *,
    correction_item_id: str,
    stale_item_id: str,
) -> SecondBrainCheck:
    active_items = active.get("results") or []
    active_ids = {str(item.get("item_id")) for item in active_items}
    bundle_refs = set(str(ref) for ref in bundle.get("raw_refs") or [])
    provenance = bundle.get("provenance") or []
    passed = correction_item_id in active_ids
    passed = passed and stale_item_id not in active_ids
    passed = passed and bool(bundle_refs)
    passed = passed and any(item.get("source_origin") for item in provenance)
    refs = [
        str(item["raw_ref"])
        for item in active_items
        if str(item.get("item_id")) == correction_item_id
    ]
    return SecondBrainCheck(
        name="project_restart_quality",
        passed=passed,
        detail="restart context reconstructs current lifecycle memory and refs"
        if passed
        else "restart context missed current lifecycle memory or refs",
        refs=refs,
    )


def _check_correction_recall(
    active: dict[str, Any],
    recall: list[dict[str, Any]],
    correction_item_id: str,
    wrong_item_id: str,
) -> SecondBrainCheck:
    active_ids = {str(item["item_id"]) for item in active["results"]}
    recall_ids = [str(item["item_id"]) for item in recall]
    passed = correction_item_id in active_ids and recall_ids[:1] == [correction_item_id]
    passed = passed and wrong_item_id not in active_ids
    refs = [
        str(item["raw_ref"])
        for item in active["results"]
        if str(item["item_id"]) == correction_item_id
    ]
    return SecondBrainCheck(
        name="correction_learning",
        passed=passed,
        detail=(
            "correction supersedes stale fact"
            if passed
            else "correction was not preferred over stale fact"
        ),
        refs=refs,
    )


def _check_contradiction_detection(
    contradictions: dict[str, Any],
    contradiction_item_id: str,
    contested_item_id: str,
) -> SecondBrainCheck:
    diagnostics = contradictions.get("diagnostics") or {}
    result_ids = {str(item["item_id"]) for item in contradictions.get("results") or []}
    passed = contradiction_item_id in result_ids and contested_item_id in result_ids
    passed = passed and diagnostics.get("answer_model_used") is False
    refs = [
        str(item["raw_ref"])
        for item in contradictions.get("results") or []
        if str(item["item_id"]) in {contradiction_item_id, contested_item_id}
    ]
    return SecondBrainCheck(
        name="contradiction_detection",
        passed=passed,
        detail="conflicting memories are linked and retrievable"
        if passed
        else "conflicting memories were not both retrievable",
        refs=refs,
    )


def _check_bundle_provenance(bundle: dict[str, Any]) -> SecondBrainCheck:
    provenance = bundle.get("provenance") or []
    raw_refs = bundle.get("raw_refs") or []
    direct = [item for item in provenance if item.get("direct_evidence") is True]
    trusted = [
        item
        for item in provenance
        if item.get("raw_ref") and item.get("raw_refs") and item.get("source_origin")
    ]
    passed = bool(raw_refs) and bool(provenance) and bool(direct)
    passed = passed and len(trusted) == len(provenance)
    return SecondBrainCheck(
        name="provenance",
        passed=passed,
        detail="bundle provenance includes raw refs, origin, and evidence type"
        if passed
        else "bundle provenance is incomplete",
        refs=[str(ref) for ref in raw_refs[:5]],
    )


def _check_active_recall_injection(bundle: dict[str, Any]) -> SecondBrainCheck:
    active_recall = bundle.get("active_recall") or []
    messages = bundle.get("messages") or []
    diagnostics = (bundle.get("diagnostics") or {}).get("active_recall") or {}
    provenance = bundle.get("provenance") or []
    active_refs = {str(item.get("raw_ref")) for item in active_recall if item.get("raw_ref")}
    message_sources = [message.get("source") for message in messages]
    provenance_refs = {
        str(item.get("raw_ref"))
        for item in provenance
        if item.get("surface") == "active_recall"
    }
    passed = bool(active_recall)
    passed = passed and "active_recall" in message_sources
    passed = passed and bool(active_refs & provenance_refs)
    passed = passed and diagnostics.get("count") == len(active_recall)
    passed = passed and diagnostics.get("query_required") is False
    passed = passed and diagnostics.get("answer_model_used") is False
    if "long_term" in message_sources:
        passed = passed and message_sources.index("active_recall") < message_sources.index(
            "long_term"
        )
    return SecondBrainCheck(
        name="active_recall_injection",
        passed=passed,
        detail="bundle injects proactive lifecycle evidence before LTM recall"
        if passed
        else "bundle did not inject auditable proactive active recall",
        refs=sorted(active_refs),
    )


def _check_core_boundary(bundle: dict[str, Any]) -> SecondBrainCheck:
    diagnostics = bundle.get("diagnostics") or {}
    passed = diagnostics.get("bundle_only") is True
    passed = passed and diagnostics.get("answer_model_used") is False
    return SecondBrainCheck(
        name="core_model_boundary",
        passed=passed,
        detail="MemoryForge built context without answering"
        if passed
        else "context path crossed answer-model boundary",
        refs=[],
    )


def _check_core_model_injected_memory_contract(bundle: dict[str, Any]) -> SecondBrainCheck:
    diagnostics = bundle.get("diagnostics") or {}
    messages = bundle.get("messages") or []
    provenance = bundle.get("provenance") or []
    long_term_recall = bundle.get("long_term_recall") or []
    sources = {str(message.get("source")) for message in messages}
    active_provenance = [
        item for item in provenance if item.get("surface") == "active_recall"
    ]
    direct_refs = [
        ref
        for item in provenance
        if item.get("direct_evidence") is True
        for ref in item.get("raw_refs", [])
    ]
    passed = diagnostics.get("bundle_only") is True
    passed = passed and diagnostics.get("answer_model_used") is False
    passed = passed and "active_recall" in sources
    passed = passed and "long_term" in sources
    passed = passed and bool(long_term_recall)
    passed = passed and bool(active_provenance)
    passed = passed and bool(direct_refs)
    return SecondBrainCheck(
        name="core_model_injected_memory_contract",
        passed=passed,
        detail="core model receives injected active recall, LTM recall, and refs without answer model"
        if passed
        else "core model injection contract is incomplete",
        refs=[str(ref) for ref in direct_refs[:5]],
    )


def _check_context_budget(bundle: dict[str, Any], *, max_tokens: int) -> SecondBrainCheck:
    token_estimate = int(bundle.get("token_estimate") or 0)
    passed = 0 < token_estimate <= max_tokens
    return SecondBrainCheck(
        name="context_budget",
        passed=passed,
        detail=f"bundle token estimate {token_estimate} <= {max_tokens}"
        if passed
        else f"bundle token estimate {token_estimate} exceeded {max_tokens}",
        refs=[],
    )


def _dimension_coverage(
    checks: list[SecondBrainCheck],
    *,
    mode: str,
) -> dict[str, dict[str, Any]]:
    required = INGEST_DIMENSION_CHECKS if mode == "ingest-only" else CONTEXT_DIMENSION_CHECKS
    by_name = {check.name: check for check in checks}
    coverage: dict[str, dict[str, Any]] = {}
    for dimension, check_names in required.items():
        matched = [by_name[name] for name in check_names if name in by_name]
        refs = _dedupe_refs([ref for check in matched for ref in check.refs])
        coverage[dimension] = {
            "passed": bool(matched) and all(check.passed for check in matched),
            "checks": [check.name for check in matched],
            "refs": refs[:8],
        }
    return coverage


def _item_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get(key) or "") for key in ("preview", "content"))


def _contains_all(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for ref in refs:
        if not ref or ref in seen:
            continue
        seen.add(ref)
        deduped.append(ref)
    return deduped


def _json_dict(value: Any) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
