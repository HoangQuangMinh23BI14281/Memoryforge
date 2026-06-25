import json

from benchmarks.locomo_benchmark import (
    _case_already_ingested,
)
from benchmarks.locomo_benchmark import (
    _mode_contract as _locomo_mode_contract,
)
from benchmarks.locomo_benchmark import (
    _performance_summary as _locomo_performance_summary,
)
from benchmarks.locomo_benchmark import (
    _run_context_only_case as _run_locomo_context_only_case,
)
from benchmarks.longmemeval_benchmark import (
    _answer_matches,
    _clean_core_answer,
    _db_counts,
    _mode_contract,
    _performance_summary,
    _run_case,
)
from memoryforge import MemoryForge
from memoryforge.agents.operators import SubAgentOperationResult
from memoryforge.benchmark import BenchmarkCase, LoComoAdapter
from memoryforge.memory.longterm.models import LongTermRecallResult, MetadataField
from memoryforge.memory.longterm.retrieval import _recall_text


def test_longmemeval_core_answer_cleaning_and_normalized_exact_match():
    assert (
        _clean_core_answer("<think>hidden reasoning</think>\nYou bought it downtown.")
        == "You bought it downtown."
    )
    assert (
        _answer_matches(
            "the sports store downtown",
            "You bought your new tennis racket from a sports store downtown.",
        )
        is True
    )
    assert _answer_matches("February 14th", "February 14.") is True
    assert _answer_matches("February 14th", "back in February.") is False


def test_answer_evidence_snippet_keeps_wider_adjacent_context():
    answer_turn = "[8:user] I volunteered at the fundraising dinner on Valentine's Day. "
    content = (
        answer_turn
        + "supporting a good cause is fantastic. " * 18
        + "[9:assistant] The fundraising dinner sounds meaningful for animal welfare."
    )
    result = LongTermRecallResult(
        item_id="item",
        source_type="rlm_chunk",
        source_id="chunk",
        content_id="content",
        raw_ref="rlm_chunk:chunk",
        preview=content[:160],
        score=1.0,
        streams={"bm25": {"rank": 1}},
        metadata={
            MetadataField.ANSWER_EVIDENCE: True,
            MetadataField.ANSWER_EVIDENCE_RANGES: [
                {"start": 0, "end": len(answer_turn)}
            ],
        },
        content=content,
    )

    snippet = _recall_text(
        result,
        "snippet",
        query="When did I volunteer at the local animal shelter fundraising dinner?",
    )

    assert "Valentine's Day" in snippet
    assert len(snippet) > 720


def test_longmemeval_performance_summary_separates_answer_latency():
    summary = _performance_summary(
        [
            {
                "latency_ms": 1_250.0,
                "correct": True,
                "core_answer_runner": {"elapsed_seconds": 0.5},
                "diagnostics": {
                    "context": {
                        "token_estimate": 3_000,
                        "raw_ref_count": 4,
                        "long_term_recall_count": 5,
                    },
                    "context_bundle": {
                        "latency_ms": {
                            "context_build": 10.0,
                            "long_term_recall": 40.0,
                            "recall_injection": 5.0,
                            "total": 60.0,
                        }
                    },
                },
            },
            {
                "latency_ms": 980.0,
                "correct": False,
                "core_answer_runner": {"elapsed_seconds": 0.25},
                "diagnostics": {
                    "context": {
                        "token_estimate": 3_800,
                        "raw_ref_count": 3,
                        "long_term_recall_count": 4,
                    },
                    "context_bundle": {
                        "latency_ms": {
                            "context_build": 12.0,
                            "long_term_recall": 30.0,
                            "recall_injection": 4.0,
                            "total": 50.0,
                        }
                    },
                },
            },
        ]
    )

    assert summary["exact_score"] == 0.5
    assert summary["true_miss_count"] == 1
    assert summary["answer_latency_ms"]["avg"] == 375.0
    assert summary["memoryforge_query_latency_ms"]["avg"] == 55.0
    assert summary["ingest_or_setup_latency_ms"]["avg"] == 685.0
    assert summary["context_tokens"]["p50"] <= 4_000
    assert summary["targets"]["answer_latency_separated"] is True
    assert summary["refs_included_per_answer"]["avg"] == summary["raw_refs_per_answer"]["avg"]
    assert summary["semantic_score"] is None
    assert summary["semantic_score_available"] is False


def test_longmemeval_context_mode_reuses_unchanged_ingestion_indexes(tmp_path):
    db_path = str(tmp_path / "memory.db")
    case = BenchmarkCase(
        case_id="case-1",
        question="Who owns Project Atlas?",
        answer="Mina",
        metadata={
            "raw": {
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": "Mina owns Project Atlas.",
                        }
                    ]
                ],
                "haystack_dates": ["20 June 2026"],
                "haystack_session_ids": ["session-a"],
            }
        },
    )
    protected_tables = {
        "rlm_buffers",
        "rlm_chunks",
        "long_term_items",
        "vec_index",
        "search_fts",
    }

    mf = MemoryForge(db_path)
    try:
        preload = _run_case(
            mf=mf,
            operator=None,
            case=case,
            agent_id="benchmark-agent",
            mode="ingest-only",
            top_k=3,
            chunk_size=1_000,
            overlap=100,
            context_limit=8_000,
            recall_content_policy="champion",
        )
        ingest_counts = _db_counts(db_path)
        first = _run_case(
            mf=mf,
            operator=None,
            case=case,
            agent_id="benchmark-agent",
            mode="context-only",
            top_k=3,
            chunk_size=1_000,
            overlap=100,
            context_limit=8_000,
            recall_content_policy="champion",
        )
        first_counts = _db_counts(db_path)
        second = _run_case(
            mf=mf,
            operator=None,
            case=case,
            agent_id="benchmark-agent",
            mode="context-only",
            top_k=3,
            chunk_size=1_000,
            overlap=100,
            context_limit=8_000,
            recall_content_policy="champion",
        )
        second_counts = _db_counts(db_path)

        assert preload["diagnostics"]["mode_contract"]["ingests"] is True
        manifest = preload["diagnostics"]["ingestion_manifest"]
        assert manifest["chunk_size"] == 1_000
        assert manifest["overlap"] == 100
        assert manifest["long_term_item_count"] > 0
        assert preload["diagnostics"]["ltm_indexed_count"] == manifest["long_term_item_count"]
        assert first["diagnostics"]["mode_contract"]["ingests"] is False
        assert first["diagnostics"]["ingestion"]["performed"] is False
        assert first["diagnostics"]["ingestion"]["preingested_ltm_count"] > 0
        assert first["diagnostics"]["rlm_buffer_id"] is None
        assert first["diagnostics"]["rlm_chunk_count"] == 0
        assert first["diagnostics"]["ltm_indexed_count"] == 0
        assert second["diagnostics"]["context_bundle"]["answer_model_used"] is False
        assert second["diagnostics"]["context_bundle"]["raw_refs"]
        assert second["diagnostics"]["context_bundle"]["provenance"]
        assert all(
            item.get("raw_refs")
            for item in second["diagnostics"]["context_bundle"]["provenance"]
        )
        assert "subagent" not in first
        assert "subagent" not in second
        for table in protected_tables:
            assert first_counts[table] == ingest_counts[table]
            assert second_counts[table] == first_counts[table]
    finally:
        mf.close()


def test_longmemeval_rlm_ingest_marks_answer_evidence_by_session_span(tmp_path):
    db_path = str(tmp_path / "memory.db")
    case = BenchmarkCase(
        case_id="case-evidence",
        question="What degree did I graduate with?",
        answer="Business Administration",
        metadata={
            "question_date": "2026/06/20",
            "question_type": "single-session-user",
            "raw": {
                "question_date": "2026/06/20",
                "question_type": "single-session-user",
                "answer_session_ids": ["answer-session"],
                "haystack_sessions": [
                    [
                        {
                            "role": "user",
                            "content": "Morning routine notes. " * 20,
                        }
                    ],
                    [
                        {
                            "role": "user",
                            "content": "Paperwork and onboarding notes. " * 20,
                        },
                        {
                            "role": "assistant",
                            "content": (
                                "Congratulations on your degree in Business Administration. "
                                "Keep the reimbursement paperwork organized."
                            ),
                            "has_answer": True,
                        },
                    ],
                ],
                "haystack_dates": ["2026/06/18", "2026/06/19"],
                "haystack_session_ids": ["distractor-session", "answer-session"],
            },
        },
    )

    mf = MemoryForge(db_path)
    try:
        result = _run_case(
            mf=mf,
            operator=None,
            case=case,
            agent_id="benchmark-agent",
            mode="ingest-only",
            top_k=3,
            chunk_size=300,
            overlap=30,
            context_limit=8_000,
            recall_content_policy="champion",
        )
        annotation = result["diagnostics"]["answer_evidence_annotation"]
        metadata_rows = mf.long_term.conn.execute(
            """
            SELECT metadata
            FROM long_term_items
            WHERE agent_id = ?
            """,
            ("benchmark-agent",),
        ).fetchall()
        evidence_metadata = [
            json.loads(row[0])
            for row in metadata_rows
            if json.loads(row[0]).get(MetadataField.ANSWER_EVIDENCE) is True
        ]

        assert annotation["answer_session_ids"] == ["answer-session"]
        assert annotation["matched_item_count"] > 0
        assert annotation["matched_session_ids"] == ["answer-session"]
        assert evidence_metadata
        assert all(
            metadata[MetadataField.BENCHMARK_CASE_ID] == "case-evidence"
            for metadata in evidence_metadata
        )
        assert all(
            metadata[MetadataField.BENCHMARK_SESSION_ID] == "answer-session"
            for metadata in evidence_metadata
        )
        assert all(
            "answer-session" in metadata[MetadataField.EVIDENCE_IDS]
            for metadata in evidence_metadata
        )
        assert any(
            metadata.get(MetadataField.ANSWER_EVIDENCE_RANGES)
            for metadata in evidence_metadata
        )
    finally:
        mf.close()


def test_longmemeval_core_answer_result_uses_core_runner_not_subagent_key(tmp_path):
    class FakeCoreAnswerRunner:
        def execute(self, task):
            return SubAgentOperationResult(
                kind=task.kind,
                text="Mina owns Project Atlas.",
                provider="unit-core-runner",
                model="unit-model",
                elapsed_seconds=0.01,
                input_hash="unit-hash",
                cached=False,
            )

    db_path = str(tmp_path / "memory.db")
    case = BenchmarkCase(
        case_id="case-1",
        question="Who owns Project Atlas?",
        answer="Mina",
        metadata={
            "raw": {
                "haystack_sessions": [
                    [{"role": "user", "content": "Mina owns Project Atlas."}]
                ],
                "haystack_dates": ["20 June 2026"],
                "haystack_session_ids": ["session-a"],
            }
        },
    )

    mf = MemoryForge(db_path)
    try:
        result = _run_case(
            mf=mf,
            operator=FakeCoreAnswerRunner(),
            case=case,
            agent_id="benchmark-agent",
            mode="core-answer",
            top_k=3,
            chunk_size=1_000,
            overlap=100,
            context_limit=8_000,
            recall_content_policy="champion",
        )

        assert result["correct"] is True
        assert result["core_answer_runner"]["provider"] == "unit-core-runner"
        assert "subagent" not in result
    finally:
        mf.close()


def test_longmemeval_context_only_special_probe_does_not_call_lcm_worker(tmp_path):
    db_path = str(tmp_path / "memory.db")
    case = BenchmarkCase(
        case_id="lcm-dag-probe-unit",
        question="What is the retention key for Project Atlas?",
        answer="amber-17",
        metadata={
            "probe": {
                "case_id": "lcm-dag-probe-unit",
                "question": "What is the retention key for Project Atlas?",
                "answer": "amber-17",
                "fact": "Project Atlas uses retention key amber-17.",
            }
        },
    )

    mf = MemoryForge(db_path)
    try:
        _run_case(
            mf=mf,
            operator=None,
            case=case,
            agent_id="benchmark-agent",
            mode="ingest-only",
            top_k=3,
            chunk_size=1_000,
            overlap=100,
            context_limit=8_000,
            recall_content_policy="snippet",
            is_special=True,
        )
        result = _run_case(
            mf=mf,
            operator=None,
            case=case,
            agent_id="benchmark-agent",
            mode="context-only",
            top_k=3,
            chunk_size=1_000,
            overlap=100,
            context_limit=8_000,
            recall_content_policy="snippet",
            is_special=True,
        )

        contract = result["diagnostics"]["mode_contract"]
        assert contract["uses_lcm_worker"] is False
        assert contract["uses_core_answer_runner"] is False
        assert result["diagnostics"]["compaction"] is None
        assert result["core_answer_runner"] is None
        assert result["prediction"] is None
    finally:
        mf.close()


def test_longmemeval_mode_contract_marks_only_special_core_answer_lcm_worker():
    normal_core = _mode_contract("core-answer", is_special=False)
    special_core = _mode_contract("core-answer", is_special=True)
    special_context = _mode_contract("context-only", is_special=True)

    assert normal_core["uses_lcm_worker"] is False
    assert special_core["uses_lcm_worker"] is True
    assert special_core["uses_core_answer_runner"] is True
    assert special_context["ingests"] is False
    assert special_context["uses_lcm_worker"] is False
    assert special_context["uses_core_answer_runner"] is False


def test_locomo_context_only_result_is_bundle_with_provenance(tmp_path):
    adapter = LoComoAdapter()
    sample = {
        "sample_id": "locomo-unit",
        "conversation": {
            "speaker_a": "Caroline",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:1",
                    "text": "I researched adoption agencies yesterday.",
                },
                {
                    "speaker": "Melanie",
                    "dia_id": "D1:2",
                    "text": "That sounds like a long research task.",
                },
            ],
        },
        "qa": [
            {
                "question": "What did Caroline research?",
                "answer": "adoption agencies",
                "evidence": ["D1:1"],
                "category": 1,
            }
        ],
    }
    case = next(iter(adapter.parse_cases(sample)))
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        adapter.ingest_case(mf, case, agent_id="locomo")
        assert _case_already_ingested(mf, adapter, case, agent_id="locomo") is True
        result = _run_locomo_context_only_case(
            mf=mf,
            adapter=adapter,
            case=case,
            agent_id="locomo",
            top_k=5,
            context_limit=8_000,
            recall_content_policy="snippet",
        )
        diagnostics = result["diagnostics"]
        context_bundle = diagnostics["context_bundle"]
        retrieval = context_bundle["retrieval"]
        summary = _locomo_performance_summary([result])

        assert result["prediction"] is None
        assert result["core_answer_runner"] is None
        assert diagnostics["mode_contract"] == _locomo_mode_contract("context-only")
        assert diagnostics["ingestion"]["performed"] is False
        assert diagnostics["ingestion"]["preingested"] is True
        assert context_bundle["bundle_only"] is True
        assert context_bundle["answer_model_used"] is False
        assert context_bundle["raw_refs"]
        assert context_bundle["provenance"]
        assert all(item.get("raw_refs") for item in context_bundle["provenance"])
        assert retrieval["selection"] == "local_model_free"
        assert retrieval["llm_selection_used"] is False
        assert {"bm25", "vector"} & set(retrieval["stream_counts"])
        assert summary["memoryforge_query_latency_ms"]["count"] == 1
        assert summary["answer_latency_ms"]["count"] == 0
        assert summary["answer_latency_separated"] is True
    finally:
        mf.close()
