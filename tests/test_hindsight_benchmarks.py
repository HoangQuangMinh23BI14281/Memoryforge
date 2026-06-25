from __future__ import annotations

from pathlib import Path

import pytest

from memoryforge.api import MemoryForge
from memoryforge.benchmark import LoComoAdapter, LongMemEvalAdapter
from memoryforge.memory.longterm.models import MetadataField

HINDSIGHT_ROOT = Path(
    "/mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource/hindsight/hindsight-dev"
)
LOCOMO10 = HINDSIGHT_ROOT / "benchmarks" / "locomo" / "datasets" / "locomo10.json"


def test_locomo_hindsight_fixture_ingests_sessions_and_retrieves_evidence(tmp_path):
    if not LOCOMO10.exists():
        pytest.skip(f"Hindsight LoCoMo fixture not found: {LOCOMO10}")

    adapter = LoComoAdapter()
    cases = adapter.load_cases(str(LOCOMO10))
    case = next(case for case in cases if case.question == "What did Caroline research?")
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        message_ids = adapter.ingest_case(mf, case, agent_id="locomo")
        result = adapter.run_case(mf, case, agent_id="locomo", top_k=10)

        assert len(cases) > 100
        assert message_ids
        assert "adoption agencies" in result.prediction.lower()
        assert result.correct is True
    finally:
        mf.close()


def test_longmemeval_hindsight_schema_ingests_sessions_and_retrieves_answer(tmp_path):
    adapter = LongMemEvalAdapter()
    payload = [
        {
            "question_id": "lme-smoke-1",
            "question": "Which cafe did Alex mention?",
            "answer": "Blue Bottle",
            "question_type": "single-session-user",
            "question_date": "2023/05/21 (Sun) 09:00",
            "haystack_session_ids": ["session-a"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "I met Sam at Blue Bottle before the product review.",
                        "has_answer": True,
                    },
                    {
                        "role": "assistant",
                        "content": "Noted that the cafe was Blue Bottle.",
                    },
                ]
            ],
        }
    ]
    case = next(iter(adapter.parse_cases(payload)))
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        message_ids = adapter.ingest_case(mf, case, agent_id="longmemeval")
        result = adapter.run_case(mf, case, agent_id="longmemeval", top_k=5)

        assert message_ids
        assert result.correct is True
        assert "Blue Bottle" in result.prediction
    finally:
        mf.close()


def test_longmemeval_adapter_marks_answer_evidence_metadata(tmp_path):
    adapter = LongMemEvalAdapter()
    payload = [
        {
            "question_id": "lme-evidence-1",
            "question": "Which cafe did Alex mention?",
            "answer": "Blue Bottle",
            "question_type": "single-session-user",
            "question_date": "2023/05/21 (Sun) 09:00",
            "haystack_session_ids": ["session-a"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "I met Sam at Blue Bottle before the product review.",
                        "has_answer": True,
                    },
                    {
                        "role": "assistant",
                        "content": "Noted that the cafe was Blue Bottle.",
                        "has_answer": False,
                    },
                ]
            ],
        }
    ]
    case = next(iter(adapter.parse_cases(payload)))
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        adapter.ingest_case(mf, case, agent_id="longmemeval")
        hits = mf.recall_long_term("longmemeval", "Blue Bottle", top_k=2)

        assert hits[0]["metadata"][MetadataField.ANSWER_EVIDENCE] is True
        assert hits[0]["metadata"][MetadataField.BENCHMARK_SESSION_ID] == "session-a"
        assert hits[0]["streams"]["rerank"]["answer_evidence_bonus"] == 1
    finally:
        mf.close()


def test_locomo_adapter_marks_dialogue_evidence_metadata(tmp_path):
    adapter = LoComoAdapter()
    sample = {
        "sample_id": "locomo-mini",
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
        hits = mf.recall_long_term("locomo", "adoption agencies", top_k=2)

        assert hits[0]["metadata"][MetadataField.ANSWER_EVIDENCE] is True
        assert hits[0]["metadata"][MetadataField.BENCHMARK_DIALOGUE_ID] == "D1:1"
        assert hits[0]["streams"]["rerank"]["answer_evidence_bonus"] == 1
    finally:
        mf.close()
