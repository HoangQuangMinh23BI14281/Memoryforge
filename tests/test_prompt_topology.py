import json

import pytest

from benchmarks.longmemeval_benchmark import (
    _answer_with_core_answer_runner,
    _model_messages_from_payload,
)
from memoryforge import MemoryForge
from memoryforge.agents import SubAgentOperationResult


class CapturingOperator:
    def __init__(self):
        self.task = None

    def execute(self, task):
        self.task = task
        return SubAgentOperationResult(
            kind=task.kind,
            provider="test",
            model="test-model",
            text="amber-17",
            elapsed_seconds=0.0,
            input_hash="input-hash",
        )


def test_core_context_bundle_model_payload_excludes_audit_fields(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Project Atlas key is amber-17."}],
            session_id="session-1",
        )

        bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Project Atlas key",
            top_k=1,
        )
        model_payload = bundle.to_model_payload()
        serialized_model_payload = json.dumps(model_payload)

        assert set(model_payload) == {"messages"}
        assert "diagnostics" not in serialized_model_payload
        assert "provenance" not in serialized_model_payload
        assert "budget" not in serialized_model_payload
        assert bundle.to_dict()["diagnostics"]["render_contract"]["audit_fields_rendered"] is False
        with pytest.raises(ValueError):
            _model_messages_from_payload(bundle.to_dict())
    finally:
        mf.close()


def test_core_answer_runner_rejects_audit_fields_and_prompts_only_with_messages():
    with pytest.raises(ValueError):
        _model_messages_from_payload(
            {
                "messages": [],
                "diagnostics": {"secret": "MUST_NOT_REACH_MODEL"},
            }
        )

    operator = CapturingOperator()
    result = _answer_with_core_answer_runner(
        operator,
        case_id="case-1",
        question="What is the key?",
        question_type=None,
        question_date=None,
        pipeline="test",
        model_payload={
            "messages": [
                {
                    "role": "system",
                    "source": "long_term",
                    "source_id": "long_term_recall",
                    "content": "Project Atlas key is amber-17.",
                }
            ]
        },
    )

    assert result["answer"] == "amber-17"
    assert operator.task is not None
    assert "lcm_context_messages" in operator.task.user_prompt
    assert "diagnostics" not in operator.task.user_prompt
    assert "provenance" not in operator.task.user_prompt
    assert "budget" not in operator.task.user_prompt
