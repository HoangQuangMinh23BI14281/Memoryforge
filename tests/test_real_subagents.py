import os
import shutil
from pathlib import Path

import pytest

from memoryforge import MemoryForge
from memoryforge.agents import SubAgentOperator, SubAgentTask
from memoryforge.lcm import ContextBudget

if os.environ.get("MEMORYFORGE_REAL_SUBAGENT") != "1":
    pytestmark = pytest.mark.skip(
        reason="Set MEMORYFORGE_REAL_SUBAGENT=1 to run real Codex sub-agent tests"
    )


def _real_codex_project_root() -> Path:
    requested_runner = os.environ.get("MEMORYFORGE_SUBAGENT_RUNNER", "").lower()
    if requested_runner and requested_runner != "codex":
        raise AssertionError(
            "Real sub-agent tests must use Codex CLI; unset MEMORYFORGE_SUBAGENT_RUNNER "
            "or set it to codex."
        )
    if shutil.which("codex") is None:
        raise AssertionError("Real sub-agent tests require Codex CLI on PATH")
    model = (
        os.environ.get("MEMORYFORGE_SUBAGENT_MODEL")
        or os.environ.get("MEMORYFORGE_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    if not model:
        raise AssertionError(
            "Real Codex sub-agent tests require MEMORYFORGE_SUBAGENT_MODEL, "
            "MEMORYFORGE_MODEL, or OPENAI_MODEL"
        )
    root = Path(os.environ.get("MEMORYFORGE_REAL_PROJECT_ROOT", os.getcwd())).expanduser().resolve()
    if not root.exists():
        raise AssertionError(f"MEMORYFORGE_REAL_PROJECT_ROOT does not exist: {root}")
    return root


def test_real_subagent_operator_calls_codex_cli():
    project_root = _real_codex_project_root()
    operator = SubAgentOperator(runner="codex", project_root=str(project_root), timeout_s=180)

    result = operator.execute(
        SubAgentTask(
            kind="smoke.real",
            system_prompt="Return only this token: MEMORYFORGE_REAL_OK",
            user_prompt="No analysis is needed.",
            max_tokens=16,
            temperature=0,
        )
    )

    assert result.provider == "codex"
    assert result.model
    assert "MEMORYFORGE_REAL_OK" in result.text


def test_real_lcm_compaction_uses_codex_cli_operator(tmp_path):
    project_root = _real_codex_project_root()
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        session_id = "real-lcm-session"
        mf.store_conversation(
            "real-agent",
            [
                {
                    "role": "user",
                    "content": (
                        f"Message {index}: Alice owns authentication, Bob owns billing, "
                        "and Carol owns observability. Preserve this as source-backed context. "
                        "file_real123 "
                    )
                    * 12,
                }
                for index in range(6)
            ],
            session_id=session_id,
        )

        result = mf.lcm_compact_if_needed(
            "real-agent",
            session_id,
            force=True,
            max_rounds=1,
            runner="codex",
            project_root=str(project_root),
            budget=ContextBudget(
                model_context_limit=400,
                reserved_output_tokens=80,
                compaction_buffer=40,
                soft_threshold_fraction=0.5,
            ),
        )

        assert result.triggered is True
        assert result.rounds == 1
        assert result.summary_node_ids
        assert result.context.has_summary is True
        assert "[L3 DETERMINISTIC COMPACTION" not in result.context.messages[0].content
    finally:
        mf.close()


def test_real_rlm_run_uses_codex_cli_operator(tmp_path):
    project_root = _real_codex_project_root()
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        result = mf.rlm_run(
            "real-agent",
            "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability.\n\n" * 30,
            limit=1,
            batch_size=1,
            chunk_size=900,
            runner="codex",
            project_root=str(project_root),
            timeout_s=180,
            synthesize=False,
        )

        assert result["runner"] == "codex"
        assert len(result["records"]) == 1
        assert result["records"][0]["metadata"]["operation"] == "rlm.analyze"
        assert result["records"][0]["metadata"]["spawned"] is True
        assert result["aggregate"]["metadata"]["operation"] == "rlm.aggregate.deterministic"
    finally:
        mf.close()
