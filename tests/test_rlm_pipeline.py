import sqlite3
import threading
import time

import pytest

import memoryforge.rlm.runner as rlm_runner
from memoryforge import MemoryForge
from memoryforge.agents import SubAgentOperationResult, TransientSubAgentRunnerError
from memoryforge.lcm import SummaryDAG


def test_rlm_load_search_chunk_get_dispatch(tmp_path):
    db_path = str(tmp_path / "memory.db")
    source = tmp_path / "large.md"
    source.write_text(
        "# Notes\n\n"
        + "Alice designs SQLite memory. " * 200
        + "\n\nBob reviews the memory index. " * 120,
        encoding="utf-8",
    )

    mf = MemoryForge(db_path)
    try:
        loaded = mf.rlm_load("agent", source, name="notes", chunk_size=1_200, overlap=120)
        assert loaded["lossless"] is True
        assert loaded["chunk_count"] > 1

        results = mf.rlm_search("agent", "Alice SQLite", buffer_id=loaded["buffer_id"], limit=3)
        assert results
        assert "content" not in results[0]
        assert results[0]["ref"].startswith("rlm_chunk:")

        chunk = mf.rlm_chunk_get(results[0]["chunk_id"])
        assert chunk is not None
        assert "Alice" in chunk["content"]
        assert chunk["byte_range"]["end"] > chunk["byte_range"]["start"]

        plan = mf.rlm_dispatch(
            "agent",
            buffer_id=loaded["buffer_id"],
            query="Alice SQLite",
            limit=4,
            batch_size=2,
        )
        assert plan["run_id"]
        assert plan["batch_count"] == 2
        assert all(
            ref.startswith("rlm_chunk:") for batch in plan["batches"] for ref in batch["refs"]
        )
    finally:
        mf.close()


def test_rlm_dispatch_auto_batches_all_selected_chunks(tmp_path):
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        loaded = mf.rlm.load(
            agent_id="agent",
            value=(
                "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability.\n\n"
                * 120
            ),
            chunk_size=350,
            overlap=0,
        )

        plan = mf.rlm_dispatch("agent", buffer_id=loaded["buffer_id"], limit=4)

        assert plan["batch_count"] == 1
        assert plan["batch_size"] == 4
        assert len(plan["batches"][0]["chunk_ids"]) == 4
    finally:
        mf.close()


def test_rlm_load_skips_unchanged_source_and_reuses_indexes(tmp_path):
    db_path = str(tmp_path / "memory.db")
    source = tmp_path / "large.md"
    source.write_text("Alice designs SQLite memory.\n\n" * 300, encoding="utf-8")

    mf = MemoryForge(db_path)
    try:
        first = mf.rlm_load("agent", source, name="notes", chunk_size=1_200, overlap=120)
        first_counts = _table_counts(db_path)
        second = mf.rlm_load("agent", source, name="notes", chunk_size=1_200, overlap=120)
        second_counts = _table_counts(db_path)

        assert first["deduped"] is False
        assert second["deduped"] is True
        assert second["buffer_id"] == first["buffer_id"]
        assert second_counts == first_counts
        assert first["ingestion_manifest"]["content_hash"] == second["ingestion_manifest"]["content_hash"]
        assert first["ingestion_manifest"]["chunk_size"] == 1_200
        assert first["ingestion_manifest"]["overlap"] == 120
        assert first["ingestion_manifest"]["vector"]["model_key"] == second["ingestion_manifest"]["vector"]["model_key"]
        assert second["ingestion_manifest"]["rlm_deduped"] is True
        assert second["ingestion_manifest"]["ltm_deduped"] is True
        assert second["ingestion_manifest"]["vector"]["add_stats"]["encoded"] == 0
    finally:
        mf.close()


def test_rlm_load_rechunks_when_chunk_config_changes(tmp_path):
    db_path = str(tmp_path / "memory.db")
    source = tmp_path / "large.md"
    source.write_text("Alice designs SQLite memory.\n\n" * 300, encoding="utf-8")

    mf = MemoryForge(db_path)
    try:
        first = mf.rlm_load("agent", source, name="notes", chunk_size=1_200, overlap=120)
        second = mf.rlm_load("agent", source, name="notes", chunk_size=2_000, overlap=120)

        assert second["deduped"] is False
        assert second["buffer_id"] != first["buffer_id"]
    finally:
        mf.close()


def test_rlm_record_and_aggregate_preserve_lcm_dag_refs(tmp_path):
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        loaded = mf.rlm_load(
            "agent",
            "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability.",
            chunk_size=1_000,
        )
        plan = mf.rlm_dispatch("agent", buffer_id=loaded["buffer_id"], limit=10, batch_size=1)
        first_chunk = plan["batches"][0]["chunk_ids"][0]

        recorded = mf.rlm_record_result(
            "agent",
            plan["run_id"],
            [first_chunk],
            "Sub-agent finding: Alice owns authentication.",
            batch_index=0,
        )
        aggregate = mf.rlm_aggregate("agent", plan["run_id"])

        assert recorded["summary_node_id"] in aggregate["child_node_ids"]
        assert first_chunk in aggregate["source_chunk_ids"]
        assert aggregate["summary_node_id"]

        dag = SummaryDAG(str(tmp_path / "memory.db"))
        try:
            node = dag.get_node(aggregate["summary_node_id"])
            assert node is not None
            assert f"rlm_chunk:{first_chunk}" in node.source_refs
        finally:
            dag.close()
    finally:
        mf.close()


def _table_counts(db_path: str) -> dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("rlm_buffers", "rlm_chunks", "long_term_items", "vec_index")
        }


def test_rlm_run_spawns_runner_and_writes_lcm_dag(tmp_path):
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        result = mf.rlm_run(
            "agent",
            "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability." * 80,
            limit=2,
            batch_size=1,
            chunk_size=1_000,
            runner="mock",
            synthesize=True,
        )

        assert result["runner"] == "mock"
        assert result["plan"]["batch_count"] == 2
        assert len(result["records"]) == 2
        assert result["records"][0]["metadata"]["operation"] == "rlm.analyze"
        assert result["aggregate"]["metadata"]["operation"] == "rlm.synthesize"
        assert result["aggregate"]["summary_node_id"]
        assert len(result["aggregate"]["source_chunk_ids"]) == 2

        dag = SummaryDAG(str(tmp_path / "memory.db"))
        try:
            node = dag.get_node(result["aggregate"]["summary_node_id"])
            assert node is not None
            assert {
                f"rlm_chunk:{chunk_id}" for chunk_id in result["aggregate"]["source_chunk_ids"]
            }.issubset(set(node.source_refs))
        finally:
            dag.close()
    finally:
        mf.close()


def test_default_recursive_token_limit_grows_for_gpt55():
    assert rlm_runner._default_recursive_token_limit("gpt-5.5") == 64_000
    assert rlm_runner._default_recursive_token_limit("gpt-5.2") == 8_000


def test_rlm_run_can_analyze_batches_in_parallel(monkeypatch, tmp_path):
    class SlowOperator:
        provider = "slow"
        model = "test-model"
        active = 0
        max_active = 0
        lock = threading.Lock()

        def __init__(self, **_kwargs):
            pass

        def analyze_rlm_batch(self, *, plan, batch, chunks):
            del plan, chunks
            batch_index = int(batch["batch_index"])
            with self.lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            try:
                time.sleep(0.2)
                return SubAgentOperationResult(
                    kind="rlm.analyze",
                    provider=self.provider,
                    model=self.model,
                    text=f"analysis for batch {batch_index}",
                    elapsed_seconds=0.2,
                    input_hash=f"hash-{batch_index}",
                )
            finally:
                with self.lock:
                    type(self).active -= 1

        def synthesize_rlm_analyses(self, *, plan, analyses):  # pragma: no cover - guarded
            raise AssertionError("synthesis is disabled in this test")

    monkeypatch.setattr(rlm_runner, "SubAgentOperator", SlowOperator)
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        result = mf.rlm_run(
            "agent",
            "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability.\n\n"
            * 120,
            limit=4,
            batch_size=1,
            chunk_size=350,
            overlap=0,
            runner="mock",
            synthesize=False,
            recursive=False,
            max_workers=4,
        )

        assert result["plan"]["batch_count"] == 4
        assert result["plan"]["max_workers"] == 4
        assert SlowOperator.max_active > 1
        assert [record["metadata"]["batch_index"] for record in result["records"]] == [0, 1, 2, 3]
    finally:
        mf.close()


def test_rlm_run_fails_fast_on_batch_error_by_default(monkeypatch, tmp_path):
    class FailingOperator:
        provider = "failing"
        model = "test-model"

        def __init__(self, **_kwargs):
            pass

        def analyze_rlm_batch(self, *, plan, batch, chunks):
            del plan, chunks
            if int(batch["batch_index"]) == 1:
                raise RuntimeError("batch failed")
            return SubAgentOperationResult(
                kind="rlm.analyze",
                provider=self.provider,
                model=self.model,
                text="analysis",
                elapsed_seconds=0.01,
                input_hash="hash",
            )

        def synthesize_rlm_analyses(self, *, plan, analyses):  # pragma: no cover - guarded
            raise AssertionError("synthesis is disabled in this test")

    monkeypatch.setattr(rlm_runner, "SubAgentOperator", FailingOperator)
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        with pytest.raises(RuntimeError, match="batch failed"):
            mf.rlm_run(
                "agent",
                "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability.\n\n"
                * 120,
                limit=3,
                batch_size=1,
                chunk_size=350,
                overlap=0,
                runner="mock",
                synthesize=False,
                recursive=False,
            )
    finally:
        mf.close()


def test_rlm_run_can_store_partial_records_for_failed_batch(monkeypatch, tmp_path):
    class PartiallyFailingOperator:
        provider = "partial"
        model = "test-model"

        def __init__(self, **_kwargs):
            pass

        def analyze_rlm_batch(self, *, plan, batch, chunks):
            del plan, chunks
            batch_index = int(batch["batch_index"])
            if batch_index == 1:
                raise RuntimeError("batch failed")
            return SubAgentOperationResult(
                kind="rlm.analyze",
                provider=self.provider,
                model=self.model,
                text=f"analysis for batch {batch_index}",
                elapsed_seconds=0.25,
                input_hash=f"hash-{batch_index}",
            )

        def synthesize_rlm_analyses(self, *, plan, analyses):  # pragma: no cover - guarded
            raise AssertionError("synthesis is disabled in this test")

    monkeypatch.setattr(rlm_runner, "SubAgentOperator", PartiallyFailingOperator)
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        result = mf.rlm_run(
            "agent",
            "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability.\n\n"
            * 120,
            limit=4,
            batch_size=1,
            chunk_size=350,
            overlap=0,
            runner="mock",
            synthesize=False,
            recursive=False,
            allow_partial=True,
            max_workers=4,
        )

        assert len(result["records"]) == 3
        assert result["failures"][0]["batch_index"] == 1
        assert result["failures"][0]["chunk_ids"]
        assert result["aggregate"]["summary_node_id"]
        assert result["run_metrics"]["partial"] is True
        assert result["run_metrics"]["complete"] is False
        assert result["run_metrics"]["succeeded_batch_count"] == 3
        assert result["run_metrics"]["failed_batch_count"] == 1
        assert result["run_metrics"]["chunk_coverage_ratio"] == 0.75
        assert result["run_metrics"]["model_elapsed_seconds"] == 0.75
        assert result["run_metrics"]["analysis_input_tokens"] > 0
        assert result["run_metrics"]["analysis_output_tokens_estimate"] > 0
        assert result["run_metrics"]["analysis_total_tokens_estimate"] > result["run_metrics"][
            "analysis_input_tokens"
        ]
        assert result["run_metrics"]["total_tokens_estimate"] == result["run_metrics"][
            "analysis_total_tokens_estimate"
        ]
        assert result["run_metrics"]["cost_usd"] is None
        assert result["run_metrics"]["cost_source"] == "not_reported_by_runner"
        assert result["records"][0]["metadata"]["input_tokens_estimate"] > 0
        assert result["records"][0]["metadata"]["output_tokens_estimate"] > 0
        assert result["aggregate"]["metadata"]["run_metrics"]["partial"] is True
        assert result["plan"]["run_metrics"]["partial"] is True
    finally:
        mf.close()


def test_rlm_run_retries_transient_batch_failure(monkeypatch, tmp_path):
    class FlakyOperator:
        provider = "flaky"
        model = "test-model"
        attempts: dict[int, int] = {}

        def __init__(self, **_kwargs):
            pass

        def analyze_rlm_batch(self, *, plan, batch, chunks):
            del plan, chunks
            batch_index = int(batch["batch_index"])
            self.attempts[batch_index] = self.attempts.get(batch_index, 0) + 1
            if batch_index == 0 and self.attempts[batch_index] == 1:
                raise TransientSubAgentRunnerError("temporary timeout")
            return SubAgentOperationResult(
                kind="rlm.analyze",
                provider=self.provider,
                model=self.model,
                text=f"analysis for batch {batch_index}",
                elapsed_seconds=0.25,
                input_hash=f"hash-{batch_index}",
            )

        def synthesize_rlm_analyses(self, *, plan, analyses):  # pragma: no cover - guarded
            raise AssertionError("synthesis is disabled in this test")

    FlakyOperator.attempts = {}
    monkeypatch.setattr(rlm_runner, "SubAgentOperator", FlakyOperator)
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        result = mf.rlm_run(
            "agent",
            "Alice owns authentication.\n\nBob owns billing.\n\n" * 120,
            limit=2,
            batch_size=1,
            chunk_size=350,
            overlap=0,
            runner="mock",
            synthesize=False,
            recursive=False,
            max_retries=1,
        )

        assert result["failures"] == []
        assert result["records"][0]["metadata"]["retry_count"] == 1
        assert result["run_metrics"]["retry_count"] == 1
        assert result["run_metrics"]["complete"] is True
    finally:
        mf.close()


def test_rlm_run_does_not_retry_non_transient_batch_failure(monkeypatch, tmp_path):
    class NonTransientFailingOperator:
        provider = "nontransient"
        model = "test-model"
        attempts = 0

        def __init__(self, **_kwargs):
            pass

        def analyze_rlm_batch(self, *, plan, batch, chunks):
            del plan, batch, chunks
            self.__class__.attempts += 1
            raise RuntimeError("bad prompt")

        def synthesize_rlm_analyses(self, *, plan, analyses):  # pragma: no cover - guarded
            raise AssertionError("synthesis is disabled in this test")

    NonTransientFailingOperator.attempts = 0
    monkeypatch.setattr(rlm_runner, "SubAgentOperator", NonTransientFailingOperator)
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        with pytest.raises(RuntimeError, match="bad prompt"):
            mf.rlm_run(
                "agent",
                "Alice owns authentication.\n\nBob owns billing.\n\n" * 120,
                limit=1,
                batch_size=1,
                chunk_size=350,
                overlap=0,
                runner="mock",
                synthesize=False,
                recursive=False,
                max_retries=3,
            )

        assert NonTransientFailingOperator.attempts == 1
    finally:
        mf.close()


def test_rlm_run_raises_when_recursive_reduction_exceeds_limit(tmp_path):
    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        with pytest.raises(RuntimeError, match="RLM recursive reduction FAILED"):
            mf.rlm_run(
                "agent",
                "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns observability."
                * 120,
                limit=2,
                batch_size=1,
                chunk_size=1_000,
                runner="mock",
                synthesize=True,
                recursive=True,
                max_recursive_rounds=1,
                recursive_token_limit=1,
            )
    finally:
        mf.close()


def test_rlm_loads_python_file_as_lossless_document(tmp_path):
    path = tmp_path / "service.py"
    path.write_text(
        "def authenticate(name: str) -> bool:\n"
        "    return name.lower() == 'alice'\n\n"
        "def login() -> bool:\n"
        "    return authenticate('Alice')\n",
        encoding="utf-8",
    )

    mf = MemoryForge(str(tmp_path / "memory.db"))
    try:
        loaded = mf.rlm_load("agent", path)
        results = mf.rlm_search(
            "agent", "authenticate Alice", buffer_id=loaded["buffer_id"], limit=5
        )

        assert loaded["content_type"] == "docs"
        assert all("content" not in result for result in results)
    finally:
        mf.close()
