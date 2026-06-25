from memoryforge.benchmark import run_second_brain_benchmark
from memoryforge.cli.main import main


def test_second_brain_benchmark_checks_lifecycle_without_answer_model(tmp_path):
    result = run_second_brain_benchmark(str(tmp_path / "memory.db"))
    checks = {check["name"]: check for check in result["checks"]}

    assert result["dataset"] == "second-brain"
    assert result["mode"] == "context-only"
    assert result["passed"] is True
    assert result["score"] == result["total"]
    assert result["diagnostics"]["answer_model_used"] is False
    assert result["diagnostics"]["context_built"] is True
    assert result["diagnostics"]["all_required_dimensions_passed"] is True
    assert set(result["diagnostics"]["dimension_coverage"]) == {
        "cross_session_continuity",
        "correction_learning",
        "contradiction_detection",
        "lifecycle_active_recall",
        "project_restart_quality",
        "answer_provenance",
        "context_budget_efficiency",
        "core_model_injected_memory",
    }
    assert checks["active_recall_lifecycle"]["passed"] is True
    assert checks["correction_learning"]["passed"] is True
    assert checks["contradiction_detection"]["passed"] is True
    assert checks["provenance"]["refs"]
    assert checks["active_recall_injection"]["passed"] is True
    assert checks["active_recall_injection"]["refs"]
    assert checks["core_model_boundary"]["passed"] is True
    assert checks["core_model_injected_memory_contract"]["passed"] is True


def test_second_brain_benchmark_ingest_only_mode_does_not_build_context(tmp_path):
    result = run_second_brain_benchmark(str(tmp_path / "memory.db"), mode="ingest-only")
    checks = {check["name"]: check for check in result["checks"]}

    assert result["dataset"] == "second-brain"
    assert result["mode"] == "ingest-only"
    assert result["passed"] is True
    assert result["diagnostics"]["answer_model_used"] is False
    assert result["diagnostics"]["context_built"] is False
    assert result["diagnostics"]["all_required_dimensions_passed"] is True
    assert set(result["diagnostics"]["dimension_coverage"]) == {
        "semantic_ledger_ingestion",
        "correction_metadata",
        "contradiction_metadata",
    }
    assert result["diagnostics"]["db_counts"]["long_term_items"] > 0
    assert checks["ingested_raw_evidence"]["passed"] is True
    assert checks["ingested_correction"]["passed"] is True
    assert checks["ingested_contradiction"]["passed"] is True


def test_second_brain_benchmark_cli(tmp_path, capsys):
    exit_code = main(
        [
            "--db",
            str(tmp_path / "memory.db"),
            "benchmark",
            "--dataset",
            "second-brain",
            "--mode",
            "context-only",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"dataset": "second-brain"' in output
    assert '"mode": "context-only"' in output
    assert '"passed": true' in output


def test_second_brain_benchmark_cli_ingest_only(tmp_path, capsys):
    exit_code = main(
        [
            "--db",
            str(tmp_path / "memory.db"),
            "benchmark",
            "--dataset",
            "second-brain",
            "--mode",
            "ingest-only",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"mode": "ingest-only"' in output
    assert '"context_built": false' in output
