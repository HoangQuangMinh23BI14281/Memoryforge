import pytest

from memoryforge import MemoryForge
from memoryforge.cli.main import main
from memoryforge.init import init_project
from memoryforge.runtime import RuntimeIntegrationError, resolve_runtime_integration


def _python_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    return project


def test_runtime_context_requires_configured_codex_delivery(tmp_path):
    project = _python_project(tmp_path)

    with pytest.raises(RuntimeIntegrationError, match="Could not identify active runtime"):
        resolve_runtime_integration(project)

    init_project(str(project), agent_id="agent", configure_codex=False, force=True)

    with pytest.raises(RuntimeIntegrationError, match="missing MemoryForge MCP delivery"):
        resolve_runtime_integration(project, runtime="codex")


def test_runtime_context_bundle_uses_core_runtime_boundary(tmp_path):
    project = _python_project(tmp_path)
    initialized = init_project(str(project), agent_id="agent", force=True)
    mf = MemoryForge(initialized["db_path"])
    try:
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Project Atlas uses SQLite memory."}],
            session_id="session-1",
        )

        payload = mf.build_runtime_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Project Atlas memory",
            project_root=project,
        )

        assert payload["runtime"]["runtime"] == "codex"
        assert payload["runtime"]["mcp_configured"] is True
        assert payload["delivery"] == "mcp:build_context_bundle"
        assert payload["context_bundle"]["diagnostics"]["bundle_only"] is True
        assert payload["context_bundle"]["diagnostics"]["answer_model_used"] is False
        assert payload["context_bundle"]["diagnostics"]["runtime"]["answer_model_used"] is False
        assert payload["context_bundle"]["diagnostics"]["runtime"]["db_path_verified"] is True
    finally:
        mf.close()


def test_runtime_context_fails_when_validated_delivery_points_at_different_db(tmp_path):
    project = _python_project(tmp_path)
    initialized = init_project(str(project), agent_id="agent", force=True)
    mf = MemoryForge(str(tmp_path / "other-memory.db"))
    try:
        with pytest.raises(RuntimeIntegrationError, match="active MemoryForge database"):
            mf.build_runtime_context_bundle(
                agent_id="agent",
                session_id="session-1",
                query="Project Atlas memory",
                project_root=project,
            )

        with pytest.raises(RuntimeIntegrationError, match="active MemoryForge database"):
            resolve_runtime_integration(
                project,
                expected_db_path=str(tmp_path / "other-memory.db"),
            )
        assert initialized["db_path"] != str(tmp_path / "other-memory.db")
    finally:
        mf.close()


def test_runtime_context_cli_outputs_validated_bundle(tmp_path, capsys):
    project = _python_project(tmp_path)
    initialized = init_project(str(project), agent_id="agent", force=True)
    mf = MemoryForge(initialized["db_path"])
    try:
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Project Atlas uses SQLite memory."}],
            session_id="session-1",
        )
    finally:
        mf.close()

    exit_code = main(
        [
            "--db",
            initialized["db_path"],
            "runtime-context",
            "--agent-id",
            "agent",
            "--session-id",
            "session-1",
            "--query",
            "Project Atlas memory",
            "--project-root",
            str(project),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"runtime": "codex"' in output
    assert '"delivery": "mcp:build_context_bundle"' in output
    assert '"answer_model_used": false' in output
