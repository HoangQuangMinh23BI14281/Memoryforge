import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from memoryforge.agents import BaseSubAgentRunner, SubAgentOperator, SubAgentResponse
from memoryforge.agents.subagents import (
    CodexSubAgentRunner,
    OpenAIResponsesRunner,
    SubAgentRunnerError,
    TransientSubAgentRunnerError,
    create_subagent_runner,
)


class CountingRunner(BaseSubAgentRunner):
    provider = "counting"

    def __init__(self):
        super().__init__(model="test-model")
        self.prompts = []
        self.complete_count = 0

    def complete(self, prompt: str) -> SubAgentResponse:
        self.complete_count += 1
        self.prompts.append(prompt)
        return SubAgentResponse(self.provider, self.model, "operator result", 0.01)

    def stream(self, prompt: str):
        raise AssertionError("SubAgentOperator must use the sync complete() contract")


def _empty_codex_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_codex_runner_requires_explicit_model(monkeypatch, tmp_path):
    _empty_codex_home(monkeypatch, tmp_path)
    monkeypatch.delenv("MEMORYFORGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(
        "memoryforge.agents.subagents.shutil.which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    with pytest.raises(SubAgentRunnerError, match="requires --model"):
        create_subagent_runner("codex")


def test_auto_runner_does_not_use_codex_default_model(monkeypatch, tmp_path):
    _empty_codex_home(monkeypatch, tmp_path)
    monkeypatch.delenv("MEMORYFORGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MEMORYFORGE_SUBAGENT_CMD", raising=False)
    monkeypatch.setattr(
        "memoryforge.agents.subagents.shutil.which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    with pytest.raises(SubAgentRunnerError, match="no explicit model"):
        create_subagent_runner("auto")


def test_auto_runner_uses_codex_when_model_is_explicit(monkeypatch, tmp_path):
    _empty_codex_home(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMORYFORGE_MODEL", "gpt-5.2")
    monkeypatch.delenv("MEMORYFORGE_SUBAGENT_CMD", raising=False)
    monkeypatch.setattr(
        "memoryforge.agents.subagents.shutil.which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    runner = create_subagent_runner("auto")

    assert isinstance(runner, CodexSubAgentRunner)
    assert runner.model == "gpt-5.2"


def test_codex_command_forces_model_config_override():
    runner = CodexSubAgentRunner(
        binary="/usr/bin/codex", model="gpt-5.2", project_root="/tmp/project"
    )

    command = runner._build_command(output_path="/tmp/out.md")  # type: ignore[arg-type]

    assert command[0:2] == ["/usr/bin/codex", "exec"]
    assert "--model" in command
    assert command[command.index("--model") + 1] == "gpt-5.2"
    assert "-c" in command
    assert 'model="gpt-5.2"' in command
    assert command[-1] == "-"


def test_openai_runner_sends_exact_model_to_custom_proxy(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return (
                b'{"output_text":"OK","usage":{"input_tokens":7,'
                b'"output_tokens":3,"total_tokens":10}}'
            )

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = request.data.decode("utf-8")
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    monkeypatch.setattr("memoryforge.agents.subagents.urllib.request.urlopen", fake_urlopen)
    runner = OpenAIResponsesRunner(
        api_key=None, model="gpt-5.2", base_url="https://opusmax.shop", timeout_s=12
    )

    response = runner.complete("Return OK.")

    assert response.text == "OK"
    assert response.input_tokens == 7
    assert response.output_tokens == 3
    assert response.total_tokens == 10
    assert captured["url"] == "https://opusmax.shop/responses"
    assert '"model": "gpt-5.2"' in captured["body"]
    assert captured["authorization"] is None
    assert captured["timeout"] == 12


def test_openai_runner_reports_http_error_body(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            403,
            "Forbidden",
            hdrs={},
            fp=BytesIO(b'{"error":"missing api key"}'),
        )

    monkeypatch.setattr("memoryforge.agents.subagents.urllib.request.urlopen", fake_urlopen)
    runner = OpenAIResponsesRunner(api_key=None, model="gpt-5.2", base_url="https://opusmax.shop")

    with pytest.raises(SubAgentRunnerError, match="missing api key"):
        runner.complete("Return OK.")


def test_openai_runner_marks_rate_limit_as_transient(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs={},
            fp=BytesIO(b'{"error":"rate limited"}'),
        )

    monkeypatch.setattr("memoryforge.agents.subagents.urllib.request.urlopen", fake_urlopen)
    runner = OpenAIResponsesRunner(api_key=None, model="gpt-5.2", base_url="https://opusmax.shop")

    with pytest.raises(TransientSubAgentRunnerError, match="rate limited"):
        runner.complete("Return OK.")


def test_openai_runner_accepts_extra_headers(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def read(self):
            return b'{"output_text":"OK"}'

    def fake_urlopen(request, timeout):
        captured["x_proxy_token"] = request.get_header("X-proxy-token")
        return Response()

    monkeypatch.setenv("MEMORYFORGE_OPENAI_HEADERS", '{"X-Proxy-Token":"abc"}')
    monkeypatch.setattr("memoryforge.agents.subagents.urllib.request.urlopen", fake_urlopen)
    runner = OpenAIResponsesRunner(api_key=None, model="gpt-5.2", base_url="https://opusmax.shop")

    runner.complete("Return OK.")

    assert captured["x_proxy_token"] == "abc"


def test_auto_runner_ignores_openai_base_url_and_uses_codex(monkeypatch, tmp_path):
    _empty_codex_home(monkeypatch, tmp_path)
    monkeypatch.setenv("MEMORYFORGE_OPENAI_BASE_URL", "https://opusmax.shop")
    monkeypatch.setenv("MEMORYFORGE_MODEL", "gpt-5.2")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMORYFORGE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMORYFORGE_SUBAGENT_CMD", raising=False)
    monkeypatch.setattr(
        "memoryforge.agents.subagents.shutil.which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    runner = create_subagent_runner("auto")

    assert isinstance(runner, CodexSubAgentRunner)
    assert runner.model == "gpt-5.2"


def test_auto_runner_uses_project_codex_runner_without_cli_args(monkeypatch, tmp_path):
    home = _empty_codex_home(monkeypatch, tmp_path)
    codex_dir = home / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text(
        "\n".join(
            [
                'model_provider = "OpenAI"',
                'model = "gpt-5.5"',
                "",
                "[model_providers.OpenAI]",
                'base_url = "https://opusmax.shop"',
                'wire_api = "responses"',
            ]
        ),
        encoding="utf-8",
    )
    (codex_dir / "auth.json").write_text(
        json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "secret"}), encoding="utf-8"
    )
    project = tmp_path / "project"
    memory_dir = project / ".memoryforge"
    memory_dir.mkdir(parents=True)
    (memory_dir / "config.json").write_text(
        json.dumps(
            {
                "subagent": {
                    "runner": "codex",
                    "model": "gpt-5.2",
                    "codex_sync": True,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MEMORYFORGE_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MEMORYFORGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMORYFORGE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMORYFORGE_SUBAGENT_CMD", raising=False)
    monkeypatch.setattr(
        "memoryforge.agents.subagents.shutil.which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    runner = create_subagent_runner("auto", project_root=str(project))

    assert isinstance(runner, CodexSubAgentRunner)
    assert runner.model == "gpt-5.2"


def test_subagent_operator_is_shared_cached_execution_layer():
    runner = CountingRunner()
    operator = SubAgentOperator(subagent=runner)
    plan = {"run_id": "run_1", "query": "Alice"}
    batch = {"batch_index": 0, "chunk_ids": ["rchunk_1"]}
    chunks = [
        {
            "chunk_id": "rchunk_1",
            "source_path": "notes.md",
            "byte_range": {"start": 0, "end": 12},
            "char_range": {"start": 0, "end": 12},
            "content": "Alice owns auth.",
        }
    ]

    first = operator.analyze_rlm_batch(plan=plan, batch=batch, chunks=chunks)
    second = operator.analyze_rlm_batch(plan=plan, batch=batch, chunks=chunks)
    lcm = operator.compact_lcm_context(
        level=1,
        system_prompt="Summarise compactly.",
        user_prompt="Alice owns auth.",
        max_tokens=2000,
        temperature=0.3,
    )

    assert first.kind == "rlm.analyze"
    assert first.cached is False
    assert second.cached is True
    assert second.input_hash == first.input_hash
    assert lcm.kind == "lcm.compact.l1"
    assert runner.complete_count == 2
    assert len(runner.prompts) == 2
    assert "Operation: rlm.analyze" in runner.prompts[0]
    assert "Operation: lcm.compact.l1" in runner.prompts[1]
