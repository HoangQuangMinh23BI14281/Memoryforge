"""Sub-agent execution backends for the RLM pipeline."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from memoryforge.agents.codex_sync import project_subagent_config


class SubAgentRunnerError(RuntimeError):
    """Raised when no sub-agent runner can execute a prompt."""


class TransientSubAgentRunnerError(SubAgentRunnerError):
    """Raised when retrying the same sub-agent operation may succeed."""


_TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class SubAgentResponse:
    provider: str
    model: str | None
    text: str
    elapsed_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class BaseSubAgentRunner:
    """Blocking sub-agent runner interface.

    RLM may execute multiple sync calls concurrently at the batch scheduler
    level, but individual runners expose one complete() call and do not stream.
    """

    provider = "base"

    def __init__(
        self, *, model: str | None = None, project_root: str | None = None, timeout_s: float = 900.0
    ):
        self.model = model
        self.project_root = str(Path(project_root).expanduser().resolve()) if project_root else None
        self.timeout_s = timeout_s

    def complete(self, prompt: str) -> SubAgentResponse:
        raise NotImplementedError


class MockSubAgentRunner(BaseSubAgentRunner):
    provider = "mock"

    def complete(self, prompt: str) -> SubAgentResponse:
        started = time.monotonic()
        token_match = re.search(r"Return only this token:\s*([A-Z0-9_]+)", prompt)
        if token_match:
            text = token_match.group(1)
        else:
            chunk_ids = re.findall(r"##\s+(rlm_chunk:[^\s]+)", prompt)
            refs = " ".join(chunk_ids[:4]) if chunk_ids else "mock-subagent-output"
            text = f"Mock sub-agent result {refs}".strip()
        return SubAgentResponse(self.provider, self.model, text, time.monotonic() - started)


class CommandSubAgentRunner(BaseSubAgentRunner):
    provider = "command"

    def __init__(
        self,
        command: list[str],
        *,
        model: str | None = None,
        project_root: str | None = None,
        timeout_s: float = 900.0,
        provider: str = "command",
    ):
        super().__init__(model=model, project_root=project_root, timeout_s=timeout_s)
        if not command:
            raise SubAgentRunnerError("Sub-agent command is empty")
        self.command = command
        self.provider = provider

    def complete(self, prompt: str) -> SubAgentResponse:
        started = time.monotonic()
        command, stdin_text, cleanup_paths = self._prepare_command(prompt)
        env = os.environ.copy()
        env["MEMORYFORGE_SUBAGENT"] = "1"
        try:
            completed = subprocess.run(
                command,
                input=stdin_text,
                text=True,
                capture_output=True,
                cwd=self.project_root,
                env=env,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientSubAgentRunnerError(
                f"{self.provider} sub-agent timed out after {self.timeout_s} seconds"
            ) from exc
        finally:
            for path in cleanup_paths:
                path.unlink(missing_ok=True)
        if completed.returncode != 0:
            raise SubAgentRunnerError(
                f"{self.provider} sub-agent failed with exit code {completed.returncode}: "
                f"{_compact_preview(completed.stderr or completed.stdout, 1200)}"
            )
        text = completed.stdout.strip()
        if not text:
            text = completed.stderr.strip()
        if not text:
            raise SubAgentRunnerError(f"{self.provider} sub-agent returned empty output")
        return SubAgentResponse(self.provider, self.model, text, time.monotonic() - started)

    def _prepare_command(self, prompt: str) -> tuple[list[str], str | None, list[Path]]:
        cleanup_paths: list[Path] = []
        if any("{prompt_file}" in arg for arg in self.command):
            handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False)
            with handle:
                handle.write(prompt)
            prompt_path = Path(handle.name)
            cleanup_paths.append(prompt_path)
            return (
                [arg.replace("{prompt_file}", str(prompt_path)) for arg in self.command],
                None,
                cleanup_paths,
            )
        return self.command, prompt, cleanup_paths


class CodexSubAgentRunner(BaseSubAgentRunner):
    provider = "codex"

    def __init__(
        self,
        *,
        binary: str,
        model: str | None = None,
        project_root: str | None = None,
        timeout_s: float = 900.0,
    ):
        super().__init__(model=model, project_root=project_root, timeout_s=timeout_s)
        self.binary = binary

    def complete(self, prompt: str) -> SubAgentResponse:
        started = time.monotonic()
        output_handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".md", delete=False
        )
        output_path = Path(output_handle.name)
        output_handle.close()
        command = self._build_command(output_path)
        env = os.environ.copy()
        env["MEMORYFORGE_SUBAGENT"] = "1"
        if self.model:
            env["MEMORYFORGE_MODEL"] = self.model
            env["OPENAI_MODEL"] = self.model
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                env=env,
                timeout=self.timeout_s,
                check=False,
            )
            text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
        except subprocess.TimeoutExpired as exc:
            raise TransientSubAgentRunnerError(
                f"codex sub-agent timed out after {self.timeout_s} seconds"
            ) from exc
        finally:
            output_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            raise SubAgentRunnerError(
                f"codex sub-agent failed with exit code {completed.returncode}: "
                f"{_compact_preview(completed.stderr or completed.stdout, 1200)}"
            )
        text = text or completed.stdout.strip()
        if not text:
            raise SubAgentRunnerError("codex sub-agent returned empty output")
        return SubAgentResponse(self.provider, self.model, text, time.monotonic() - started)

    def _build_command(self, output_path: Path) -> list[str]:
        command = [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
        ]
        if self.project_root:
            command.extend(["-C", self.project_root])
        if self.model:
            command.extend(["--model", self.model, "-c", f"model={json.dumps(self.model)}"])
        command.append("-")
        return command


class OpenAIResponsesRunner(BaseSubAgentRunner):
    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        project_root: str | None = None,
        timeout_s: float = 900.0,
    ):
        super().__init__(model=model, project_root=project_root, timeout_s=timeout_s)
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def complete(self, prompt: str) -> SubAgentResponse:
        started = time.monotonic()
        payload = json.dumps({"model": self.model, "input": prompt}).encode("utf-8")
        headers = _openai_headers(self.api_key)
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_body(exc)
            error_class = (
                TransientSubAgentRunnerError
                if exc.code in _TRANSIENT_HTTP_STATUS_CODES
                else SubAgentRunnerError
            )
            raise error_class(
                f"openai sub-agent request failed: HTTP {exc.code} {exc.reason}. {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            error_class = (
                TransientSubAgentRunnerError
                if _is_transient_url_error(exc)
                else SubAgentRunnerError
            )
            raise error_class(f"openai sub-agent request failed: {exc}") from exc
        data = json.loads(body)
        text = _extract_openai_text(data)
        if not text:
            raise SubAgentRunnerError("openai sub-agent returned empty output")
        usage = _extract_usage(data)
        return SubAgentResponse(
            self.provider,
            self.model,
            text,
            time.monotonic() - started,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
        )


def create_subagent_runner(
    runner: str | None = None,
    *,
    model: str | None = None,
    project_root: str | None = None,
    timeout_s: float = 900.0,
    base_url: str | None = None,
) -> BaseSubAgentRunner:
    requested = (
        os.environ.get("MEMORYFORGE_SUBAGENT_RUNNER") if runner in {None, "auto"} else runner
    ) or "auto"
    requested = requested.lower()
    project_config = project_subagent_config(project_root)
    resolved_model = (
        model
        or os.environ.get("MEMORYFORGE_SUBAGENT_MODEL")
        or _config_str(project_config, "model")
        or os.environ.get("MEMORYFORGE_MODEL")
        or os.environ.get("OPENAI_MODEL")
    )
    if requested == "auto" and _config_str(project_config, "runner") not in {None, "", "auto"}:
        requested = str(project_config["runner"]).lower()

    if requested == "mock":
        return MockSubAgentRunner(model=resolved_model, project_root=project_root, timeout_s=timeout_s)
    if requested == "command":
        return _command_runner(resolved_model, project_root, timeout_s)
    if requested == "codex":
        _require_explicit_model("codex", resolved_model)
        return _codex_runner(resolved_model, project_root, timeout_s)
    if requested == "openai":
        raise SubAgentRunnerError(
            "OpenAI sub-agent runner is disabled in this build. Use --runner codex or MEMORYFORGE_SUBAGENT_RUNNER=codex."
        )
    if requested != "auto":
        raise SubAgentRunnerError(f"Unknown sub-agent runner: {runner}")

    command = os.environ.get("MEMORYFORGE_SUBAGENT_CMD")
    if command:
        return _command_runner(resolved_model, project_root, timeout_s)
    if shutil.which("codex") and resolved_model:
        return _codex_runner(resolved_model, project_root, timeout_s)
    if shutil.which("codex"):
        raise SubAgentRunnerError(
            "Sub-agent CLI found, but no explicit model was provided. Set --model or MEMORYFORGE_MODEL; "
            "MemoryForge will not silently use Codex defaults."
        )
    raise SubAgentRunnerError(
        "No sub-agent runner found. Install Codex CLI, set MEMORYFORGE_MODEL or MEMORYFORGE_SUBAGENT_MODEL, "
        "or set MEMORYFORGE_SUBAGENT_CMD."
    )


def _command_runner(
    model: str | None, project_root: str | None, timeout_s: float
) -> CommandSubAgentRunner:
    command = os.environ.get("MEMORYFORGE_SUBAGENT_CMD")
    if not command:
        raise SubAgentRunnerError("MEMORYFORGE_SUBAGENT_CMD is required for --runner command")
    return CommandSubAgentRunner(
        shlex.split(command), model=model, project_root=project_root, timeout_s=timeout_s
    )


def _require_explicit_model(provider: str, model: str | None) -> None:
    if not model:
        raise SubAgentRunnerError(
            f"{provider} runner requires --model or MEMORYFORGE_MODEL; MemoryForge will not silently use CLI defaults."
        )


def _codex_runner(
    model: str | None, project_root: str | None, timeout_s: float
) -> CodexSubAgentRunner:
    binary = shutil.which("codex")
    if not binary:
        raise SubAgentRunnerError("codex CLI not found on PATH")
    return CodexSubAgentRunner(
        binary=binary, model=model, project_root=project_root, timeout_s=timeout_s
    )


def _openai_runner(
    model: str | None,
    project_root: str | None,
    timeout_s: float,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenAIResponsesRunner:
    resolved_api_key = (
        api_key or os.environ.get("MEMORYFORGE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    resolved_base_url = _resolved_base_url(base_url)
    if not model:
        raise SubAgentRunnerError("MEMORYFORGE_MODEL or --model is required for --runner openai")
    if not resolved_api_key and _is_official_openai_url(resolved_base_url):
        raise SubAgentRunnerError("OPENAI_API_KEY is required for official OpenAI --runner openai")
    return OpenAIResponsesRunner(
        api_key=resolved_api_key,
        model=model,
        base_url=resolved_base_url,
        project_root=project_root,
        timeout_s=timeout_s,
    )


def _resolved_base_url(base_url: str | None = None) -> str:
    return (
        base_url
        or os.environ.get("MEMORYFORGE_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")


def _has_explicit_base_url(base_url: str | None = None) -> bool:
    return bool(
        base_url
        or os.environ.get("MEMORYFORGE_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )


def _is_official_openai_url(base_url: str) -> bool:
    return base_url.rstrip("/") == "https://api.openai.com/v1"


def _config_str(config: dict[str, object], key: str) -> str | None:
    value = config.get(key)
    return value if isinstance(value, str) and value else None


def _openai_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MemoryForge/6.0",
    }
    extra_headers = os.environ.get("MEMORYFORGE_OPENAI_HEADERS")
    if extra_headers:
        try:
            decoded = json.loads(extra_headers)
        except json.JSONDecodeError as exc:
            raise SubAgentRunnerError("MEMORYFORGE_OPENAI_HEADERS must be a JSON object") from exc
        if not isinstance(decoded, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in decoded.items()
        ):
            raise SubAgentRunnerError(
                "MEMORYFORGE_OPENAI_HEADERS must be a JSON object with string keys and values"
            )
        headers.update({str(key): str(value) for key, value in decoded.items()})
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _read_http_error_body(exc: urllib.error.HTTPError, limit: int = 1200) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    compact = _compact_preview(body, limit)
    if compact:
        return f"Response body: {compact}"
    return "Set MEMORYFORGE_OPENAI_API_KEY or OPENAI_API_KEY if the proxy requires Authorization."


def _is_transient_url_error(exc: urllib.error.URLError) -> bool:
    reason = getattr(exc, "reason", None)
    return isinstance(reason, TimeoutError | ConnectionError)


def _extract_openai_text(data: dict[str, object]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    parts: list[str] = []
    output = data.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content_items = item.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip()


def _extract_usage(data: dict[str, object]) -> dict[str, int]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        key: int(value)
        for key, value in {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }.items()
        if isinstance(value, int)
    }


def _extract_chunk_ids(text: str) -> list[str]:
    seen: set[str] = set()
    chunk_ids: list[str] = []
    for clean in re.findall(r"(?:rlm_chunk:)?(rchunk_[A-Za-z0-9_]+)", text):
        if clean not in seen:
            seen.add(clean)
            chunk_ids.append(clean)
    return chunk_ids


def _compact_preview(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
