"""Shared sub-agent operators used by RLM and LCM."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field, replace
from typing import Any

from memoryforge.agents.runners import BaseSubAgentRunner, create_subagent_runner


@dataclass(frozen=True)
class SubAgentTask:
    """A model-backed operation with explicit provenance and budget metadata."""

    kind: str
    system_prompt: str
    user_prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubAgentOperationResult:
    """Result returned by a shared sub-agent operator."""

    kind: str
    provider: str
    model: str | None
    text: str
    elapsed_seconds: float
    input_hash: str
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class SubAgentOperator:
    """Single execution layer for all model-backed RLM/LCM sub-agent work."""

    def __init__(
        self,
        *,
        runner: str | None = "auto",
        model: str | None = None,
        project_root: str | None = None,
        timeout_s: float = 900.0,
        base_url: str | None = None,
        subagent: BaseSubAgentRunner | None = None,
    ):
        self.runner = runner
        self.model = model
        self.project_root = project_root
        self.timeout_s = timeout_s
        self.base_url = base_url
        self._subagent = subagent
        self._cache: dict[str, SubAgentOperationResult] = {}
        self._cache_lock = threading.Lock()
        self._runner_lock = threading.Lock()

    @property
    def provider(self) -> str:
        return self._subagent.provider if self._subagent is not None else str(self.runner or "auto")

    def execute(self, task: SubAgentTask) -> SubAgentOperationResult:
        input_hash = _task_hash(task)
        prompt = _compose_operator_prompt(task, input_hash)
        cache_key = _hash_text(prompt)
        with self._cache_lock:
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                return replace(cached, cached=True, elapsed_seconds=0.0)

        runner = self._runner()
        response = runner.complete(prompt)
        if response.model:
            with self._runner_lock:
                self.model = response.model
        result = SubAgentOperationResult(
            kind=task.kind,
            provider=response.provider,
            model=response.model,
            text=response.text,
            elapsed_seconds=response.elapsed_seconds,
            input_hash=input_hash,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            cost_usd=response.cost_usd,
        )
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def analyze_rlm_batch(
        self,
        *,
        plan: dict[str, Any],
        batch: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> SubAgentOperationResult:
        return self.execute(
            SubAgentTask(
                kind="rlm.analyze",
                system_prompt=_RLM_ANALYZE_SYSTEM,
                user_prompt=_format_rlm_batch_payload(plan, batch, chunks),
                metadata={
                    "run_id": plan["run_id"],
                    "batch_index": batch["batch_index"],
                    "chunk_ids": list(batch["chunk_ids"]),
                    "query": plan.get("query"),
                    "recursive_round": plan.get("recursive_round", 0),
                    "memoryforge_tools": ["rlm_chunk_get", "rlm_search", "recall_memory"],
                },
            )
        )

    def synthesize_rlm_analyses(
        self,
        *,
        plan: dict[str, Any],
        analyses: list[dict[str, Any]],
    ) -> SubAgentOperationResult:
        return self.execute(
            SubAgentTask(
                kind="rlm.synthesize",
                system_prompt=_RLM_SYNTHESIZE_SYSTEM,
                user_prompt=_format_rlm_synthesis_payload(plan, analyses),
                metadata={
                    "run_id": plan["run_id"],
                    "analysis_count": len(analyses),
                    "query": plan.get("query"),
                    "recursive_round": plan.get("recursive_round", 0),
                    "memoryforge_tools": ["rlm_chunk_get", "rlm_search", "recall_memory"],
                },
            )
        )

    def compact_lcm_context(
        self,
        *,
        level: int,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> SubAgentOperationResult:
        return self.execute(
            SubAgentTask(
                kind=f"lcm.compact.l{level}",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata={"level": level},
            )
        )

    def _runner(self) -> BaseSubAgentRunner:
        with self._runner_lock:
            if self._subagent is None:
                self._subagent = create_subagent_runner(
                    self.runner,
                    model=self.model,
                    project_root=self.project_root,
                    timeout_s=self.timeout_s,
                    base_url=self.base_url,
                )
            return self._subagent


_RLM_ANALYZE_SYSTEM = """Analyze only the chunks included in this operation.

Return concise findings with explicit rlm_chunk:<id> citations.
If the runtime exposes MemoryForge MCP tools, bounded tool calls are allowed to fetch cited refs or nearby chunks.
Do not invent facts outside the chunk payload.
Treat this analysis as a lossy view; exact source remains recoverable through chunk references."""

_RLM_SYNTHESIZE_SYSTEM = """Synthesize sub-agent findings into one concise answer.

Preserve every rlm_chunk reference that supports a finding.
If the synthesis is still too large, keep source refs and make it safe for another recursive RLM round.
Do not replace source chunks; this synthesis is only a DAG parent node."""


def _compose_operator_prompt(task: SubAgentTask, input_hash: str) -> str:
    sections = [
        "# MemoryForge sub-agent operation",
        "",
        f"Operation: {task.kind}",
        f"Input hash: {input_hash}",
    ]
    if task.max_tokens is not None:
        sections.append(f"Output budget: <= {task.max_tokens} tokens")
    if task.temperature is not None:
        sections.append(f"Temperature request: {task.temperature}")
    if task.metadata:
        sections.extend(["", "## Operation metadata", _stable_json(task.metadata)])
    sections.extend(
        [
            "",
            "## Global rules",
            "- Work only from the source payload.",
            "- Preserve source IDs, message IDs, file IDs, chunk IDs, file paths, decisions, constraints, and unresolved questions.",
            "- If details are too large, keep exact references to raw source IDs instead of dropping them.",
            "- If MemoryForge MCP/CLI tools are available, you may use `rlm_chunk_get`, `rlm_search`, and `recall_memory` to rehydrate cited refs; keep tool use bounded and cite every raw ref used.",
            "- Do not invent facts.",
            "",
            "## System task",
            task.system_prompt.strip(),
            "",
            "## Source payload",
            task.user_prompt.strip(),
        ]
    )
    return "\n".join(sections).strip()


def _format_rlm_batch_payload(
    plan: dict[str, Any], batch: dict[str, Any], chunks: list[dict[str, Any]]
) -> str:
    sections = [
        f"Run ID: {plan['run_id']}",
        f"Batch index: {batch['batch_index']}",
        f"Query: {plan.get('query') or '<full-buffer pass>'}",
        "",
    ]
    for chunk in chunks:
        sections.extend(
            [
                f"## rlm_chunk:{chunk['chunk_id']}",
                f"Source: {chunk.get('source_path') or chunk.get('buffer_name')}",
                f"Byte range: {chunk['byte_range']['start']}..{chunk['byte_range']['end']}",
                f"Char range: {chunk['char_range']['start']}..{chunk['char_range']['end']}",
                "",
                str(chunk.get("content") or ""),
                "",
            ]
        )
    return "\n".join(sections).strip()


def _format_rlm_synthesis_payload(plan: dict[str, Any], analyses: list[dict[str, Any]]) -> str:
    sections = [
        f"Run ID: {plan['run_id']}",
        f"Query: {plan.get('query') or '<full-buffer pass>'}",
        "",
    ]
    for item in analyses:
        sections.extend(
            [
                f"## Batch {item['batch_index']} — {', '.join(item['chunk_ids'])}",
                item["analysis"].strip(),
                "",
            ]
        )
    return "\n".join(sections).strip()


def _task_hash(task: SubAgentTask) -> str:
    return _hash_text(
        _stable_json(
            {
                "kind": task.kind,
                "system_prompt": task.system_prompt,
                "user_prompt": task.user_prompt,
                "max_tokens": task.max_tokens,
                "temperature": task.temperature,
                "metadata": task.metadata,
            }
        )
    )


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
