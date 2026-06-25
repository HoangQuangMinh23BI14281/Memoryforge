"""Sub-agent backed provider for LCM Level 1/Level 2 compaction."""

from __future__ import annotations

from memoryforge.agents import BaseSubAgentRunner, SubAgentOperator


class SubAgentLCMProvider:
    name = "sub-agent"

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
        self._operator = SubAgentOperator(
            runner=runner,
            model=model,
            project_root=project_root,
            timeout_s=timeout_s,
            base_url=base_url,
            subagent=subagent,
        )

    @property
    def provider(self) -> str:
        return self._operator.provider

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = self._operator.compact_lcm_context(
            level=1 if max_tokens >= 1500 else 2,
            system_prompt=(
                "MemoryForge LCM compaction sub-agent.\n"
                "You are a bounded compaction worker. Return only the compacted context.\n\n"
                f"{system_prompt}"
            ),
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response.model:
            self.model = response.model
        return response.text
