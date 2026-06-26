"""Build the bounded context sent to an LLM from lossless LCM state."""

from __future__ import annotations

from dataclasses import dataclass

from memoryforge.lcm.store import ImmutableMessageStore, MessagePart, StoredMessage
from memoryforge.lcm.summary import SummaryDAG, SummaryNode
from memoryforge.lcm.tokens import TokenEstimator


@dataclass(frozen=True)
class ContextBudget:
    model_context_limit: int = 200_000
    reserved_output_tokens: int = 4_000
    compaction_buffer: int = 2_000
    soft_threshold_fraction: float = 0.6

    @property
    def hard_limit(self) -> int:
        return max(
            1, self.model_context_limit - self.reserved_output_tokens - self.compaction_buffer
        )

    @property
    def soft_limit(self) -> int:
        return max(1, int(self.hard_limit * self.soft_threshold_fraction))


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str
    source: str
    source_id: str


@dataclass(frozen=True)
class BuiltContext:
    messages: list[LLMMessage]
    token_estimate: int
    budget: ContextBudget
    has_summary: bool
    truncated: bool
    summary_node_ids: list[str]
    raw_message_ids: list[str]
    summary_tokens: int
    raw_tokens: int


class ContextBuilder:
    def __init__(
        self,
        store: ImmutableMessageStore,
        dag: SummaryDAG,
        *,
        estimator: TokenEstimator | None = None,
    ):
        self.store = store
        self.dag = dag
        self.estimator = estimator or TokenEstimator(heuristic_only=True)

    def build(
        self,
        session_id: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
    ) -> BuiltContext:
        resolved_budget = budget or ContextBudget()
        context_items = self.store.get_context_items(session_id)
        if context_items:
            return self._build_from_snapshot(
                context_items,
                system_prompt=system_prompt,
                budget=resolved_budget,
            )
        return self._build_from_legacy_state(
            session_id,
            system_prompt=system_prompt,
            budget=resolved_budget,
        )

    def _build_from_snapshot(
        self,
        context_items: list[tuple[str, str]],
        *,
        system_prompt: str,
        budget: ContextBudget,
    ) -> BuiltContext:
        """Build context from the ordered context-state snapshot."""

        messages: list[LLMMessage] = []
        system_tokens = self.estimator.estimate(system_prompt) if system_prompt else 0
        if system_prompt:
            messages.append(LLMMessage("system", system_prompt, "system", "system"))

        summary_node_ids = [
            item_id for item_type, item_id in context_items if item_type == "summary"
        ]
        summary_nodes_list = [
            node for node_id in summary_node_ids if (node := self.dag.get_node(node_id))
        ]
        summary_nodes = {node.id: node for node in summary_nodes_list}

        summary_tokens_total = 0
        for node in summary_nodes_list:
            full_content = f"[LCM summary node {node.id}]\n{node.content}"
            summary_tokens_total += self.estimator.estimate_cached(
                full_content, f"summary:{node.id}"
            )

        message_overhead = len(summary_nodes_list) * 5
        available_for_raw = max(
            1,
            budget.hard_limit - system_tokens - summary_tokens_total - message_overhead,
        )
        message_ids = [item_id for item_type, item_id in context_items if item_type == "message"]
        candidate_messages = self.store.get_messages_by_ids(message_ids)
        selected_tail, raw_tokens, truncated = self._select_raw_tail(
            candidate_messages,
            max_tokens=available_for_raw,
        )
        selected_by_id = {message.source_id: message for message in selected_tail}

        for item_type, item_id in context_items:
            if item_type == "summary":
                node = summary_nodes.get(item_id)
                if node is not None:
                    messages.append(
                        LLMMessage(
                            "assistant",
                            f"[LCM summary node {node.id}]\n{node.content}",
                            "summary",
                            node.id,
                        )
                    )
            elif item_type == "message":
                selected = selected_by_id.get(item_id)
                if selected is not None:
                    messages.append(selected)

        raw_tokens_with_overhead = raw_tokens + (len(selected_tail) * 5)
        return BuiltContext(
            messages=messages,
            token_estimate=system_tokens
            + summary_tokens_total
            + raw_tokens_with_overhead
            + message_overhead,
            budget=budget,
            has_summary=bool(summary_nodes),
            truncated=truncated,
            summary_node_ids=[node.id for node in summary_nodes.values()],
            raw_message_ids=[
                message.source_id for message in selected_tail if message.source == "message"
            ],
            summary_tokens=summary_tokens_total,
            raw_tokens=raw_tokens,
        )

    def _build_from_legacy_state(
        self,
        session_id: str,
        *,
        system_prompt: str,
        budget: ContextBudget,
    ) -> BuiltContext:
        """Build context from stored messages and summary nodes."""
        raw_messages = self.store.get_messages(session_id, include_summaries=False)
        active_summaries = sorted(
            self.dag.get_active_summaries(session_id), key=lambda node: node.created_at
        )
        covered_until = self._latest_covered_index(raw_messages, active_summaries)
        raw_tail = raw_messages[covered_until + 1 :] if covered_until >= 0 else raw_messages

        messages: list[LLMMessage] = []
        summary_tokens = 0
        if system_prompt:
            system_tokens = self.estimator.estimate(system_prompt)
            messages.append(LLMMessage("system", system_prompt, "system", "system"))
        else:
            system_tokens = 0

        for node in active_summaries:
            text = f"[LCM summary node {node.id}]\n{node.content}"
            token_count = self.estimator.estimate_cached(text, f"summary:{node.id}")
            summary_tokens += token_count
            messages.append(LLMMessage("assistant", text, "summary", node.id))

        # Add message formatting overhead for summaries
        summary_formatting_overhead = len(active_summaries) * 5

        base_tokens = system_tokens + summary_tokens + summary_formatting_overhead
        selected_tail, raw_tokens, truncated = self._select_raw_tail(
            raw_tail,
            max_tokens=max(1, budget.hard_limit - base_tokens),
        )
        messages.extend(selected_tail)

        raw_tokens_with_overhead = raw_tokens + (len(selected_tail) * 5)

        return BuiltContext(
            messages=messages,
            token_estimate=base_tokens + raw_tokens_with_overhead,
            budget=budget,
            has_summary=bool(active_summaries),
            truncated=truncated,
            summary_node_ids=[node.id for node in active_summaries],
            raw_message_ids=[
                message.source_id for message in selected_tail if message.source == "message"
            ],
            summary_tokens=summary_tokens,
            raw_tokens=raw_tokens,
        )

    def _select_raw_tail(
        self,
        raw_messages: list[StoredMessage],
        *,
        max_tokens: int,
    ) -> tuple[list[LLMMessage], int, bool]:
        selected: list[tuple[LLMMessage, int]] = []
        used = 0
        truncated = False
        for message in reversed(raw_messages):
            rendered = self._render_message(message)
            token_count = self.estimator.estimate(rendered.content)
            if selected and used + token_count > max_tokens:
                truncated = True
                break
            if not selected and token_count > max_tokens:
                truncated = True
            selected.append((rendered, token_count))
            used += token_count
            if used >= max_tokens:
                truncated = len(selected) < len(raw_messages)
                break
        selected.reverse()
        return [item[0] for item in selected], used, truncated or len(selected) < len(raw_messages)

    def _render_message(self, message: StoredMessage) -> LLMMessage:
        parts = [self._render_part(part) for part in message.parts]
        return LLMMessage(
            role=message.role,
            content="\n".join(part for part in parts if part),
            source="message",
            source_id=message.id,
        )

    @staticmethod
    def _render_part(part: MessagePart) -> str:
        if part.part_type == "tool":
            tool_name = part.tool_name or "tool"
            if part.compacted_at is not None:
                return f"[Tool output compacted: {tool_name}; original part_id={part.id}]"
            return f"[Tool {tool_name}]\n{part.content}"
        return part.content

    @staticmethod
    def _latest_covered_index(
        raw_messages: list[StoredMessage],
        active_summaries: list[SummaryNode],
    ) -> int:
        if not active_summaries:
            return -1
        index_by_id = {message.id: index for index, message in enumerate(raw_messages)}
        covered = -1
        for node in active_summaries:
            end_index = index_by_id.get(node.span_end)
            if end_index is not None:
                covered = max(covered, end_index)
        return covered
