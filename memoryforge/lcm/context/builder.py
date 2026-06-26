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
        system_message, system_tokens, system_truncated = self._render_system_message(
            system_prompt,
            max_tokens=budget.hard_limit,
        )
        if system_message is not None:
            messages.append(system_message)

        summary_node_ids = [
            item_id for item_type, item_id in context_items if item_type == "summary"
        ]
        summary_nodes_list = [
            node for node_id in summary_node_ids if (node := self.dag.get_node(node_id))
        ]
        summary_messages, summary_tokens_total, summary_truncated = self._render_summary_nodes(
            summary_nodes_list,
            max_tokens=max(0, budget.hard_limit - system_tokens),
        )
        summary_messages_by_id = {message.source_id: message for message in summary_messages}

        message_overhead = len(summary_messages) * 5
        available_for_raw = max(
            0,
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
                selected_summary = summary_messages_by_id.get(item_id)
                if selected_summary is not None:
                    messages.append(selected_summary)
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
            has_summary=bool(summary_messages),
            truncated=system_truncated or summary_truncated or truncated,
            summary_node_ids=[message.source_id for message in summary_messages],
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
        system_message, system_tokens, system_truncated = self._render_system_message(
            system_prompt,
            max_tokens=budget.hard_limit,
        )
        if system_message is not None:
            messages.append(system_message)

        summary_messages, summary_tokens, summary_truncated = self._render_summary_nodes(
            active_summaries,
            max_tokens=max(0, budget.hard_limit - system_tokens),
        )
        messages.extend(summary_messages)

        # Add message formatting overhead for summaries
        summary_formatting_overhead = len(summary_messages) * 5

        base_tokens = system_tokens + summary_tokens + summary_formatting_overhead
        selected_tail, raw_tokens, truncated = self._select_raw_tail(
            raw_tail,
            max_tokens=max(0, budget.hard_limit - base_tokens),
        )
        messages.extend(selected_tail)

        raw_tokens_with_overhead = raw_tokens + (len(selected_tail) * 5)

        return BuiltContext(
            messages=messages,
            token_estimate=base_tokens + raw_tokens_with_overhead,
            budget=budget,
            has_summary=bool(summary_messages),
            truncated=system_truncated or summary_truncated or truncated,
            summary_node_ids=[message.source_id for message in summary_messages],
            raw_message_ids=[
                message.source_id for message in selected_tail if message.source == "message"
            ],
            summary_tokens=summary_tokens,
            raw_tokens=raw_tokens,
        )

    def _render_system_message(
        self,
        system_prompt: str,
        *,
        max_tokens: int,
    ) -> tuple[LLMMessage | None, int, bool]:
        if not system_prompt:
            return None, 0, False
        message = LLMMessage("system", system_prompt, "system", "system")
        token_count = self.estimator.estimate(message.content)
        if token_count <= max_tokens:
            return message, token_count, False
        fitted, fitted_tokens = self._fit_message_to_budget(message, max_tokens=max_tokens)
        return fitted, fitted_tokens, True

    def _render_summary_nodes(
        self,
        nodes: list[SummaryNode],
        *,
        max_tokens: int,
    ) -> tuple[list[LLMMessage], int, bool]:
        rendered: list[LLMMessage] = []
        content_tokens = 0
        context_used = 0
        truncated = False
        for node in nodes:
            full_content = f"[LCM summary node {node.id}]\n{node.content}"
            token_count = self.estimator.estimate_cached(full_content, f"summary:{node.id}")
            context_cost = token_count + 5
            remaining = max_tokens - context_used
            if context_cost > remaining:
                truncated = True
                fitted, fitted_tokens = self._fit_message_to_budget(
                    LLMMessage("assistant", full_content, "summary", node.id),
                    max_tokens=remaining - 5,
                )
                if fitted is not None:
                    rendered.append(fitted)
                    content_tokens += fitted_tokens
                break
            rendered.append(LLMMessage("assistant", full_content, "summary", node.id))
            content_tokens += token_count
            context_used += context_cost
        return rendered, content_tokens, truncated or len(rendered) < len(nodes)

    def _select_raw_tail(
        self,
        raw_messages: list[StoredMessage],
        *,
        max_tokens: int,
    ) -> tuple[list[LLMMessage], int, bool]:
        selected: list[tuple[LLMMessage, int]] = []
        raw_used = 0
        context_used = 0
        truncated = False
        for message in reversed(raw_messages):
            rendered = self._render_message(message)
            token_count = self.estimator.estimate(rendered.content)
            context_cost = token_count + 5
            if context_cost > max_tokens:
                truncated = True
                if not selected:
                    fitted, fitted_tokens = self._fit_message_to_budget(
                        rendered,
                        max_tokens=max_tokens - 5,
                    )
                    if fitted is not None:
                        selected.append((fitted, fitted_tokens))
                        raw_used += fitted_tokens
                break
            if selected and context_used + context_cost > max_tokens:
                truncated = True
                break
            selected.append((rendered, token_count))
            raw_used += token_count
            context_used += context_cost
            if context_used >= max_tokens:
                truncated = len(selected) < len(raw_messages)
                break
        selected.reverse()
        return (
            [item[0] for item in selected],
            raw_used,
            truncated or len(selected) < len(raw_messages),
        )

    def _fit_message_to_budget(
        self,
        message: LLMMessage,
        *,
        max_tokens: int,
    ) -> tuple[LLMMessage | None, int]:
        if max_tokens <= 0:
            return None, 0
        label = "raw message" if message.source == "message" else message.source
        marker = f"\n\n[LCM {label} truncated for context budget; source_id={message.source_id}]"
        marker_tokens = self.estimator.estimate(marker)
        if marker_tokens > max_tokens:
            return None, 0
        max_content_tokens = max(0, max_tokens - marker_tokens)
        max_chars = max_content_tokens * 4
        content = message.content[:max_chars].rstrip() if max_chars else ""
        fitted_content = f"{content}{marker}" if content else marker.strip()
        while fitted_content and self.estimator.estimate(fitted_content) > max_tokens:
            if content:
                content = content[:-16].rstrip()
                fitted_content = f"{content}{marker}" if content else marker.strip()
            else:
                return None, 0
        fitted = LLMMessage(message.role, fitted_content, message.source, message.source_id)
        return fitted, self.estimator.estimate(fitted.content)

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
