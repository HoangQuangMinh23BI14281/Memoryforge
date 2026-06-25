"""Threshold-driven LCM compaction runtime."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass

from memoryforge.lcm.compaction.compactor import LCMCompactor, Message
from memoryforge.lcm.compaction.file_ids import (
    strip_file_ids_footer,
    strip_source_refs_footer,
)
from memoryforge.lcm.compaction.pruner import PruneResult, ToolOutputPruner
from memoryforge.lcm.compaction.subagent_provider import SubAgentLCMProvider
from memoryforge.lcm.context import BuiltContext, ContextBudget, ContextBuilder
from memoryforge.lcm.events import EventBus
from memoryforge.lcm.store import ImmutableMessageStore, StoredMessage
from memoryforge.lcm.summary import SummaryDAG, SummaryNode
from memoryforge.lcm.tokens import TokenEstimator


@dataclass(frozen=True)
class ThresholdDecision:
    token_estimate: int
    soft_limit: int
    hard_limit: int
    soft_overflow: bool
    hard_overflow: bool


@dataclass(frozen=True)
class CompactionRunResult:
    triggered: bool
    rounds: int
    before_tokens: int
    after_tokens: int
    delta_tokens: int
    expanded: bool
    effective: bool
    summary_node_ids: list[str]
    pruned: PruneResult
    decision: ThresholdDecision
    context: BuiltContext
    deferred: bool = False
    reason: str | None = None
    cached: bool = False


@dataclass(frozen=True)
class _SummaryWriteResult:
    node_id: str
    cached: bool


class LCMCompactionEngine:
    _lock_guard = threading.Lock()
    _session_locks: dict[str, threading.Lock] = {}

    def __init__(
        self,
        db_path: str,
        *,
        store: ImmutableMessageStore | None = None,
        dag: SummaryDAG | None = None,
        compactor: LCMCompactor | None = None,
        event_bus: EventBus | None = None,
        estimator: TokenEstimator | None = None,
        runner: str | None = "auto",
        model: str | None = None,
        project_root: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 900.0,
    ):
        self.store = store or ImmutableMessageStore(db_path)
        self.dag = dag or SummaryDAG(db_path)
        self.compactor = compactor or LCMCompactor(
            SubAgentLCMProvider(
                runner=runner,
                model=model,
                project_root=project_root,
                base_url=base_url,
                timeout_s=timeout_s,
            )
        )
        self.event_bus = event_bus or EventBus(db_path)
        self.estimator = estimator or TokenEstimator(heuristic_only=True)
        self.builder = ContextBuilder(self.store, self.dag, estimator=self.estimator)
        self.pruner = ToolOutputPruner(self.store)

    def assess(
        self,
        session_id: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
    ) -> ThresholdDecision:
        context = self.builder.build(session_id, system_prompt=system_prompt, budget=budget)
        return ThresholdDecision(
            token_estimate=context.token_estimate,
            soft_limit=context.budget.soft_limit,
            hard_limit=context.budget.hard_limit,
            soft_overflow=context.token_estimate >= context.budget.soft_limit,
            hard_overflow=context.token_estimate >= context.budget.hard_limit,
        )

    def compact_if_needed(
        self,
        agent_id: str,
        session_id: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
        force: bool = False,
        defer_soft: bool = False,
        max_rounds: int = 3,
        keep_recent_messages: int = 2,
    ) -> CompactionRunResult:
        """Compact session context with session-level locking."""
        resolved_budget = budget or ContextBudget()

        # Acquire session-level advisory lock to prevent concurrent compaction
        lock_acquired = self._acquire_compaction_lock(session_id, timeout_ms=5000)
        if not lock_acquired:
            # Another compaction is running for this session
            # Return current state without triggering
            before_context = self.builder.build(
                session_id, system_prompt=system_prompt, budget=resolved_budget
            )
            decision = self.assess(session_id, system_prompt=system_prompt, budget=resolved_budget)
            return CompactionRunResult(
                triggered=False,
                rounds=0,
                before_tokens=before_context.token_estimate,
                after_tokens=before_context.token_estimate,
                delta_tokens=0,
                expanded=False,
                effective=False,
                summary_node_ids=[],
                pruned=PruneResult(0, 0, 0, 0),
                decision=decision,
                context=before_context,
            )

        try:
            return self._compact_with_lock(
                agent_id,
                session_id,
                system_prompt=system_prompt,
                budget=resolved_budget,
                force=force,
                defer_soft=defer_soft,
                max_rounds=max_rounds,
                keep_recent_messages=keep_recent_messages,
            )
        finally:
            # Always release lock
            self._release_compaction_lock(session_id)

    def _acquire_compaction_lock(self, session_id: str, timeout_ms: int = 5000) -> bool:
        """Acquire an in-process session lock without adding a SQLite table."""

        with self._lock_guard:
            lock = self._session_locks.setdefault(session_id, threading.Lock())
        return lock.acquire(timeout=timeout_ms / 1000)

    def _release_compaction_lock(self, session_id: str) -> None:
        """Release the in-process session lock."""

        lock = self._session_locks.get(session_id)
        if lock is not None and lock.locked():
            lock.release()

    def _compact_with_lock(
        self,
        agent_id: str,
        session_id: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget,
        force: bool = False,
        defer_soft: bool = False,
        max_rounds: int = 3,
        keep_recent_messages: int = 2,
    ) -> CompactionRunResult:
        """Internal compaction logic with lock held."""
        before_context = self.builder.build(session_id, system_prompt=system_prompt, budget=budget)
        decision = self.assess(session_id, system_prompt=system_prompt, budget=budget)
        zero_prune = PruneResult(0, 0, 0, 0)
        if not force and not decision.soft_overflow:
            return CompactionRunResult(
                triggered=False,
                rounds=0,
                before_tokens=before_context.token_estimate,
                after_tokens=before_context.token_estimate,
                delta_tokens=0,
                expanded=False,
                effective=False,
                summary_node_ids=[],
                pruned=zero_prune,
                decision=decision,
                context=before_context,
            )
        if defer_soft and not force and decision.soft_overflow and not decision.hard_overflow:
            self.event_bus.publish_coalesced(
                "lcm.compaction.deferred",
                {
                    "reason": "soft_overflow_deferred",
                    "tokens": before_context.token_estimate,
                    "soft_limit": budget.soft_limit,
                    "hard_limit": budget.hard_limit,
                },
                agent_id=agent_id,
                session_id=session_id,
                coalesce_key=f"soft:{budget.soft_limit}:hard:{budget.hard_limit}",
            )
            return CompactionRunResult(
                triggered=False,
                rounds=0,
                before_tokens=before_context.token_estimate,
                after_tokens=before_context.token_estimate,
                delta_tokens=0,
                expanded=False,
                effective=False,
                summary_node_ids=[],
                pruned=zero_prune,
                decision=decision,
                context=before_context,
                deferred=True,
                reason="soft_overflow_deferred",
            )

        self.event_bus.publish(
            "lcm.compaction.triggered",
            {
                "tokens": before_context.token_estimate,
                "soft_limit": budget.soft_limit,
                "hard_limit": budget.hard_limit,
                "force": force,
            },
            agent_id=agent_id,
            session_id=session_id,
        )
        pruned = self.pruner.prune(session_id)
        created_summary_ids: list[str] = []
        cached_summary_ids: list[str] = []
        rounds = 0

        for round_index in range(max(0, max_rounds)):
            current_context = self.builder.build(
                session_id, system_prompt=system_prompt, budget=budget
            )
            if not force and current_context.token_estimate <= budget.soft_limit:
                break
            candidates = self._compaction_candidates(
                session_id, keep_recent_messages=keep_recent_messages
            )
            if candidates:
                summary_result = self._summarize_messages(agent_id, session_id, candidates)
                if summary_result is None:
                    self.event_bus.publish(
                        "lcm.compaction.skipped",
                        {
                            "reason": "no_context_convergence",
                            "messages": len(candidates),
                        },
                        agent_id=agent_id,
                        session_id=session_id,
                    )
                    break
                node_id = summary_result.node_id
                created_summary_ids.append(node_id)
                if summary_result.cached:
                    cached_summary_ids.append(node_id)
                rounds += 1
                self.event_bus.publish(
                    "lcm.compaction.round",
                    {
                        "round": round_index + 1,
                        "summary_node_id": node_id,
                        "messages": len(candidates),
                        "cached": summary_result.cached,
                    },
                    agent_id=agent_id,
                    session_id=session_id,
                )
                continue
            active_summaries = self.dag.get_active_summaries(session_id)
            if len(active_summaries) > 1:
                condensed_node_id = self._condense_summaries(agent_id, session_id, active_summaries)
                if condensed_node_id is None:
                    self.event_bus.publish(
                        "lcm.compaction.skipped",
                        {
                            "reason": "no_context_convergence",
                            "summaries": len(active_summaries),
                        },
                        agent_id=agent_id,
                        session_id=session_id,
                    )
                    break
                created_summary_ids.append(condensed_node_id)
                rounds += 1
                self.event_bus.publish(
                    "lcm.compaction.condensed",
                    {
                        "round": round_index + 1,
                        "summary_node_id": condensed_node_id,
                        "children": len(active_summaries),
                    },
                    agent_id=agent_id,
                    session_id=session_id,
                )
                continue
            break

        after_context = self.builder.build(session_id, system_prompt=system_prompt, budget=budget)
        delta_tokens = after_context.token_estimate - before_context.token_estimate
        expanded = delta_tokens > 0
        effective = rounds > 0 and delta_tokens < 0
        self.event_bus.publish(
            "lcm.compaction.completed",
            {
                "before_tokens": before_context.token_estimate,
                "after_tokens": after_context.token_estimate,
                "delta_tokens": delta_tokens,
                "expanded": expanded,
                "effective": effective,
                "rounds": rounds,
                "summaries": created_summary_ids,
                "cached_summaries": cached_summary_ids,
                "pruned_count": pruned.pruned_count,
            },
            agent_id=agent_id,
            session_id=session_id,
        )
        return CompactionRunResult(
            triggered=True,
            rounds=rounds,
            before_tokens=before_context.token_estimate,
            after_tokens=after_context.token_estimate,
            delta_tokens=delta_tokens,
            expanded=expanded,
            effective=effective,
            summary_node_ids=created_summary_ids,
            pruned=pruned,
            decision=decision,
            context=after_context,
            cached=bool(cached_summary_ids),
            reason="cached_compaction_reused" if cached_summary_ids else None,
        )

    def _compaction_candidates(
        self,
        session_id: str,
        *,
        keep_recent_messages: int,
    ) -> list[StoredMessage]:
        context_items = self.store.get_context_items(session_id)
        if context_items:
            message_ids = [
                item_id for item_type, item_id in context_items if item_type == "message"
            ]
            messages = self.store.get_messages_by_ids(message_ids)
            if keep_recent_messages and len(messages) > keep_recent_messages:
                return messages[:-keep_recent_messages]
            return []

        raw_messages = self.store.get_messages(session_id, include_summaries=False)
        active_summaries = self.dag.get_active_summaries(session_id)
        covered_until = ContextBuilder._latest_covered_index(raw_messages, active_summaries)
        uncovered = raw_messages[covered_until + 1 :] if covered_until >= 0 else raw_messages
        if keep_recent_messages and len(uncovered) > keep_recent_messages:
            return uncovered[:-keep_recent_messages]
        return []

    def _summarize_messages(
        self,
        agent_id: str,
        session_id: str,
        messages: list[StoredMessage],
    ) -> _SummaryWriteResult | None:
        input_ref = self._compaction_input_ref(messages)
        message_refs = [f"message:{message.id}" for message in messages]
        source_refs = [*message_refs, input_ref]
        cached_node = self.dag.find_by_source_ref(session_id, input_ref)
        if cached_node is not None:
            summary_text = self._summary_body(cached_node.content)
            if self._summary_would_expand_messages(summary_text, messages):
                return None
            if (
                cached_node.span_start == messages[0].id
                and cached_node.span_end == messages[-1].id
            ):
                node_id = cached_node.id
            else:
                node_id = self.dag.create_leaf(
                    session_id,
                    summary_text,
                    messages[0].id,
                    messages[-1].id,
                    file_ids=cached_node.file_ids,
                    source_refs=source_refs,
                )
                self.store.append_text_message(
                    agent_id,
                    session_id,
                    "assistant",
                    summary_text,
                    is_summary=True,
                )
            self.store.swap_context_items(session_id, [message.id for message in messages], node_id)
            self.event_bus.publish(
                "lcm.compaction.cache_hit",
                {
                    "source_summary_node_id": cached_node.id,
                    "summary_node_id": node_id,
                    "messages": len(messages),
                },
                agent_id=agent_id,
                session_id=session_id,
            )
            return _SummaryWriteResult(node_id=node_id, cached=True)

        compacted = self.compactor.compact(
            [
                Message(
                    role=message.role, content=self._message_text(message), message_id=message.id
                )
                for message in messages
            ]
        )
        summary_text = compacted.summary
        cacheable = True
        if self._summary_would_expand_messages(summary_text, messages):
            summary_text = self._reference_summary(messages)
            cacheable = False
            if self._summary_would_expand_messages(summary_text, messages):
                return None
        node_id = self.dag.create_leaf(
            session_id,
            summary_text,
            messages[0].id,
            messages[-1].id,
            file_ids=compacted.file_ids_preserved,
            source_refs=source_refs if cacheable else message_refs,
        )
        self.store.swap_context_items(session_id, [message.id for message in messages], node_id)
        self.store.append_text_message(
            agent_id,
            session_id,
            "assistant",
            summary_text,
            is_summary=True,
        )
        return _SummaryWriteResult(node_id=node_id, cached=False)

    def _condense_summaries(
        self,
        agent_id: str,
        session_id: str,
        summaries: list[SummaryNode],
    ) -> str | None:
        ordered = sorted(summaries, key=lambda node: node.created_at)
        compacted = self.compactor.compact(
            [
                Message(role="assistant", content=node.content, message_id=node.id)
                for node in ordered
            ]
        )
        condensed_summary = compacted.summary
        if self._summary_would_expand_nodes(condensed_summary, ordered):
            condensed_summary = self._reference_condensed_summary(ordered)
            if self._summary_would_expand_nodes(condensed_summary, ordered):
                return None
        node_id = self.dag.condense(
            [node.id for node in ordered],
            condensed_summary,
            file_ids=compacted.file_ids_preserved,
        )
        self.store.swap_context_items(session_id, [node.id for node in ordered], node_id)
        self.store.append_text_message(
            agent_id,
            session_id,
            "assistant",
            condensed_summary,
            is_summary=True,
        )
        return node_id

    def _summary_would_expand_messages(
        self, summary_text: str, messages: list[StoredMessage]
    ) -> bool:
        if len(messages) <= 1:
            return False
        raw_cost = sum(self.estimator.estimate(self._message_text(message)) for message in messages)
        raw_cost += len(messages) * 5
        return self._summary_context_cost(summary_text) >= raw_cost

    def _summary_would_expand_nodes(
        self, summary_text: str, nodes: list[SummaryNode]
    ) -> bool:
        if len(nodes) <= 1:
            return False
        raw_cost = sum(self._summary_context_cost(node.content) for node in nodes)
        return self._summary_context_cost(summary_text) >= raw_cost

    def _summary_context_cost(self, summary_text: str) -> int:
        return self.estimator.estimate(f"[LCM summary node preview]\n{summary_text}") + 5

    @staticmethod
    def _message_text(message: StoredMessage) -> str:
        parts = []
        for part in message.parts:
            if part.part_type == "tool":
                parts.append(f"[Tool {part.tool_name or 'tool'}]\n{part.content}")
            else:
                parts.append(part.content)
        return "\n".join(parts)

    @staticmethod
    def _reference_summary(messages: list[StoredMessage]) -> str:
        return (
            f"[LCM deterministic compaction: {len(messages)} messages. "
            f"Raw sources preserved from {messages[0].id} to {messages[-1].id}.]"
        )

    @staticmethod
    def _reference_condensed_summary(nodes: list[SummaryNode]) -> str:
        return (
            f"[LCM deterministic condensation: {len(nodes)} summary nodes. "
            f"Raw sources preserved from {nodes[0].span_start} to {nodes[-1].span_end}.]"
        )

    @staticmethod
    def _summary_body(summary_text: str) -> str:
        return strip_file_ids_footer(strip_source_refs_footer(summary_text)).strip()

    @staticmethod
    def _compaction_input_ref(messages: list[StoredMessage]) -> str:
        payload = []
        for message in messages:
            payload.append(
                {
                    "role": message.role,
                    "parts": [
                        {
                            "type": part.part_type,
                            "content_id": part.content_id,
                            "token_estimate": part.token_estimate,
                            "tool_name": part.tool_name,
                            "tool_state": part.tool_state,
                            "is_protected": part.is_protected,
                        }
                        for part in message.parts
                    ],
                }
            )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"content:lcm-compaction-input:{hashlib.sha256(encoded).hexdigest()}"

    def close(self) -> None:
        self.store.close()
        self.dag.close()
        self.event_bus.close()
