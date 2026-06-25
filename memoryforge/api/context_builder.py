"""Core context bundle assembly and prompt topology contract."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryforge.api.context_assembly import (
    BUNDLE_ACTIVE_RECALL_LIMIT,
    PROMPT_AUDIT_FIELDS,
    ActiveRecallInjectionResult,
    RecallInjectionResult,
    budget_payload,
    context_messages,
    context_with_injected_system_message,
    format_active_recall_block,
    recall_policy_chain,
    resolved_injection_budget,
)
from memoryforge.api.context_bundle import CoreContextBundle
from memoryforge.lcm import BuiltContext, ContextBudget, ImmutableMessageStore, LCMCompactionEngine
from memoryforge.lcm.compaction.file_ids import public_source_refs
from memoryforge.memory.longterm.models import (
    LongTermRecallResult,
    MetadataField,
    metadata_temporal_provenance,
)
from memoryforge.runtime import resolve_runtime_integration

if TYPE_CHECKING:
    from memoryforge.memory.longterm.store import LongTermMemoryIndex


class ContextBuilderMixin:
    if TYPE_CHECKING:
        db_path: str

        @property
        def lcm_engine(self) -> LCMCompactionEngine: ...

        @property
        def lcm_store(self) -> ImmutableMessageStore: ...

        @property
        def long_term(self) -> LongTermMemoryIndex: ...

        def lcm_build_context(
            self,
            session_id: str,
            *,
            system_prompt: str = "",
            budget: ContextBudget | None = None,
        ) -> BuiltContext: ...

        def active_recall(
            self,
            agent_id: str,
            *,
            session_id: str | None = None,
            focus: str | None = None,
            project_root: str | None = None,
            limit: int = 8,
            include_content: bool = False,
        ) -> dict[str, Any]: ...

    def build_core_context_bundle(
        self,
        *,
        agent_id: str,
        session_id: str,
        query: str,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
        top_k: int = 5,
        include_content: bool = False,
        recall_content_policy: str = "snippet",
        long_term_token_budget: int | None = None,
    ) -> CoreContextBundle:
        """Build the context payload that the active core model should read.

        This method performs retrieval and context assembly only. It intentionally
        does not call an answer model.
        """

        started = time.perf_counter()
        context_started = time.perf_counter()
        base_context = self.lcm_build_context(
            session_id,
            system_prompt=system_prompt,
            budget=budget,
        )
        context_build_ms = (time.perf_counter() - context_started) * 1000.0

        recall_started = time.perf_counter()
        recall = self.long_term.recall(
            agent_id=agent_id,
            query=query,
            top_k=top_k,
            include_content=include_content,
            session_id=session_id,
        )
        recall_ms = (time.perf_counter() - recall_started) * 1000.0

        active_started = time.perf_counter()
        active_recall = self.active_recall(
            agent_id,
            session_id=session_id,
            focus=query,
            limit=min(BUNDLE_ACTIVE_RECALL_LIMIT, max(1, top_k)),
            include_content=False,
        )
        active_recall_items = list(active_recall.get("results") or [])
        active_recall_ms = (time.perf_counter() - active_started) * 1000.0

        inject_started = time.perf_counter()
        ltm_injection = self._inject_long_term_recall_budgeted(
            base_context,
            recall,
            content_policy=recall_content_policy,
            query=query,
            token_budget=long_term_token_budget,
        )
        context = ltm_injection.context
        remaining_budget = max(0, context.budget.hard_limit - context.token_estimate)
        active_injection = self._inject_active_recall_budgeted(
            context,
            active_recall_items,
            token_budget=remaining_budget,
        )
        context = active_injection.context
        inject_ms = (time.perf_counter() - inject_started) * 1000.0

        messages = context_messages(context)
        summary_nodes = self._summary_node_payloads(context.summary_node_ids)
        long_term_recall = [result.to_dict() for result in ltm_injection.injected_recall]
        recall_text_diagnostics = self.long_term.recall_injection_diagnostics(
            ltm_injection.injected_recall,
            content_policy=ltm_injection.effective_policy,
            query=query,
        )
        raw_refs = self._bundle_raw_refs(
            context,
            ltm_injection.injected_recall,
            summary_nodes,
            active_injection.injected_items,
        )
        provenance = self._bundle_provenance(
            context,
            ltm_injection.injected_recall,
            summary_nodes,
            active_injection.injected_items,
        )
        diagnostics = self._bundle_diagnostics(
            context=context,
            recall=ltm_injection.injected_recall,
            retrieved_recall_count=len(recall),
            recall_text_diagnostics=recall_text_diagnostics,
            active_recall=active_recall,
            ltm_injection=ltm_injection,
            active_injection=active_injection,
            active_recall_ms=active_recall_ms,
            context_build_ms=context_build_ms,
            recall_ms=recall_ms,
            inject_ms=inject_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            recall_content_policy=recall_content_policy,
        )

        return CoreContextBundle(
            messages=messages,
            active_recall=active_injection.injected_items,
            long_term_recall=long_term_recall,
            summary_nodes=summary_nodes,
            raw_refs=raw_refs,
            token_estimate=context.token_estimate,
            budget=budget_payload(context.budget),
            provenance=provenance,
            diagnostics=diagnostics,
        )

    def build_runtime_context_bundle(
        self,
        *,
        agent_id: str,
        session_id: str,
        query: str,
        project_root: str | Path,
        runtime: str = "auto",
        system_prompt: str = "",
        budget: ContextBudget | None = None,
        top_k: int = 5,
        include_content: bool = False,
        recall_content_policy: str = "snippet",
        long_term_token_budget: int | None = None,
    ) -> dict[str, Any]:
        """Build a context bundle only after validating active-runtime delivery."""

        integration = resolve_runtime_integration(
            project_root, runtime=runtime, expected_db_path=self.db_path
        )
        context_bundle = self.build_core_context_bundle(
            agent_id=agent_id,
            session_id=session_id,
            query=query,
            system_prompt=system_prompt,
            budget=budget,
            top_k=top_k,
            include_content=include_content,
            recall_content_policy=recall_content_policy,
            long_term_token_budget=long_term_token_budget,
        )
        bundle = context_bundle.to_dict()
        bundle["diagnostics"]["runtime"] = integration.diagnostics
        return {
            "runtime": integration.to_dict(),
            "delivery": integration.delivery,
            "model_payload": context_bundle.to_model_payload(),
            "context_bundle": bundle,
        }

    def _inject_long_term_recall(
        self,
        context: BuiltContext,
        recall: list[LongTermRecallResult],
        *,
        content_policy: str = "auto",
        query: str = "",
    ) -> BuiltContext:
        return self._inject_long_term_recall_budgeted(
            context,
            recall,
            content_policy=content_policy,
            query=query,
        ).context

    def _inject_long_term_recall_budgeted(
        self,
        context: BuiltContext,
        recall: list[LongTermRecallResult],
        *,
        content_policy: str = "auto",
        query: str = "",
        token_budget: int | None = None,
    ) -> RecallInjectionResult:
        requested_policy = str(content_policy or "snippet")
        fallback_chain = recall_policy_chain(requested_policy)
        resolved_budget = resolved_injection_budget(context, token_budget)
        if not recall:
            return RecallInjectionResult(
                context=context,
                injected_recall=[],
                requested_policy=requested_policy,
                effective_policy=requested_policy,
                token_budget=resolved_budget,
                token_estimate=0,
                skipped_count=0,
                skipped_reason=None,
                fallback_chain=fallback_chain,
            )
        if resolved_budget <= 0:
            return RecallInjectionResult(
                context=context,
                injected_recall=[],
                requested_policy=requested_policy,
                effective_policy="none",
                token_budget=resolved_budget,
                token_estimate=0,
                skipped_count=len(recall),
                skipped_reason="no_context_headroom",
                fallback_chain=fallback_chain,
            )

        for policy in fallback_chain:
            block = self.long_term.format_recall_block(
                recall,
                content_policy=policy,
                query=query,
            )
            block_tokens = self.lcm_engine.builder.estimator.estimate(block)
            if block_tokens <= resolved_budget:
                return RecallInjectionResult(
                    context=context_with_injected_system_message(
                        context,
                        content=block,
                        source="long_term",
                        source_id="long_term_recall",
                        token_estimate=block_tokens,
                    ),
                    injected_recall=list(recall),
                    requested_policy=requested_policy,
                    effective_policy=policy,
                    token_budget=resolved_budget,
                    token_estimate=block_tokens,
                    skipped_count=0,
                    skipped_reason=None
                    if policy == requested_policy
                    else "policy_fallback",
                    fallback_chain=fallback_chain,
                )

        for count in range(len(recall) - 1, 0, -1):
            selected = recall[:count]
            block = self.long_term.format_recall_block(
                selected,
                content_policy="preview",
                query=query,
            )
            block_tokens = self.lcm_engine.builder.estimator.estimate(block)
            if block_tokens <= resolved_budget:
                return RecallInjectionResult(
                    context=context_with_injected_system_message(
                        context,
                        content=block,
                        source="long_term",
                        source_id="long_term_recall",
                        token_estimate=block_tokens,
                    ),
                    injected_recall=list(selected),
                    requested_policy=requested_policy,
                    effective_policy="preview",
                    token_budget=resolved_budget,
                    token_estimate=block_tokens,
                    skipped_count=len(recall) - count,
                    skipped_reason="token_budget_trimmed",
                    fallback_chain=fallback_chain,
                )

        return RecallInjectionResult(
            context=context,
            injected_recall=[],
            requested_policy=requested_policy,
            effective_policy="none",
            token_budget=resolved_budget,
            token_estimate=0,
            skipped_count=len(recall),
            skipped_reason="token_budget_exhausted",
            fallback_chain=fallback_chain,
        )

    def _inject_active_recall(
        self,
        context: BuiltContext,
        active_recall: list[dict[str, Any]],
    ) -> BuiltContext:
        return self._inject_active_recall_budgeted(context, active_recall).context

    def _inject_active_recall_budgeted(
        self,
        context: BuiltContext,
        active_recall: list[dict[str, Any]],
        *,
        token_budget: int | None = None,
    ) -> ActiveRecallInjectionResult:
        resolved_budget = resolved_injection_budget(context, token_budget)
        if not active_recall:
            return ActiveRecallInjectionResult(
                context=context,
                injected_items=[],
                token_budget=resolved_budget,
                token_estimate=0,
                skipped_count=0,
                skipped_reason=None,
            )
        if resolved_budget <= 0:
            return ActiveRecallInjectionResult(
                context=context,
                injected_items=[],
                token_budget=resolved_budget,
                token_estimate=0,
                skipped_count=len(active_recall),
                skipped_reason="no_context_headroom",
            )

        for count in range(len(active_recall), 0, -1):
            selected = active_recall[:count]
            recall_block = format_active_recall_block(selected)
            active_tokens = self.lcm_engine.builder.estimator.estimate(recall_block)
            if active_tokens <= resolved_budget:
                return ActiveRecallInjectionResult(
                    context=context_with_injected_system_message(
                        context,
                        content=recall_block,
                        source="active_recall",
                        source_id="active_recall",
                        token_estimate=active_tokens,
                    ),
                    injected_items=list(selected),
                    token_budget=resolved_budget,
                    token_estimate=active_tokens,
                    skipped_count=len(active_recall) - count,
                    skipped_reason=None
                    if count == len(active_recall)
                    else "token_budget_trimmed",
                )

        return ActiveRecallInjectionResult(
            context=context,
            injected_items=[],
            token_budget=resolved_budget,
            token_estimate=0,
            skipped_count=len(active_recall),
            skipped_reason="token_budget_exhausted",
        )

    def _summary_node_payloads(self, summary_node_ids: list[str]) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for node_id in summary_node_ids:
            node = self.lcm_engine.dag.get_node(node_id)
            if node is None:
                continue
            payloads.append(
                {
                    "id": node.id,
                    "session_id": node.session_id,
                    "level": node.level,
                    "span_start": node.span_start,
                    "span_end": node.span_end,
                    "parent_node_ids": node.parent_node_ids,
                    "file_ids": node.file_ids,
                    "source_refs": public_source_refs(node.source_refs),
                    "created_at": node.created_at,
                    "direct_evidence": False,
                }
            )
        return payloads

    @staticmethod
    def _bundle_raw_refs(
        context: BuiltContext,
        recall: list[LongTermRecallResult],
        summary_nodes: list[dict[str, Any]],
        active_recall: list[dict[str, Any]],
    ) -> list[str]:
        refs: list[str] = []
        refs.extend(str(item["raw_ref"]) for item in active_recall if item.get("raw_ref"))
        refs.extend(result.raw_ref for result in recall)
        refs.extend(f"message:{message_id}" for message_id in context.raw_message_ids)
        for node in summary_nodes:
            refs.extend(str(ref) for ref in node.get("source_refs", []) if ref)
            span_start = str(node.get("span_start") or "")
            span_end = str(node.get("span_end") or "")
            if span_start:
                refs.append(f"message:{span_start}")
            if span_end and span_end != span_start:
                refs.append(f"message:{span_end}")
        return _dedupe_ordered(refs)

    def _bundle_provenance(
        self,
        context: BuiltContext,
        recall: list[LongTermRecallResult],
        summary_nodes: list[dict[str, Any]],
        active_recall: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        provenance: list[dict[str, Any]] = []
        for item in active_recall:
            metadata = item.get("metadata") or {}
            raw_ref = str(item.get("raw_ref") or "")
            provenance.append(
                {
                    "source_type": item.get("source_type"),
                    "source_id": item.get("source_id"),
                    "item_id": item.get("item_id"),
                    "content_id": item.get("content_id"),
                    "raw_ref": raw_ref,
                    "kind": metadata.get(MetadataField.KIND),
                    "confidence": metadata.get(MetadataField.CONFIDENCE),
                    "freshness": metadata.get(MetadataField.FRESHNESS),
                    "timestamp": metadata_temporal_provenance(metadata),
                    "source_origin": metadata.get(MetadataField.SOURCE_ORIGIN)
                    or metadata.get(MetadataField.SOURCE)
                    or metadata.get(MetadataField.ROLE)
                    or item.get("source_type"),
                    "raw_refs": metadata.get(MetadataField.RAW_REFS) or [raw_ref],
                    "direct_evidence": True,
                    "surface": "active_recall",
                    "active_recall_score": item.get("active_recall_score"),
                    "reasons": item.get("reasons") or [],
                }
            )
        for result in recall:
            metadata = result.metadata or {}
            provenance.append(
                {
                    "source_type": result.source_type,
                    "source_id": result.source_id,
                    "item_id": result.item_id,
                    "content_id": result.content_id,
                    "raw_ref": result.raw_ref,
                    "kind": metadata.get(MetadataField.KIND),
                    "confidence": metadata.get(MetadataField.CONFIDENCE),
                    "freshness": metadata.get(MetadataField.FRESHNESS),
                    "timestamp": metadata_temporal_provenance(metadata),
                    "source_origin": metadata.get(MetadataField.SOURCE_ORIGIN)
                    or metadata.get(MetadataField.SOURCE)
                    or metadata.get(MetadataField.ROLE)
                    or result.source_type,
                    "raw_refs": metadata.get(MetadataField.RAW_REFS) or [result.raw_ref],
                    "direct_evidence": True,
                    "streams": result.streams,
                }
            )
        for node in summary_nodes:
            source_refs = public_source_refs([str(ref) for ref in node.get("source_refs", [])])
            provenance.append(
                {
                    "source_type": "summary",
                    "source_id": node["id"],
                    "raw_ref": f"summary:{node['id']}",
                    "timestamp": node.get("created_at"),
                    "source_origin": "synthesis",
                    "raw_refs": source_refs,
                    "direct_evidence": False,
                    "source_refs": source_refs,
                }
            )
        message_roles = self._message_roles(context.raw_message_ids)
        for message_id in context.raw_message_ids:
            provenance.append(
                {
                    "source_type": "message",
                    "source_id": message_id,
                    "raw_ref": f"message:{message_id}",
                    "source_origin": message_roles.get(message_id, "message"),
                    "raw_refs": [f"message:{message_id}"],
                    "direct_evidence": True,
                }
            )
        return provenance

    def _message_roles(self, message_ids: list[str]) -> dict[str, str]:
        if not message_ids:
            return {}
        placeholders = ",".join("?" for _message_id in message_ids)
        rows = self.lcm_store.conn.execute(
            f"""
            SELECT id, role
            FROM messages
            WHERE id IN ({placeholders})
            """,
            message_ids,
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    @staticmethod
    def _bundle_diagnostics(
        *,
        context: BuiltContext,
        recall: list[LongTermRecallResult],
        retrieved_recall_count: int,
        recall_text_diagnostics: list[dict[str, Any]],
        active_recall: dict[str, Any],
        ltm_injection: RecallInjectionResult,
        active_injection: ActiveRecallInjectionResult,
        active_recall_ms: float,
        context_build_ms: float,
        recall_ms: float,
        inject_ms: float,
        total_ms: float,
        recall_content_policy: str,
    ) -> dict[str, Any]:
        stream_counts: dict[str, int] = {}
        stream_champions: dict[str, str] = {}
        selected_refs: list[str] = []
        active_items = list(active_recall.get("results") or [])
        active_diagnostics = active_recall.get("diagnostics") or {}
        for result in recall:
            selected_refs.append(result.raw_ref)
            for stream_name, detail in result.streams.items():
                stream_counts[stream_name] = stream_counts.get(stream_name, 0) + 1
                if int(detail.get("rank", 0) or 0) == 1:
                    stream_champions[stream_name] = result.raw_ref

        return {
            "bundle_only": True,
            "answer_model_used": False,
            "retrieval": {
                "long_term_count": len(recall),
                "long_term_retrieved_count": retrieved_recall_count,
                "long_term_skipped_count": ltm_injection.skipped_count,
                "stream_counts": stream_counts,
                "stream_champions": stream_champions,
                "selected_refs": selected_refs,
                "recall_text": recall_text_diagnostics,
                "truncated_text_count": sum(
                    1 for item in recall_text_diagnostics if item.get("truncated") is True
                ),
                "fusion": "rrf_rank_fusion",
                "selection": "local_model_free",
                "llm_selection_used": False,
                "llm_rerank_used": False,
            },
            "active_recall": {
                "count": len(active_injection.injected_items),
                "retrieved_count": len(active_items),
                "skipped_count": active_injection.skipped_count,
                "selected_refs": [
                    str(item["raw_ref"])
                    for item in active_injection.injected_items
                    if item.get("raw_ref")
                ],
                "policy": active_diagnostics.get("policy"),
                "query_required": active_diagnostics.get("query_required") is True,
                "answer_model_used": active_diagnostics.get("answer_model_used") is True,
                "semantic_focus_used": active_diagnostics.get("semantic_focus_used") is True,
                "injection": {
                    "token_budget": active_injection.token_budget,
                    "token_estimate": active_injection.token_estimate,
                    "skipped_reason": active_injection.skipped_reason,
                },
                "latency_ms": active_recall_ms,
            },
            "context": {
                "recall_content_policy": recall_content_policy,
                "effective_recall_content_policy": ltm_injection.effective_policy,
                "long_term_recall_injection": {
                    "requested_policy": ltm_injection.requested_policy,
                    "effective_policy": ltm_injection.effective_policy,
                    "token_budget": ltm_injection.token_budget,
                    "token_estimate": ltm_injection.token_estimate,
                    "skipped_count": ltm_injection.skipped_count,
                    "skipped_reason": ltm_injection.skipped_reason,
                    "fallback_chain": ltm_injection.fallback_chain,
                },
                "message_count": len(context.messages),
                "token_estimate": context.token_estimate,
                "has_summary": context.has_summary,
                "truncated": context.truncated,
                "summary_node_ids": context.summary_node_ids,
                "raw_message_ids": context.raw_message_ids,
                "summary_tokens": context.summary_tokens,
                "raw_tokens": context.raw_tokens,
                "sources": [message.source for message in context.messages],
            },
            "render_contract": {
                "model_readable_field": "messages",
                "audit_fields": list(PROMPT_AUDIT_FIELDS),
                "audit_fields_rendered": False,
            },
            "latency_ms": {
                "context_build": context_build_ms,
                "long_term_recall": recall_ms,
                "recall_injection": inject_ms,
                "total": total_ms,
            },
        }


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped

