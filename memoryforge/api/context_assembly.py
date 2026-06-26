"""Context assembly helpers for core-model payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memoryforge.lcm import BuiltContext, ContextBudget, LLMMessage
from memoryforge.memory.longterm.models import LongTermRecallResult, MetadataField

BUNDLE_ACTIVE_RECALL_LIMIT = 3
ACTIVE_RECALL_PREVIEW_CHARS = 240
PROMPT_AUDIT_FIELDS = (
    "active_recall",
    "long_term_recall",
    "summary_nodes",
    "raw_refs",
    "token_estimate",
    "budget",
    "provenance",
    "diagnostics",
)


@dataclass(frozen=True)
class RecallInjectionResult:
    context: BuiltContext
    injected_recall: list[LongTermRecallResult]
    requested_policy: str
    effective_policy: str
    token_budget: int
    token_estimate: int
    skipped_count: int
    skipped_reason: str | None
    fallback_chain: list[str]


@dataclass(frozen=True)
class ActiveRecallInjectionResult:
    context: BuiltContext
    injected_items: list[dict[str, Any]]
    token_budget: int
    token_estimate: int
    skipped_count: int
    skipped_reason: str | None


def resolved_injection_budget(context: BuiltContext, requested_budget: int | None) -> int:
    headroom = max(0, context.budget.hard_limit - context.token_estimate)
    if requested_budget is None:
        return headroom
    return max(0, min(int(requested_budget), headroom))


def context_messages(context: BuiltContext) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.role),
            "source": str(message.source),
            "source_id": str(message.source_id),
            "content": str(message.content),
        }
        for message in context.messages
    ]


def budget_payload(budget: ContextBudget) -> dict[str, Any]:
    return {
        "model_context_limit": budget.model_context_limit,
        "reserved_output_tokens": budget.reserved_output_tokens,
        "compaction_buffer": budget.compaction_buffer,
        "soft_threshold_fraction": budget.soft_threshold_fraction,
        "hard_limit": budget.hard_limit,
        "soft_limit": budget.soft_limit,
    }


def context_with_injected_system_message(
    context: BuiltContext,
    *,
    content: str,
    source: str,
    source_id: str,
    token_estimate: int,
) -> BuiltContext:
    recall_message = LLMMessage(
        role="system",
        content=content,
        source=source,
        source_id=source_id,
    )
    messages = list(context.messages)
    insert_at = 1 if messages and messages[0].source == "system" else 0
    messages.insert(insert_at, recall_message)
    return BuiltContext(
        messages=messages,
        token_estimate=context.token_estimate + token_estimate,
        budget=context.budget,
        has_summary=context.has_summary,
        truncated=context.truncated,
        summary_node_ids=context.summary_node_ids,
        raw_message_ids=context.raw_message_ids,
        summary_tokens=context.summary_tokens + token_estimate,
        raw_tokens=context.raw_tokens,
    )


def recall_policy_chain(requested_policy: str) -> list[str]:
    policy = requested_policy or "snippet"
    if policy == "full":
        return ["full", "snippet", "preview"]
    if policy in {"auto", "champion"}:
        return [policy, "snippet", "preview"]
    if policy == "snippet":
        return ["snippet", "preview"]
    if policy == "preview":
        return ["preview"]
    return ["snippet", "preview"]


def format_active_recall_block(active_recall: list[dict[str, Any]]) -> str:
    if not active_recall:
        return "[MemoryForge active recall]\nNo active recall."
    lines = ["[MemoryForge active recall: proactive durable memory for the core model]"]
    for index, item in enumerate(active_recall, start=1):
        metadata = item.get("metadata") or {}
        preview = str(item.get("preview") or "").strip()
        if len(preview) > ACTIVE_RECALL_PREVIEW_CHARS:
            preview = f"{preview[: ACTIVE_RECALL_PREVIEW_CHARS].rstrip()} ..."
        reasons = ", ".join(str(reason) for reason in item.get("reasons") or [])
        source_type = item.get("source_type") or "memory"
        confidence = metadata.get(MetadataField.CONFIDENCE) or "unknown"
        lines.append(
            f"{index}. {item.get('raw_ref')} "
            f"(source={source_type}, confidence={confidence}, reasons={reasons})\n"
            f"   {preview}"
        )
    return "\n".join(lines)


def active_recall_reason(source_type: str) -> str:
    if source_type in {"correction", "contradiction"}:
        return source_type
    return "recent_evidence"
