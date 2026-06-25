"""Tool-output pruning for context assembly without deleting raw content."""

from __future__ import annotations

from dataclasses import dataclass

from memoryforge.lcm.store import ImmutableMessageStore


@dataclass(frozen=True)
class PruneResult:
    pruned_count: int
    pruned_tokens: int
    candidates_scanned: int
    skipped_recent_messages: int


class ToolOutputPruner:
    def __init__(self, store: ImmutableMessageStore):
        self.store = store

    def prune(
        self,
        session_id: str,
        *,
        protect_recent_messages: int = 2,
        protect_recent_user_turns: int | None = None,
        protect_tokens: int = 40_000,
        min_prunable_tokens: int = 20_000,
    ) -> PruneResult:
        messages = self.store.get_messages(session_id, include_summaries=True)
        if not messages:
            return PruneResult(0, 0, 0, 0)

        recent_user_turn_limit = (
            protect_recent_user_turns
            if protect_recent_user_turns is not None
            else protect_recent_messages
        )
        user_turn_count = 0
        total_tool_tokens = 0
        prunable_tokens = 0
        candidates_scanned = 0
        candidate_ids: list[str] = []

        for message in reversed(messages):
            if message.is_summary:
                break
            if message.role == "user":
                user_turn_count += 1
            if user_turn_count < recent_user_turn_limit:
                continue
            for part in reversed(message.parts):
                if part.part_type != "tool":
                    continue
                candidates_scanned += 1
                if part.compacted_at is not None:
                    break
                if part.is_protected:
                    continue
                if part.tool_state and part.tool_state not in {"completed", "success"}:
                    continue
                total_tool_tokens += part.token_estimate
                if total_tool_tokens > protect_tokens:
                    candidate_ids.append(part.id)
                    prunable_tokens += part.token_estimate

        if prunable_tokens <= min_prunable_tokens:
            return PruneResult(
                0, 0, candidates_scanned, min(user_turn_count, recent_user_turn_limit)
            )
        pruned_count = self.store.mark_parts_compacted(candidate_ids)
        return PruneResult(
            pruned_count,
            prunable_tokens,
            candidates_scanned,
            min(user_turn_count, recent_user_turn_limit),
        )
