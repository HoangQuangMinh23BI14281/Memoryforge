"""LCM context and compaction facade methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memoryforge.lcm import BuiltContext, CompactionRunResult, ContextBudget, LCMCompactionEngine

if TYPE_CHECKING:
    from memoryforge.memory.longterm.models import LongTermRecallResult
    from memoryforge.memory.longterm.store import LongTermMemoryIndex


class LCMFacadeMixin:
    if TYPE_CHECKING:
        db_path: str

        @property
        def lcm_engine(self) -> LCMCompactionEngine: ...

        @property
        def long_term(self) -> LongTermMemoryIndex: ...

        def _inject_long_term_recall(
            self,
            context: BuiltContext,
            recall: list[LongTermRecallResult],
            *,
            content_policy: str = "auto",
            query: str = "",
        ) -> BuiltContext: ...

    def lcm_build_context(
        self,
        session_id: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
    ) -> BuiltContext:
        return self.lcm_engine.builder.build(session_id, system_prompt=system_prompt, budget=budget)

    def lcm_build_context_with_recall(
        self,
        session_id: str,
        agent_id: str,
        query: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        context = self.lcm_build_context(session_id, system_prompt=system_prompt, budget=budget)
        recall = self.long_term.recall(
            agent_id,
            query,
            top_k=top_k,
            include_content=True,
            session_id=session_id,
        )
        return {
            "context": self._inject_long_term_recall(context, recall),
            "long_term_recall": [result.to_dict() for result in recall],
        }

    def lcm_compact_if_needed(
        self,
        agent_id: str,
        session_id: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
        force: bool = False,
        defer_soft: bool = False,
        max_rounds: int = 3,
        runner: str | None = None,
        model: str | None = None,
        project_root: str | None = None,
        base_url: str | None = None,
    ) -> CompactionRunResult:
        owned_engine = any(value is not None for value in (runner, model, project_root, base_url))
        engine = (
            LCMCompactionEngine(
                self.db_path,
                runner=runner,
                model=model,
                project_root=project_root,
                base_url=base_url,
            )
            if owned_engine
            else self.lcm_engine
        )
        try:
            return engine.compact_if_needed(
                agent_id=agent_id,
                session_id=session_id,
                system_prompt=system_prompt,
                budget=budget,
                force=force,
                defer_soft=defer_soft,
                max_rounds=max_rounds,
            )
        finally:
            if owned_engine:
                engine.close()

    def lcm_compact_due(
        self,
        agent_id: str,
        *,
        system_prompt: str = "",
        budget: ContextBudget | None = None,
        hard_only: bool = False,
        limit: int = 20,
        max_rounds: int = 3,
        runner: str | None = None,
        model: str | None = None,
        project_root: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Compact sessions due for maintenance outside the user-question path."""

        owned_engine = any(value is not None for value in (runner, model, project_root, base_url))
        engine = (
            LCMCompactionEngine(
                self.db_path,
                runner=runner,
                model=model,
                project_root=project_root,
                base_url=base_url,
            )
            if owned_engine
            else self.lcm_engine
        )
        try:
            session_ids = [
                str(row[0])
                for row in engine.store.conn.execute(
                    """
                    SELECT id
                    FROM sessions
                    WHERE agent = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (agent_id, max(1, int(limit))),
                ).fetchall()
            ]
            results: list[dict[str, Any]] = []
            for session_id in session_ids:
                decision = engine.assess(
                    session_id,
                    system_prompt=system_prompt,
                    budget=budget,
                )
                due = decision.hard_overflow or (decision.soft_overflow and not hard_only)
                if not due:
                    results.append(
                        {
                            "session_id": session_id,
                            "status": "not_due",
                            "decision": decision.__dict__,
                            "compaction": None,
                        }
                    )
                    continue
                compaction = engine.compact_if_needed(
                    agent_id=agent_id,
                    session_id=session_id,
                    system_prompt=system_prompt,
                    budget=budget,
                    force=False,
                    defer_soft=False,
                    max_rounds=max_rounds,
                )
                results.append(
                    {
                        "session_id": session_id,
                        "status": "compacted" if compaction.triggered else "checked",
                        "decision": decision.__dict__,
                        "compaction": _compaction_payload(compaction),
                    }
                )
            return {
                "agent_id": agent_id,
                "hard_only": hard_only,
                "checked": len(results),
                "compacted": sum(item["status"] == "compacted" for item in results),
                "results": results,
            }
        finally:
            if owned_engine:
                engine.close()

def _compaction_payload(compaction: CompactionRunResult) -> dict[str, Any]:
    return {
        "triggered": compaction.triggered,
        "rounds": compaction.rounds,
        "before_tokens": compaction.before_tokens,
        "after_tokens": compaction.after_tokens,
        "delta_tokens": compaction.delta_tokens,
        "expanded": compaction.expanded,
        "effective": compaction.effective,
        "deferred": compaction.deferred,
        "reason": compaction.reason,
        "cached": compaction.cached,
        "summary_node_ids": compaction.summary_node_ids,
    }

