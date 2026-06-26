"""RLM analysis recording through LCM messages and SummaryDAG."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from memoryforge.lcm.compaction.file_ids import append_lossless_footers
from memoryforge.rlm.common import dedupe, new_id

if TYPE_CHECKING:
    from memoryforge.lcm.store import ImmutableMessageStore
    from memoryforge.lcm.summary import SummaryDAG


class RLMRecordMixin:
    if TYPE_CHECKING:
        conn: sqlite3.Connection

        def _lcm_message_store(self) -> ImmutableMessageStore: ...

        def _summary_dag(self) -> SummaryDAG: ...

    def record_result(
        self,
        agent_id: str,
        run_id: str,
        chunk_ids: list[str],
        analysis: str,
        *,
        batch_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a sub-agent result as an LCM message and SummaryDAG leaf."""

        if not chunk_ids:
            raise ValueError("chunk_ids cannot be empty")
        analysis_id = new_id("rana")
        node_content = self._analysis_node_text(run_id, chunk_ids, analysis)
        source_refs = self._chunk_refs(chunk_ids)
        message_id = self._lcm_message_store().append_text_message(
            agent_id=agent_id,
            session_id=run_id,
            role="assistant",
            content=node_content,
        )
        summary_node_id = self._summary_dag().create_leaf(
            run_id,
            node_content,
            message_id,
            message_id,
            source_refs=source_refs,
        )
        return {
            "analysis_id": analysis_id,
            "run_id": run_id,
            "message_id": message_id,
            "summary_node_id": summary_node_id,
            "chunk_ids": chunk_ids,
            "source_refs": source_refs,
            "metadata": {"batch_index": batch_index, **(metadata or {})},
            "lossless": True,
        }

    def aggregate(
        self,
        agent_id: str,
        run_id: str,
        *,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an LCM SummaryDAG parent over recorded RLM analysis leaves."""

        analyses = self._analyses_for_run(agent_id, run_id)
        if not analyses:
            raise ValueError(f"No RLM analyses found for run_id={run_id}")

        child_node_ids = [item["summary_node_id"] for item in analyses]
        source_chunk_ids = dedupe([chunk_id for item in analyses for chunk_id in item["chunk_ids"]])
        source_refs = self._chunk_refs(source_chunk_ids)

        aggregate_summary = summary or self._deterministic_aggregate_text(run_id, analyses)
        summary_node_id = (
            self._summary_dag().condense(
                child_node_ids,
                aggregate_summary,
                source_refs=source_refs,
            )
            if len(child_node_ids) > 1
            else child_node_ids[0]
        )
        analysis_message_ids = [item["message_id"] for item in analyses if item.get("message_id")]
        self._lcm_message_store().swap_context_items(run_id, analysis_message_ids, summary_node_id)
        aggregate_id = new_id("ragg")
        aggregate_message_id = self._lcm_message_store().append_text_message(
            agent_id=agent_id,
            session_id=run_id,
            role="assistant",
            content=self._aggregate_message_text(aggregate_id, summary_node_id, aggregate_summary),
            is_summary=True,
        )
        return {
            "aggregate_id": aggregate_id,
            "run_id": run_id,
            "agent_id": agent_id,
            "message_id": aggregate_message_id,
            "summary_node_id": summary_node_id,
            "child_node_ids": child_node_ids,
            "source_chunk_ids": source_chunk_ids,
            "source_refs": source_refs,
            "summary": aggregate_summary,
            "metadata": metadata or {},
            "lossless": True,
        }

    def _link_recursive_parent(
        self,
        *,
        aggregate: dict[str, Any],
        previous_aggregate: dict[str, Any],
        round_index: int,
    ) -> None:
        """Link recursive parent with atomic transaction to prevent race conditions."""
        previous_node_id = str(previous_aggregate["summary_node_id"])
        current_node_id = str(aggregate["summary_node_id"])
        child_node_ids = [previous_node_id]
        if current_node_id != previous_node_id:
            child_node_ids.append(current_node_id)
        source_chunk_ids = dedupe(
            [
                *previous_aggregate.get("source_chunk_ids", []),
                *aggregate.get("source_chunk_ids", []),
            ]
        )
        source_refs = self._chunk_refs(source_chunk_ids)
        lcm_store = self._lcm_message_store()

        try:
            recursive_node_id = self._create_recursive_parent_node(
                session_id=str(aggregate["run_id"]),
                child_node_ids=child_node_ids,
                content=str(aggregate.get("summary") or ""),
                source_refs=source_refs,
            )

            metadata = {
                **dict(aggregate.get("metadata") or {}),
                "recursive_round": round_index,
                "recursive_parent_node_ids": child_node_ids,
                "previous_aggregate_id": previous_aggregate.get("aggregate_id"),
            }

            lcm_store.swap_context_items(
                str(aggregate["run_id"]), [current_node_id], recursive_node_id
            )

            aggregate["summary_node_id"] = recursive_node_id
            aggregate["child_node_ids"] = child_node_ids
            aggregate["source_chunk_ids"] = source_chunk_ids
            aggregate["source_refs"] = source_refs
            aggregate["metadata"] = metadata
            aggregate["recursive_parent_node_ids"] = child_node_ids

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(
                f"Failed to link recursive parent for aggregate {aggregate.get('aggregate_id')}: {e}"
            ) from e

    def _create_recursive_parent_node(
        self,
        *,
        session_id: str,
        child_node_ids: list[str],
        content: str,
        source_refs: list[str],
    ) -> str:
        placeholders = ",".join("?" for _node_id in child_node_ids)
        rows = (
            self._summary_dag()
            .conn.execute(
                f"""
            SELECT id, level, span_start_message_id, span_end_message_id, file_ids, source_refs
            FROM summary_nodes
            WHERE id IN ({placeholders})
            ORDER BY created_at
            """,
                child_node_ids,
            )
            .fetchall()
        )
        if len(rows) != len(child_node_ids):
            found = {row[0] for row in rows}
            missing = [node_id for node_id in child_node_ids if node_id not in found]
            raise ValueError(f"Unknown recursive summary nodes: {missing}")

        node_file_ids = dedupe([file_id for row in rows for file_id in json.loads(row[4] or "[]")])
        node_source_refs = dedupe(
            [
                *source_refs,
                *(ref for row in rows for ref in json.loads(row[5] or "[]")),
            ]
        )
        stored_content = append_lossless_footers(
            content,
            file_ids=node_file_ids,
            source_refs=node_source_refs,
        )
        node_id = new_id("sum")
        self._summary_dag().conn.execute(
            """
            INSERT INTO summary_nodes
            (id, session_id, level, span_start_message_id, span_end_message_id,
             content, token_count, created_at, parent_node_ids, file_ids, source_refs,
             superseded, kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                session_id,
                max(int(row[1]) for row in rows) + 1,
                rows[0][2],
                rows[-1][3],
                stored_content,
                len(stored_content) // 4,
                time.time(),
                json.dumps(child_node_ids),
                json.dumps(node_file_ids),
                json.dumps(node_source_refs),
                0,
                "recursive",
            ),
        )
        self._summary_dag().conn.commit()
        return node_id

    def _analyses_for_run(self, agent_id: str, run_id: str) -> list[dict[str, Any]]:
        rows = (
            self._summary_dag()
            .conn.execute(
                """
            SELECT id, span_start_message_id, content, source_refs, created_at
            FROM summary_nodes
            WHERE session_id = ? AND kind = 'leaf'
            ORDER BY created_at
            """,
                (run_id,),
            )
            .fetchall()
        )
        analyses: list[dict[str, Any]] = []
        for row in rows:
            source_refs = json.loads(row[3] or "[]")
            chunk_ids = [
                str(ref).split(":", 1)[1]
                for ref in source_refs
                if str(ref).startswith("rlm_chunk:")
            ]
            if not chunk_ids:
                continue
            analyses.append(
                {
                    "analysis_id": str(row[0]),
                    "batch_index": len(analyses),
                    "chunk_ids": chunk_ids,
                    "analysis": self._analysis_body(str(row[2] or "")),
                    "message_id": str(row[1] or ""),
                    "summary_node_id": str(row[0]),
                    "metadata": {"agent_id": agent_id, "created_at": float(row[4] or 0)},
                }
            )
        return analyses

    @staticmethod
    def _chunk_refs(chunk_ids: list[str]) -> list[str]:
        return [f"rlm_chunk:{chunk_id}" for chunk_id in chunk_ids]

    @staticmethod
    def _analysis_node_text(run_id: str, chunk_ids: list[str], analysis: str) -> str:
        return (
            "# RLM Sub-Agent Analysis\n\n"
            f"Run: {run_id}\n"
            f"Chunks: {', '.join(f'rlm_chunk:{chunk_id}' for chunk_id in chunk_ids)}\n\n"
            f"{analysis.strip()}"
        )

    @staticmethod
    def _analysis_body(node_content: str) -> str:
        sections = node_content.split("\n\n", 2)
        if len(sections) < 3:
            return node_content.strip()
        return sections[2].strip()

    @staticmethod
    def _aggregate_message_text(aggregate_id: str, summary_node_id: str, summary: str) -> str:
        return (
            "# RLM Aggregate Message\n\n"
            f"Aggregate: {aggregate_id}\n"
            f"LCM summary node: {summary_node_id}\n\n"
            f"{summary.strip()}"
        )

    @staticmethod
    def _deterministic_aggregate_text(run_id: str, analyses: list[dict[str, Any]]) -> str:
        sections = [f"# RLM Aggregate\n\nRun: {run_id}\n"]
        for item in analyses:
            chunks = ", ".join(f"rlm_chunk:{chunk_id}" for chunk_id in item["chunk_ids"])
            sections.append(
                f"## Batch {item['batch_index']} — {chunks}\n\n{item['analysis'].strip()}\n"
            )
        return "\n".join(sections).strip()
