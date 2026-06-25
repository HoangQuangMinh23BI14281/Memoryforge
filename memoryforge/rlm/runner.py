"""End-to-end RLM run loop and recursive reduction."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memoryforge.agents import (
    SubAgentOperationResult,
    SubAgentOperator,
    TransientSubAgentRunnerError,
)
from memoryforge.rlm.common import token_estimate


class RLMRunMixin:
    if TYPE_CHECKING:

        def load(
            self,
            agent_id: str,
            value: str | Path,
            *,
            name: str | None = None,
            source_path: str | None = None,
            content_type: str | None = None,
            chunk_size: int = 12_000,
            overlap: int = 1_000,
        ) -> dict[str, Any]: ...

        def dispatch(
            self,
            agent_id: str,
            *,
            buffer_id: str | None = None,
            query: str | None = None,
            limit: int = 20,
            batch_size: int | None = None,
        ) -> dict[str, Any]: ...

        def get_chunk(
            self,
            chunk_id: str,
            *,
            include_content: bool = True,
        ) -> dict[str, Any] | None: ...

        def record_result(
            self,
            agent_id: str,
            run_id: str,
            chunk_ids: list[str],
            analysis: str,
            *,
            batch_index: int | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]: ...

        def aggregate(
            self,
            agent_id: str,
            run_id: str,
            *,
            summary: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]: ...

        def _analyses_for_run(self, agent_id: str, run_id: str) -> list[dict[str, Any]]: ...

        def _link_recursive_parent(
            self,
            *,
            aggregate: dict[str, Any],
            previous_aggregate: dict[str, Any],
            round_index: int,
        ) -> None: ...

    def run(
        self,
        agent_id: str,
        *,
        value: str | Path | None = None,
        name: str | None = None,
        buffer_id: str | None = None,
        query: str | None = None,
        limit: int = 20,
        batch_size: int | None = None,
        chunk_size: int = 12_000,
        overlap: int = 1_000,
        runner: str | None = "auto",
        model: str | None = None,
        base_url: str | None = None,
        project_root: str | None = None,
        timeout_s: float = 900.0,
        max_workers: int = 1,
        max_retries: int = 0,
        allow_partial: bool = False,
        synthesize: bool = True,
        recursive: bool = True,
        max_recursive_rounds: int = 2,
        recursive_token_limit: int | None = None,
    ) -> dict[str, Any]:
        """Run the full RLM loop: load/dispatch, spawn sub-agents, record, and aggregate."""

        loaded = None
        resolved_buffer_id = buffer_id
        if value is not None:
            loaded = self.load(
                agent_id=agent_id,
                value=value,
                name=name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            resolved_buffer_id = loaded["buffer_id"]
        if not resolved_buffer_id:
            raise ValueError("Either value or buffer_id is required")

        operator = SubAgentOperator(
            runner=runner,
            model=model,
            base_url=base_url,
            project_root=project_root,
            timeout_s=timeout_s,
        )

        first_round = self._run_recursive_round(
            agent_id=agent_id,
            buffer_id=resolved_buffer_id,
            query=query,
            limit=limit,
            batch_size=batch_size,
            operator=operator,
            max_workers=max_workers,
            max_retries=max_retries,
            allow_partial=allow_partial,
            synthesize=synthesize,
            round_index=0,
            parent_aggregate=None,
        )
        aggregate = first_round["aggregate"]
        recursive_rounds: list[dict[str, Any]] = []
        loaded_buffers = [loaded] if loaded else []
        resolved_model = _resolved_subagent_model(model, project_root)
        resolved_recursive_limit = recursive_token_limit or _default_recursive_token_limit(
            resolved_model
        )
        max_rounds = max(0, max_recursive_rounds)

        reduction_failed = False

        for round_index in range(1, max_rounds + 1):
            if not recursive or not aggregate:
                break
            aggregate_tokens = token_estimate(str(aggregate.get("summary") or ""))
            if aggregate_tokens <= resolved_recursive_limit:
                break

            is_last_round = round_index >= max_rounds
            if is_last_round and aggregate_tokens > resolved_recursive_limit:
                reduction_failed = True

            recursive_text = self._recursive_round_input(
                aggregate=aggregate,
                round_index=round_index,
                token_limit=resolved_recursive_limit,
            )
            recursive_loaded = self.load(
                agent_id=agent_id,
                value=recursive_text,
                name=f"{name or resolved_buffer_id}:recursive-{round_index}",
                content_type="docs",
                chunk_size=chunk_size,
                overlap=overlap,
            )
            loaded_buffers.append(recursive_loaded)
            next_round = self._run_recursive_round(
                agent_id=agent_id,
                buffer_id=str(recursive_loaded["buffer_id"]),
                query=query,
                limit=limit,
                batch_size=batch_size,
                operator=operator,
                max_workers=max_workers,
                max_retries=max_retries,
                allow_partial=allow_partial,
                synthesize=synthesize,
                round_index=round_index,
                parent_aggregate=aggregate,
            )
            if next_round["aggregate"]:
                self._link_recursive_parent(
                    aggregate=next_round["aggregate"],
                    previous_aggregate=aggregate,
                    round_index=round_index,
                )
            recursive_rounds.append(
                {
                    "round": round_index,
                    "loaded": recursive_loaded,
                    "plan": self._public_plan(next_round["plan"]),
                    "records": next_round["records"],
                    "failures": next_round["failures"],
                    "run_metrics": next_round["metrics"],
                    "aggregate": next_round["aggregate"],
                }
            )
            aggregate = next_round["aggregate"]

        final_tokens = token_estimate(str(aggregate.get("summary") or "")) if aggregate else 0

        if reduction_failed and final_tokens > resolved_recursive_limit:
            raise RuntimeError(
                f"RLM recursive reduction FAILED: After {len(recursive_rounds)} rounds, "
                f"aggregate still has {final_tokens} tokens (limit: {resolved_recursive_limit}). "
                f"Consider increasing max_recursive_rounds or recursive_token_limit. "
                f"Final aggregate may have lost critical chunk details."
            )

        return {
            "loaded": loaded,
            "loaded_buffers": loaded_buffers,
            "plan": self._public_plan(first_round["plan"]),
            "runner": operator.provider,
            "model": operator.model,
            "records": first_round["records"],
            "failures": first_round["failures"],
            "run_metrics": first_round["metrics"],
            "aggregate": aggregate,
            "recursive_rounds": recursive_rounds,
            "recursion": {
                "enabled": recursive,
                "triggered": bool(recursive_rounds),
                "rounds": len(recursive_rounds),
                "max_rounds": max_rounds,
                "token_limit": resolved_recursive_limit,
                "final_tokens": final_tokens,
                "reduction_success": final_tokens <= resolved_recursive_limit
                if aggregate
                else True,
                "stopped_reason": (
                    "max_rounds"
                    if recursive_rounds
                    and final_tokens > resolved_recursive_limit
                    and len(recursive_rounds) >= max_rounds
                    else "under_limit"
                    if aggregate and final_tokens <= resolved_recursive_limit
                    else "no_aggregate"
                    if not aggregate
                    else "disabled"
                ),
            },
            "lossless": True,
        }

    def _run_recursive_round(
        self,
        *,
        agent_id: str,
        buffer_id: str,
        query: str | None,
        limit: int,
        batch_size: int | None,
        operator: SubAgentOperator,
        max_workers: int,
        max_retries: int,
        allow_partial: bool,
        synthesize: bool,
        round_index: int,
        parent_aggregate: dict[str, Any] | None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        plan = self.dispatch(
            agent_id=agent_id,
            buffer_id=buffer_id,
            query=query,
            limit=limit,
            batch_size=batch_size,
        )
        plan["recursive_round"] = round_index
        if parent_aggregate:
            parent_node_id = parent_aggregate.get("summary_node_id")
            plan["parent_aggregate_id"] = parent_aggregate.get("aggregate_id")
            plan["parent_summary_node_id"] = parent_node_id
        resolved_max_workers = _resolve_max_workers(max_workers, len(plan["batches"]))
        resolved_max_retries = _resolve_max_retries(max_retries)
        plan["max_workers"] = resolved_max_workers
        plan["max_retries"] = resolved_max_retries
        plan["allow_partial"] = allow_partial

        analysis_jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        batch_token_estimates: dict[int, int] = {}
        for batch in plan["batches"]:
            chunks = [
                chunk
                for chunk_id in batch["chunk_ids"]
                if (chunk := self.get_chunk(str(chunk_id), include_content=True)) is not None
            ]
            batch_token_estimates[int(batch["batch_index"])] = _chunks_token_estimate(chunks)
            analysis_jobs.append((batch, chunks))

        responses, failures, retry_counts = _run_analysis_jobs(
            operator=operator,
            plan=plan,
            analysis_jobs=analysis_jobs,
            max_workers=resolved_max_workers,
            max_retries=resolved_max_retries,
            allow_partial=allow_partial,
        )

        records = []
        for batch in plan["batches"]:
            response = responses.get(int(batch["batch_index"]))
            if response is None:
                continue
            batch_index = int(batch["batch_index"])
            record = self.record_result(
                agent_id=agent_id,
                run_id=plan["run_id"],
                chunk_ids=list(batch["chunk_ids"]),
                analysis=response.text,
                batch_index=batch_index,
                metadata={
                    "runner": response.provider,
                    "model": response.model,
                    "elapsed_seconds": response.elapsed_seconds,
                    "operation": response.kind,
                    "input_hash": response.input_hash,
                    "cached": response.cached,
                    "retry_count": retry_counts.get(batch_index, 0),
                    "spawned": True,
                    "recursive_round": round_index,
                    "parent_aggregate_id": parent_aggregate.get("aggregate_id")
                    if parent_aggregate
                    else None,
                    "tool_context": "memoryforge-mcp",
                    "input_tokens_estimate": batch_token_estimates.get(batch_index, 0),
                    "output_tokens_estimate": token_estimate(response.text),
                    "reported_input_tokens": response.input_tokens,
                    "reported_output_tokens": response.output_tokens,
                    "reported_total_tokens": response.total_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
            record["runner"] = response.provider
            record["model"] = response.model
            records.append(record)

        metrics = _round_metrics(
            plan=plan,
            records=records,
            failures=failures,
            retry_counts=retry_counts,
            batch_token_estimates=batch_token_estimates,
            elapsed_seconds=time.perf_counter() - started,
        )
        plan["run_metrics"] = metrics

        summary = None
        synthesis_response = None
        if synthesize and records:
            analyses = self._analyses_for_run(agent_id, plan["run_id"])
            synthesis_response = operator.synthesize_rlm_analyses(plan=plan, analyses=analyses)
            summary = synthesis_response.text
        metrics = _with_synthesis_metrics(metrics, synthesis_response)
        plan["run_metrics"] = metrics

        parent_node_id = parent_aggregate.get("summary_node_id") if parent_aggregate else None
        aggregate = (
            self.aggregate(
                agent_id=agent_id,
                run_id=plan["run_id"],
                summary=summary,
                metadata={
                    "runner": synthesis_response.provider
                    if synthesis_response
                    else operator.provider,
                    "model": synthesis_response.model if synthesis_response else operator.model,
                    "operation": synthesis_response.kind
                    if synthesis_response
                    else "rlm.aggregate.deterministic",
                    "input_hash": synthesis_response.input_hash if synthesis_response else None,
                    "cached": synthesis_response.cached if synthesis_response else False,
                    "elapsed_seconds": synthesis_response.elapsed_seconds
                    if synthesis_response
                    else 0.0,
                    "output_tokens_estimate": token_estimate(synthesis_response.text)
                    if synthesis_response
                    else 0,
                    "reported_input_tokens": synthesis_response.input_tokens
                    if synthesis_response
                    else None,
                    "reported_output_tokens": synthesis_response.output_tokens
                    if synthesis_response
                    else None,
                    "reported_total_tokens": synthesis_response.total_tokens
                    if synthesis_response
                    else None,
                    "cost_usd": synthesis_response.cost_usd if synthesis_response else None,
                    "synthesis_spawned": synthesis_response is not None,
                    "recursive_round": round_index,
                    "parent_aggregate_id": parent_aggregate.get("aggregate_id")
                    if parent_aggregate
                    else None,
                    "parent_summary_node_id": parent_node_id,
                    "run_metrics": metrics,
                },
            )
            if records
            else None
        )
        return {"plan": plan, "records": records, "failures": failures, "metrics": metrics, "aggregate": aggregate}

    @staticmethod
    def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": plan["run_id"],
            "agent_id": plan["agent_id"],
            "buffer_id": plan["buffer_id"],
            "query": plan["query"],
            "retrieval_mode": plan.get(
                "retrieval_mode", "search" if plan.get("query") else "full_scan"
            ),
            "chunk_count": plan["chunk_count"],
            "batch_count": plan["batch_count"],
            "batch_size": plan["batch_size"],
            "max_workers": plan.get("max_workers", 1),
            "max_retries": plan.get("max_retries", 0),
            "allow_partial": bool(plan.get("allow_partial", False)),
            "run_metrics": plan.get("run_metrics"),
            "lossless": plan["lossless"],
            "recursive_round": plan.get("recursive_round", 0),
            "parent_aggregate_id": plan.get("parent_aggregate_id"),
            "parent_summary_node_id": plan.get("parent_summary_node_id"),
        }

    @staticmethod
    def _recursive_round_input(
        *,
        aggregate: dict[str, Any],
        round_index: int,
        token_limit: int,
    ) -> str:
        return "\n".join(
            [
                f"# Recursive RLM round {round_index}",
                "",
                f"Parent aggregate: {aggregate.get('aggregate_id')}",
                f"Parent LCM summary node: {aggregate.get('summary_node_id')}",
                f"Target token budget: <= {token_limit}",
                "",
                "Raw source chunk refs remain authoritative:",
                ", ".join(str(chunk_id) for chunk_id in aggregate.get("source_chunk_ids", [])),
                "",
                "Task: reduce this aggregate further without dropping cited refs. If a claim needs detail, keep the raw ref.",
                "",
                "## Parent aggregate summary",
                str(aggregate.get("summary") or ""),
            ]
        ).strip()


def _run_analysis_jobs(
    *,
    operator: SubAgentOperator,
    plan: dict[str, Any],
    analysis_jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    max_workers: int,
    max_retries: int,
    allow_partial: bool,
) -> tuple[dict[int, SubAgentOperationResult], list[dict[str, Any]], dict[int, int]]:
    responses: dict[int, SubAgentOperationResult] = {}
    failures: list[dict[str, Any]] = []
    retry_counts: dict[int, int] = {}
    if max_workers <= 1 or len(analysis_jobs) <= 1:
        for batch, chunks in analysis_jobs:
            try:
                response, retry_count = _execute_analysis_job(
                    operator=operator,
                    plan=plan,
                    batch=batch,
                    chunks=chunks,
                    max_retries=max_retries,
                )
                batch_index = int(batch["batch_index"])
                responses[batch_index] = response
                retry_counts[batch_index] = retry_count
            except Exception as exc:
                if not allow_partial:
                    raise
                failures.append(_analysis_failure(batch, exc))
        return responses, failures, retry_counts

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _execute_analysis_job,
                operator=operator,
                plan=plan,
                batch=batch,
                chunks=chunks,
                max_retries=max_retries,
            ): batch
            for batch, chunks in analysis_jobs
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                response, retry_count = future.result()
                batch_index = int(batch["batch_index"])
                responses[batch_index] = response
                retry_counts[batch_index] = retry_count
            except Exception as exc:
                if not allow_partial:
                    raise
                failures.append(_analysis_failure(batch, exc))
    failures.sort(key=lambda failure: int(failure["batch_index"]))
    return responses, failures, retry_counts


def _resolve_max_workers(max_workers: int, batch_count: int) -> int:
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}")
    return min(max_workers, max(1, batch_count))


def _resolve_max_retries(max_retries: int) -> int:
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")
    return max_retries


def _execute_analysis_job(
    *,
    operator: SubAgentOperator,
    plan: dict[str, Any],
    batch: dict[str, Any],
    chunks: list[dict[str, Any]],
    max_retries: int,
) -> tuple[SubAgentOperationResult, int]:
    attempts = 0
    while True:
        try:
            return (
                operator.analyze_rlm_batch(
                    plan=plan,
                    batch=batch,
                    chunks=chunks,
                ),
                attempts,
            )
        except Exception as exc:
            if attempts >= max_retries or not _is_transient_analysis_failure(exc):
                raise
            attempts += 1


def _is_transient_analysis_failure(exc: Exception) -> bool:
    return isinstance(exc, TransientSubAgentRunnerError | TimeoutError | ConnectionError)


def _analysis_failure(batch: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "batch_index": int(batch["batch_index"]),
        "chunk_ids": [str(chunk_id) for chunk_id in batch.get("chunk_ids", [])],
        "refs": [str(ref) for ref in batch.get("refs", [])],
        "error_type": type(exc).__name__,
        "error": str(exc),
        "transient": _is_transient_analysis_failure(exc),
    }


def _round_metrics(
    *,
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    retry_counts: dict[int, int],
    batch_token_estimates: dict[int, int],
    elapsed_seconds: float,
) -> dict[str, Any]:
    requested_chunk_ids = _ordered_unique(
        str(chunk_id) for batch in plan["batches"] for chunk_id in batch["chunk_ids"]
    )
    covered_chunk_ids = _ordered_unique(
        str(chunk_id) for record in records for chunk_id in record["chunk_ids"]
    )
    failed_chunk_ids = _ordered_unique(
        str(chunk_id) for failure in failures for chunk_id in failure["chunk_ids"]
    )
    requested_count = len(requested_chunk_ids)
    model_elapsed_seconds = sum(
        float((record.get("metadata") or {}).get("elapsed_seconds") or 0.0)
        for record in records
    )
    analysis_output_tokens = sum(
        int((record.get("metadata") or {}).get("output_tokens_estimate") or 0)
        for record in records
    )
    reported_input_tokens = _sum_optional_int(
        (record.get("metadata") or {}).get("reported_input_tokens") for record in records
    )
    reported_output_tokens = _sum_optional_int(
        (record.get("metadata") or {}).get("reported_output_tokens") for record in records
    )
    reported_total_tokens = _sum_optional_int(
        (record.get("metadata") or {}).get("reported_total_tokens") for record in records
    )
    cost_usd = _sum_optional_float(
        (record.get("metadata") or {}).get("cost_usd") for record in records
    )
    successful_batch_indexes = {
        int((record.get("metadata") or {}).get("batch_index", -1)) for record in records
    }
    analysis_input_tokens = sum(
        token_count
        for batch_index, token_count in batch_token_estimates.items()
        if batch_index in successful_batch_indexes
    )
    return {
        "complete": not failures,
        "partial": bool(failures),
        "allow_partial": bool(plan.get("allow_partial", False)),
        "requested_batch_count": int(plan["batch_count"]),
        "succeeded_batch_count": len(records),
        "failed_batch_count": len(failures),
        "retry_count": sum(retry_counts.values()),
        "requested_chunk_count": requested_count,
        "covered_chunk_count": len(covered_chunk_ids),
        "failed_chunk_count": len(failed_chunk_ids),
        "chunk_coverage_ratio": (
            round(len(covered_chunk_ids) / requested_count, 6) if requested_count else 1.0
        ),
        "analysis_input_tokens": analysis_input_tokens,
        "analysis_output_tokens_estimate": analysis_output_tokens,
        "analysis_total_tokens_estimate": analysis_input_tokens + analysis_output_tokens,
        "synthesis_output_tokens_estimate": 0,
        "total_tokens_estimate": analysis_input_tokens + analysis_output_tokens,
        "reported_input_tokens": reported_input_tokens,
        "reported_output_tokens": reported_output_tokens,
        "reported_total_tokens": reported_total_tokens,
        "cost_usd": cost_usd,
        "cost_source": "runner_reported" if cost_usd is not None else "not_reported_by_runner",
        "model_elapsed_seconds": round(model_elapsed_seconds, 6),
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _with_synthesis_metrics(
    metrics: dict[str, Any],
    synthesis_response: SubAgentOperationResult | None,
) -> dict[str, Any]:
    updated = dict(metrics)
    if synthesis_response is None:
        return updated
    synthesis_output_tokens = token_estimate(synthesis_response.text)
    updated["synthesis_output_tokens_estimate"] = synthesis_output_tokens
    updated["total_tokens_estimate"] = (
        int(updated.get("analysis_total_tokens_estimate") or 0) + synthesis_output_tokens
    )
    updated["model_elapsed_seconds"] = round(
        float(updated.get("model_elapsed_seconds") or 0.0)
        + float(synthesis_response.elapsed_seconds or 0.0),
        6,
    )
    updated["reported_input_tokens"] = _add_optional_int(
        updated.get("reported_input_tokens"), synthesis_response.input_tokens
    )
    updated["reported_output_tokens"] = _add_optional_int(
        updated.get("reported_output_tokens"), synthesis_response.output_tokens
    )
    updated["reported_total_tokens"] = _add_optional_int(
        updated.get("reported_total_tokens"), synthesis_response.total_tokens
    )
    updated["cost_usd"] = _add_optional_float(updated.get("cost_usd"), synthesis_response.cost_usd)
    updated["cost_source"] = (
        "runner_reported" if updated["cost_usd"] is not None else "not_reported_by_runner"
    )
    return updated


def _chunks_token_estimate(chunks: list[dict[str, Any]]) -> int:
    total = 0
    for chunk in chunks:
        token_count = chunk.get("token_count")
        if isinstance(token_count, int):
            total += token_count
        else:
            total += token_estimate(str(chunk.get("content") or ""))
    return total


def _sum_optional_int(values: Any) -> int | None:
    total = 0
    seen = False
    for value in values:
        if value is None:
            continue
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
        seen = True
    return total if seen else None


def _sum_optional_float(values: Any) -> float | None:
    total = 0.0
    seen = False
    for value in values:
        if value is None:
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
        seen = True
    return round(total, 8) if seen else None


def _add_optional_int(left: Any, right: Any) -> int | None:
    values = [left, right]
    return _sum_optional_int(values)


def _add_optional_float(left: Any, right: Any) -> float | None:
    values = [left, right]
    return _sum_optional_float(values)


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolved_subagent_model(model: str | None, project_root: str | None) -> str | None:
    if model:
        return model
    if os.environ.get("MEMORYFORGE_SUBAGENT_MODEL"):
        return os.environ["MEMORYFORGE_SUBAGENT_MODEL"]
    try:
        from memoryforge.agents.codex_sync import project_subagent_config
    except Exception:
        project_config: dict[str, Any] = {}
    else:
        project_config = project_subagent_config(project_root)
    configured_model = project_config.get("model")
    if isinstance(configured_model, str) and configured_model:
        return configured_model
    return os.environ.get("MEMORYFORGE_MODEL") or os.environ.get("OPENAI_MODEL")


def _default_recursive_token_limit(model: str | None) -> int:
    normalized = (model or "").lower()
    if normalized.startswith("gpt-5.5"):
        return 64_000
    return 8_000
