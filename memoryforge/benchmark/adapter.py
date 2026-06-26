"""Unified benchmark adapter interface."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class BenchmarkCase:
    case_id: str
    question: str
    answer: str | None = None
    context: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class BenchmarkResult:
    case_id: str
    prediction: str
    expected: str | None
    latency_ms: float
    correct: bool | None


class Searcher(Protocol):
    def search(self, agent_id: str, query: str, top_k: int = 10) -> list[dict[str, Any]]: ...


class MemoryWriter(Searcher, Protocol):
    def store_conversation(
        self,
        agent_id: str,
        turns: list[dict[str, Any]],
        session_id: str | None = None,
        event_date: float | None = None,
    ) -> list[str]: ...


class BenchmarkAdapter:
    name = "base"

    def load_cases(self, path: str) -> list[BenchmarkCase]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return list(self.parse_cases(payload))

    def parse_cases(self, payload: Any) -> Iterable[BenchmarkCase]:
        raise NotImplementedError

    def ingestion_key(self, case: BenchmarkCase) -> str:
        return case.case_id

    def prepare_sessions(self, case: BenchmarkCase) -> list[dict[str, Any]]:
        return []

    def ingest_case(
        self,
        memory: MemoryWriter,
        case: BenchmarkCase,
        agent_id: str = "benchmark",
    ) -> list[str]:
        message_ids: list[str] = []
        for session in self.prepare_sessions(case):
            turns = session.get("turns")
            if not isinstance(turns, list):
                continue
            message_ids.extend(
                memory.store_conversation(
                    agent_id=agent_id,
                    turns=turns,
                    session_id=str(session.get("session_id") or self.ingestion_key(case)),
                    event_date=session.get("event_date"),
                )
            )
        return message_ids

    def run_case(
        self,
        searcher: Searcher,
        case: BenchmarkCase,
        agent_id: str = "benchmark",
        top_k: int = 5,
    ) -> BenchmarkResult:
        started = time.perf_counter()
        results = searcher.search(agent_id, case.question, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = "\n".join(str(result.get("content", result)) for result in results)
        correct = None
        if case.answer:
            correct = case.answer.lower() in prediction.lower()
        return BenchmarkResult(
            case_id=case.case_id,
            prediction=prediction,
            expected=case.answer,
            latency_ms=latency_ms,
            correct=correct,
        )
