"""LongMemEval adapter."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from memoryforge.benchmark.adapter import BenchmarkAdapter, BenchmarkCase
from memoryforge.memory.longterm.models import MetadataField


class LongMemEvalAdapter(BenchmarkAdapter):
    name = "longmemeval"

    def parse_cases(self, payload: Any) -> Iterable[BenchmarkCase]:
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        if records is None:
            return
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            yield BenchmarkCase(
                case_id=str(record.get("id", record.get("question_id", index))),
                question=str(record.get("question", record.get("query", ""))),
                answer=str(record.get("answer")) if record.get("answer") is not None else None,
                context=record.get("haystack_sessions") or record.get("context"),
                metadata={
                    "raw": record,
                    "question_type": record.get("question_type"),
                    "question_date": record.get("question_date"),
                },
            )

    def prepare_sessions(self, case: BenchmarkCase) -> list[dict[str, Any]]:
        record = (case.metadata or {}).get("raw")
        if not isinstance(record, dict):
            return []
        haystack_sessions = record.get("haystack_sessions")
        haystack_dates = record.get("haystack_dates") or []
        haystack_session_ids = record.get("haystack_session_ids") or []
        if not isinstance(haystack_sessions, list):
            return []
        sessions: list[dict[str, Any]] = []
        for index, turns_payload in enumerate(haystack_sessions):
            if not isinstance(turns_payload, list):
                continue
            session_id = _list_get(haystack_session_ids, index, f"session_{index}")
            date_text = _list_get(haystack_dates, index, "")
            turns: list[dict[str, Any]] = []
            for turn in turns_payload:
                if not isinstance(turn, dict):
                    continue
                role = _normal_role(str(turn.get("role") or turn.get("speaker") or "user"))
                content = str(turn.get("content") or turn.get("text") or turn.get("message") or "")
                turn_metadata = {
                    MetadataField.BENCHMARK: self.name,
                    MetadataField.BENCHMARK_CASE_ID: case.case_id,
                    MetadataField.BENCHMARK_SESSION_ID: session_id,
                    MetadataField.BENCHMARK_SESSION_DATE: date_text,
                    MetadataField.QUESTION_DATE: record.get("question_date"),
                    MetadataField.QUESTION_TYPE: record.get("question_type"),
                }
                if turn.get("has_answer") is not None:
                    turn_metadata[MetadataField.ANSWER_EVIDENCE] = bool(turn.get("has_answer"))
                turns.append(
                    {
                        "role": role,
                        "metadata": turn_metadata,
                        "content": "\n".join(
                            [
                                f"Question ID: {case.case_id}",
                                f"Haystack session: {session_id}",
                                f"Session date: {date_text}",
                                f"Content: {content}",
                            ]
                        ),
                    }
                )
            if turns:
                sessions.append(
                    {
                        "session_id": f"{case.case_id}_{session_id}",
                        "turns": turns,
                        "event_date": None,
                    }
                )
        return sessions


def _list_get(values: Any, index: int, default: str) -> str:
    if isinstance(values, list) and index < len(values):
        return str(values[index])
    return default


def _normal_role(role: str) -> str:
    return role if role in {"user", "assistant"} else "user"
