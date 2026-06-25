"""LoCoMo adapter."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from memoryforge.benchmark.adapter import BenchmarkAdapter, BenchmarkCase
from memoryforge.memory.longterm.models import MetadataField


class LoComoAdapter(BenchmarkAdapter):
    name = "locomo"

    def parse_cases(self, payload: Any) -> Iterable[BenchmarkCase]:
        if _is_locomo_sample(payload):
            yield from self._parse_sample(payload)
            return
        if (
            isinstance(payload, list)
            and payload
            and all(_is_locomo_sample(item) for item in payload)
        ):
            for sample in payload:
                yield from self._parse_sample(sample)
            return
        records = (
            payload.get("qa", payload.get("data", payload))
            if isinstance(payload, dict)
            else payload
        )
        if records is None:
            return
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            yield BenchmarkCase(
                case_id=str(record.get("id", record.get("qid", index))),
                question=str(record.get("question", record.get("query", ""))),
                answer=record.get("answer"),
                context=record.get("conversation") or record.get("context"),
                metadata={"raw": record},
            )

    def ingestion_key(self, case: BenchmarkCase) -> str:
        sample_id = (case.metadata or {}).get("sample_id")
        return str(sample_id or case.case_id)

    def prepare_sessions(self, case: BenchmarkCase) -> list[dict[str, Any]]:
        metadata = case.metadata or {}
        sample = metadata.get("sample")
        if not isinstance(sample, dict):
            return []
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            return []
        speaker_a = str(conversation.get("speaker_a") or "")
        sample_id = str(sample.get("sample_id") or case.case_id)
        evidence_ids = _evidence_ids(metadata.get("evidence"))
        sessions: list[dict[str, Any]] = []
        for session_key in sorted(_session_keys(conversation), key=_session_sort_key):
            turns_payload = conversation.get(session_key)
            if not isinstance(turns_payload, list):
                continue
            date_text = str(conversation.get(f"{session_key}_date_time") or "")
            turns: list[dict[str, Any]] = []
            for turn in turns_payload:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or "")
                dia_id = str(turn.get("dia_id") or "")
                text = str(turn.get("text") or turn.get("content") or "")
                role = "user" if speaker == speaker_a else "assistant"
                turn_metadata = {
                    MetadataField.BENCHMARK: self.name,
                    MetadataField.BENCHMARK_CASE_ID: case.case_id,
                    MetadataField.BENCHMARK_SESSION_ID: session_key,
                    MetadataField.BENCHMARK_SESSION_DATE: _canonical_date(date_text),
                    MetadataField.BENCHMARK_DIALOGUE_ID: dia_id,
                    MetadataField.BENCHMARK_CATEGORY: metadata.get("category"),
                    MetadataField.EVIDENCE_IDS: sorted(evidence_ids),
                    MetadataField.ANSWER_EVIDENCE: dia_id in evidence_ids,
                }
                turns.append(
                    {
                        "role": role,
                        "metadata": turn_metadata,
                        "content": "\n".join(
                            [
                                f"Sample: {sample_id}",
                                f"Session: {session_key}",
                                f"Session date: {_canonical_date(date_text)}",
                                f"Dialogue ID: {dia_id}",
                                f"Speaker: {speaker}",
                                f"Text: {text}",
                            ]
                        ),
                    }
                )
            if turns:
                sessions.append(
                    {
                        "session_id": f"{sample_id}_{session_key}",
                        "turns": turns,
                        "event_date": None,
                    }
                )
        return sessions

    def _parse_sample(self, sample: dict[str, Any]) -> Iterable[BenchmarkCase]:
        sample_id = str(sample.get("sample_id") or "locomo")
        qa_items = sample.get("qa")
        if not isinstance(qa_items, list):
            return
        for index, qa in enumerate(qa_items):
            if not isinstance(qa, dict):
                continue
            yield BenchmarkCase(
                case_id=f"{sample_id}:{index}",
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer")) if qa.get("answer") is not None else None,
                context=None,
                metadata={
                    "sample_id": sample_id,
                    "sample": sample,
                    "qa": qa,
                    "evidence": qa.get("evidence", []),
                    "category": qa.get("category"),
                },
            )


def _is_locomo_sample(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("conversation"), dict)
        and isinstance(value.get("qa"), list)
    )


def _session_keys(conversation: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in conversation.items()
        if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list)
    ]


def _session_sort_key(key: str) -> int:
    try:
        return int(key.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _evidence_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item}
    return {str(value)}


def _canonical_date(date_text: str) -> str:
    if not date_text:
        return ""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            parsed = datetime.strptime(date_text, fmt)
            return parsed.replace(tzinfo=timezone.utc).strftime("%d %B %Y")
        except ValueError:
            continue
    return date_text
