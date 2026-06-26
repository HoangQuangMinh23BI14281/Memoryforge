"""Long-term memory result models and metadata contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


class MetadataField:
    """Metadata keys that LTM lifecycle, provenance, and benchmarks understand."""

    KIND: Final = "kind"
    CONFIDENCE: Final = "confidence"
    FRESHNESS: Final = "freshness"
    TIMESTAMP: Final = "timestamp"
    EVENT_DATE: Final = "event_date"
    CREATED_AT: Final = "created_at"
    SOURCE_ORIGIN: Final = "source_origin"
    RAW_REFS: Final = "raw_refs"
    SUPERSEDES: Final = "supersedes"
    SUPERSEDED_BY: Final = "superseded_by"
    CONTRADICTS: Final = "contradicts"
    CONTRADICTED_BY: Final = "contradicted_by"
    CONFLICTING_ITEM_IDS: Final = "conflicting_item_ids"
    VALID_TO: Final = "valid_to"
    SOURCE: Final = "source"
    ROLE: Final = "role"
    SOURCE_PATH: Final = "source_path"
    SESSION_ID: Final = "session_id"
    MERGE_KEY: Final = "merge_key"
    CONTROLLER_IGNORED: Final = "controller_ignored"
    CORRECTED_AT: Final = "corrected_at"
    CORRECTED_FACT: Final = "corrected_fact"
    RECORDED_AT: Final = "recorded_at"
    MESSAGE_ID: Final = "message_id"
    PART_ID: Final = "part_id"
    PART_INDEX: Final = "part_index"
    BENCHMARK: Final = "benchmark"
    BENCHMARK_CASE_ID: Final = "benchmark_case_id"
    BENCHMARK_SESSION_ID: Final = "benchmark_session_id"
    BENCHMARK_SESSION_DATE: Final = "benchmark_session_date"
    BENCHMARK_DIALOGUE_ID: Final = "benchmark_dialogue_id"
    BENCHMARK_CATEGORY: Final = "benchmark_category"
    QUESTION_DATE: Final = "question_date"
    QUESTION_TYPE: Final = "question_type"
    ANSWER_EVIDENCE: Final = "answer_evidence"
    ANSWER_EVIDENCE_RANGES: Final = "answer_evidence_ranges"
    EVIDENCE_IDS: Final = "evidence_ids"


class MemoryConfidence:
    HIGH: Final = "high"
    LOW: Final = "low"
    MEDIUM: Final = "medium"


TEMPORAL_PROVENANCE_FIELDS: Final = (
    MetadataField.TIMESTAMP,
    MetadataField.EVENT_DATE,
    MetadataField.CREATED_AT,
)


def metadata_temporal_provenance(metadata: dict[str, Any]) -> Any:
    for field in TEMPORAL_PROVENANCE_FIELDS:
        value = metadata.get(field)
        if value not in (None, ""):
            return value
    return None


@dataclass(frozen=True)
class LongTermRecallResult:
    item_id: str
    source_type: str
    source_id: str
    content_id: str
    raw_ref: str
    preview: str
    score: float
    streams: dict[str, dict[str, float | int]]
    metadata: dict[str, Any] | None = None
    content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "item_id": self.item_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "content_id": self.content_id,
            "raw_ref": self.raw_ref,
            "preview": self.preview,
            "score": self.score,
            "streams": self.streams,
            "metadata": self.metadata or {},
        }
        if self.content is not None:
            payload["content"] = self.content
        return payload
