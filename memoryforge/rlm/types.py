"""Shared RLM data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ChunkDraft:
    content: str
    content_type: str
    strategy: str
    byte_start: int
    byte_end: int
    char_start: int
    char_end: int
    start_line: int | None
    end_line: int | None
    has_overlap: bool = False
    metadata: dict[str, Any] | None = None
