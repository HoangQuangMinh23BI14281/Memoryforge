"""Long-term memory index package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memoryforge.memory.longterm.models import LongTermRecallResult, MemoryConfidence, MetadataField

if TYPE_CHECKING:
    from memoryforge.memory.longterm.store import LongTermMemoryIndex

__all__ = ["LongTermMemoryIndex", "LongTermRecallResult", "MemoryConfidence", "MetadataField"]


def __getattr__(name: str) -> object:
    if name == "LongTermMemoryIndex":
        from memoryforge.memory.longterm.store import LongTermMemoryIndex

        return LongTermMemoryIndex
    if name == "LongTermRecallResult":
        return LongTermRecallResult
    if name == "MemoryConfidence":
        return MemoryConfidence
    if name == "MetadataField":
        return MetadataField
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
