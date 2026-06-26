"""Memory persistence layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memoryforge.memory.longterm import LongTermMemoryIndex, LongTermRecallResult

__all__ = ["LongTermMemoryIndex", "LongTermRecallResult"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from memoryforge.memory.longterm import LongTermMemoryIndex, LongTermRecallResult

        exports = {
            "LongTermMemoryIndex": LongTermMemoryIndex,
            "LongTermRecallResult": LongTermRecallResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
