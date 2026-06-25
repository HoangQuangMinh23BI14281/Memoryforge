"""Unified retrieval layer."""

__all__ = [
    "UnifiedQueryRouter",
    "VectorIndex",
]


def __getattr__(name: str) -> object:
    if name == "UnifiedQueryRouter":
        from memoryforge.search.router import UnifiedQueryRouter

        return UnifiedQueryRouter
    if name == "VectorIndex":
        from memoryforge.search.vector import VectorIndex

        return VectorIndex
    raise AttributeError(name)
