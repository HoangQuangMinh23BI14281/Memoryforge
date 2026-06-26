"""Route queries to the active memory retrieval surface."""

from __future__ import annotations

from typing import Any


class UnifiedQueryRouter:
    """Thin compatibility router for conversation/long-term memory search."""

    def __init__(self, lcm_search: Any | None = None):
        self.lcm = lcm_search

    def query(self, query_text: str, top_k: int = 10) -> dict[str, Any]:
        return {
            "query_type": "memory",
            "results": self._call_search(self.lcm, query_text, top_k),
            "source": "lcm",
        }

    @staticmethod
    def _call_search(searcher: Any, query: str, top_k: int) -> list[dict[str, Any]]:
        if searcher is None:
            return []
        try:
            return list(searcher.search(query, top_k))
        except TypeError:
            return list(searcher.search(query=query, top_k=top_k))
