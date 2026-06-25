"""Core-model context bundle contract.

MemoryForge prepares this payload for the active runtime model. It does not
represent an answer and should not imply that LTM or LCM generated one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CoreContextBundle:
    messages: list[dict[str, str]]
    active_recall: list[dict[str, Any]]
    long_term_recall: list[dict[str, Any]]
    summary_nodes: list[dict[str, Any]]
    raw_refs: list[str]
    token_estimate: int
    budget: dict[str, Any]
    provenance: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_model_payload(self) -> dict[str, Any]:
        """Return the only payload surface intended for a core model prompt."""

        return {"messages": [dict(message) for message in self.messages]}

    def to_audit_payload(self) -> dict[str, Any]:
        """Return non-rendered metadata for middleware, logging, and debugging."""

        return {
            "active_recall": self.active_recall,
            "long_term_recall": self.long_term_recall,
            "summary_nodes": self.summary_nodes,
            "raw_refs": self.raw_refs,
            "token_estimate": self.token_estimate,
            "budget": self.budget,
            "provenance": self.provenance,
            "diagnostics": self.diagnostics,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_model_payload(),
            **self.to_audit_payload(),
        }
