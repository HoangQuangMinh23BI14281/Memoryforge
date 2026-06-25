"""Expansion for [ref:<content_id>] markers."""

from __future__ import annotations

import re

from memoryforge._core import ContentHashTable

REF_PATTERN = re.compile(r"\[ref:([A-Za-z0-9_:\-.]+)\]")


class HashRefResolver:
    def __init__(self, db_path: str):
        self.store = ContentHashTable(db_path)

    def expand(self, text: str, missing: str = "keep") -> str:
        def replace(match: re.Match[str]) -> str:
            content_id = match.group(1)
            content = self.store.retrieve(content_id)
            if content is not None:
                return content
            if missing == "empty":
                return ""
            return match.group(0)

        return REF_PATTERN.sub(replace, text)

    def extract_refs(self, text: str) -> list[str]:
        return REF_PATTERN.findall(text)
