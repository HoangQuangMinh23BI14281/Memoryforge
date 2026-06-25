"""RLM external-environment pipeline."""

from __future__ import annotations

from pathlib import Path

from memoryforge._core import ContentHashTable
from memoryforge.rlm.chunking import RLMChunkMixin, RLMContentChunker
from memoryforge.rlm.common import connect
from memoryforge.rlm.records import RLMRecordMixin
from memoryforge.rlm.runner import RLMRunMixin
from memoryforge.rlm.search import RLMSearchMixin
from memoryforge.rlm.storage import RLMStorageMixin
from memoryforge.search.vector import VectorIndex


class RLMEngine(
    RLMRunMixin,
    RLMRecordMixin,
    RLMChunkMixin,
    RLMSearchMixin,
    RLMStorageMixin,
):
    """CLI-first RLM substrate: load, chunk, search, dispatch, record, aggregate."""

    def __init__(self, db_path: str = "~/.memoryforge/memory.db", *, ensure_schema: bool = True):
        self.db_path = str(Path(db_path).expanduser())
        self.conn = connect(self.db_path)
        self._content: ContentHashTable | None = None
        self._vector: VectorIndex | None = None
        self._lcm_store = None
        self._dag = None
        self.chunker = RLMContentChunker()
        self._has_fts = True
        if ensure_schema:
            self._init_schema()
