"""Long-term memory index coordinator."""

from __future__ import annotations

from pathlib import Path

from memoryforge._core import ContentHashTable
from memoryforge.db.schema import (
    CONTENT_STORE_SCHEMA_SQL,
    LONG_TERM_SCHEMA_SQL,
)
from memoryforge.memory.longterm.indexing import LongTermIndexingMixin
from memoryforge.memory.longterm.retrieval import LongTermRetrievalMixin
from memoryforge.memory.longterm.utils import connect
from memoryforge.search.fts import ensure_search_fts
from memoryforge.search.vector import VectorIndex


class LongTermMemoryIndex(LongTermIndexingMixin, LongTermRetrievalMixin):
    """Rebuildable BM25 and vector indexes over raw immutable rows."""

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser())
        self.conn = connect(self.db_path)
        self.content = ContentHashTable(self.db_path)
        self.vector = VectorIndex(self.db_path)
        self._has_fts = True
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(CONTENT_STORE_SCHEMA_SQL)
        self.conn.executescript(LONG_TERM_SCHEMA_SQL)
        self._has_fts = ensure_search_fts(self.conn)

    def close(self) -> None:
        for component_name in ("content", "vector"):
            component = getattr(self, component_name, None)
            close = getattr(component, "close", None)
            if close:
                close()
            else:
                conn = getattr(component, "conn", None)
                if conn is not None:
                    conn.close()
            setattr(self, component_name, None)
        self.conn.close()
