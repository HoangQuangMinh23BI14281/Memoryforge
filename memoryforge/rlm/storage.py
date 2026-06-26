"""Storage helpers shared by the RLM engine mixins."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from memoryforge._core import ContentHashTable
from memoryforge.db.schema import CONTENT_STORE_SCHEMA_SQL, VECTOR_SCHEMA_SQL
from memoryforge.lcm.store import ImmutableMessageStore
from memoryforge.lcm.summary import SummaryDAG
from memoryforge.rlm.common import is_transient_sqlite_error
from memoryforge.rlm.schema import RLM_SCHEMA_SQL, ensure_rlm_columns
from memoryforge.search.fts import ensure_search_fts
from memoryforge.search.vector import VectorIndex


class RLMStorageMixin:
    db_path: str
    conn: sqlite3.Connection
    _content: ContentHashTable | None
    _vector: VectorIndex | None
    _lcm_store: ImmutableMessageStore | None
    _dag: SummaryDAG | None
    _has_fts: bool

    def _init_schema(self) -> None:
        for attempt in range(5):
            try:
                self.conn.executescript(CONTENT_STORE_SCHEMA_SQL)
                self.conn.executescript(VECTOR_SCHEMA_SQL)
                self.conn.executescript(RLM_SCHEMA_SQL)
                ensure_rlm_columns(self.conn)
                break
            except sqlite3.OperationalError as exc:
                if attempt < 4 and is_transient_sqlite_error(exc):
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise
        self._has_fts = ensure_search_fts(self.conn)

    def _table_exists(self, table_name: str) -> bool:
        try:
            row = self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
                (table_name,),
            ).fetchone()
        except sqlite3.DatabaseError:
            return False
        return row is not None

    def _vector_index(self) -> VectorIndex | None:
        if self._vector is not None:
            return self._vector
        try:
            self._vector = VectorIndex(self.db_path)
        except sqlite3.DatabaseError:
            return None
        return self._vector

    def _lcm_message_store(self) -> ImmutableMessageStore:
        if self._lcm_store is None:
            self._lcm_store = ImmutableMessageStore(self.db_path)
        return self._lcm_store

    def _summary_dag(self) -> SummaryDAG:
        if self._dag is None:
            self._dag = SummaryDAG(self.db_path)
        return self._dag

    def _content_store(self) -> ContentHashTable:
        if self._content is None:
            self._content = ContentHashTable(self.db_path)
        return self._content

    def _retrieve_content(self, content_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT content FROM content_store WHERE content_id = ?",
            (content_id,),
        ).fetchone()
        return str(row[0]) if row else None

    def close(self) -> None:
        for component_name in ("_content", "_vector"):
            component: Any = getattr(self, component_name)
            if component is None:
                continue
            conn = getattr(component, "conn", None)
            if conn is not None:
                conn.close()
            setattr(self, component_name, None)
        if self._lcm_store is not None:
            self._lcm_store.close()
            self._lcm_store = None
        if self._dag is not None:
            self._dag.close()
            self._dag = None
        self.conn.close()
