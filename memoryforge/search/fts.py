"""Shared SQLite FTS index for local lexical search."""

from __future__ import annotations

import sqlite3

SEARCH_FTS_TABLE = "search_fts"


def ensure_search_fts(conn: sqlite3.Connection) -> bool:
    """Create the shared FTS table.

    Returns True when SQLite FTS5 is available, False when a plain-table fallback
    is used. Scope values separate conversation, RLM chunk, and LTM rows.
    """

    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
                content,
                scope UNINDEXED,
                agent_id UNINDEXED,
                source_type UNINDEXED,
                source_id UNINDEXED,
                content_id UNINDEXED,
                session_id UNINDEXED,
                buffer_id UNINDEXED,
                role UNINDEXED,
                tokenize='porter unicode61'
            )
            """
        )
        conn.commit()
        return True
    except sqlite3.OperationalError:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_fts (
                content TEXT NOT NULL,
                scope TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL DEFAULT '',
                content_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                buffer_id TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_search_fts_scope_agent ON search_fts(scope, agent_id)"
        )
        conn.commit()
        return False
