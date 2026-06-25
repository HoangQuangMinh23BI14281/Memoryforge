"""Canonical SQLite schema for all MemoryForge subsystems."""

from __future__ import annotations

import sqlite3
from pathlib import Path

CONTENT_STORE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS content_store (
    content_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INTEGER,
    reference_count INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON content_store(content_hash);
CREATE INDEX IF NOT EXISTS idx_reference_count ON content_store(reference_count);
"""

VECTOR_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vec_index (
    cache_key TEXT PRIMARY KEY,
    content_id TEXT NOT NULL,
    numeric_id INTEGER NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    model TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vec_index_content_model
ON vec_index(content_id, model);
"""

LONG_TERM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS long_term_items (
    item_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    content_id TEXT NOT NULL,
    preview TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    indexed_at REAL NOT NULL,
    UNIQUE(agent_id, source_type, source_id)
);

CREATE INDEX IF NOT EXISTS idx_long_term_agent_source
ON long_term_items(agent_id, source_type, source_id);

CREATE INDEX IF NOT EXISTS idx_long_term_agent_indexed_at
ON long_term_items(agent_id, indexed_at DESC);

CREATE INDEX IF NOT EXISTS idx_long_term_agent_content
ON long_term_items(agent_id, content_id);
"""

LCM_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    agent         TEXT NOT NULL DEFAULT 'default',
    system_prompt TEXT NOT NULL DEFAULT '',
    model_id      TEXT NOT NULL DEFAULT '',
    provider_id   TEXT NOT NULL DEFAULT '',
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_agent_updated
    ON sessions(agent, updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    parent_id       TEXT REFERENCES messages(id),
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    created_at      INTEGER NOT NULL,
    agent           TEXT NOT NULL DEFAULT 'default',
    model_id        TEXT NOT NULL DEFAULT '',
    provider_id     TEXT NOT NULL DEFAULT '',
    is_summary      INTEGER NOT NULL DEFAULT 0,
    tokens_input    INTEGER DEFAULT 0,
    tokens_output   INTEGER DEFAULT 0,
    tokens_cache_read  INTEGER DEFAULT 0,
    tokens_cache_write INTEGER DEFAULT 0,
    tokens_total    INTEGER DEFAULT 0,
    cost            REAL DEFAULT 0.0,
    finish_reason   TEXT,
    error_code      TEXT,
    error_message_text TEXT,
    error_retriable INTEGER DEFAULT 0,
    mode            TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_session_summary
    ON messages(session_id, is_summary)
    WHERE is_summary = 1;

CREATE TABLE IF NOT EXISTS message_parts (
    id              TEXT PRIMARY KEY,
    message_id      TEXT NOT NULL REFERENCES messages(id),
    session_id      TEXT NOT NULL,
    part_type       TEXT NOT NULL,
    part_index      INTEGER NOT NULL DEFAULT 0,
    content_id      TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL DEFAULT '',
    tool_name       TEXT,
    tool_call_id    TEXT,
    tool_state      TEXT,
    is_protected    INTEGER NOT NULL DEFAULT 0,
    compacted_at    INTEGER,
    started_at      INTEGER,
    completed_at    INTEGER,
    token_estimate  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_parts_message_id
    ON message_parts(message_id, part_index);

CREATE INDEX IF NOT EXISTS idx_parts_content_id
    ON message_parts(content_id);

CREATE INDEX IF NOT EXISTS idx_parts_session_tool
    ON message_parts(session_id, part_type, compacted_at)
    WHERE part_type = 'tool';

CREATE TABLE IF NOT EXISTS summary_nodes (
    id                    TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL,
    level                 INTEGER NOT NULL DEFAULT 0,
    span_start_message_id TEXT NOT NULL DEFAULT '',
    span_end_message_id   TEXT NOT NULL DEFAULT '',
    content               TEXT NOT NULL DEFAULT '',
    token_count           INTEGER NOT NULL DEFAULT 0,
    created_at            INTEGER NOT NULL,
    parent_node_id        TEXT,
    model_id              TEXT NOT NULL DEFAULT '',
    provider_id           TEXT NOT NULL DEFAULT '',
    compaction_level      INTEGER NOT NULL DEFAULT 1,
    is_active             INTEGER NOT NULL DEFAULT 1,
    kind                  TEXT NOT NULL DEFAULT 'leaf',
    parent_node_ids       TEXT NOT NULL DEFAULT '[]',
    superseded            INTEGER NOT NULL DEFAULT 0,
    file_ids              TEXT NOT NULL DEFAULT '[]',
    source_refs           TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_summary_nodes_session
    ON summary_nodes(session_id, level);

CREATE INDEX IF NOT EXISTS idx_summary_nodes_span
    ON summary_nodes(session_id, span_start_message_id, span_end_message_id);

CREATE TABLE IF NOT EXISTS context_items (
    id         INTEGER PRIMARY KEY,
    session_id TEXT    NOT NULL,
    item_type  TEXT    NOT NULL CHECK(item_type IN ('message', 'summary')),
    item_id    TEXT    NOT NULL,
    position   INTEGER NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_context_items_session_pos
    ON context_items(session_id, position);

CREATE INDEX IF NOT EXISTS idx_context_items_session
    ON context_items(session_id);

CREATE INDEX IF NOT EXISTS idx_context_items_session_item
    ON context_items(session_id, item_id);

"""

MEMORYFORGE_SCHEMA_SQL = "\n\n".join(
    [
        LCM_SCHEMA_SQL,
        CONTENT_STORE_SCHEMA_SQL,
        VECTOR_SCHEMA_SQL,
        LONG_TERM_SCHEMA_SQL,
    ]
)

LCM_SUMMARY_NODE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("parent_node_id", "TEXT"),
    ("model_id", "TEXT NOT NULL DEFAULT ''"),
    ("provider_id", "TEXT NOT NULL DEFAULT ''"),
    ("compaction_level", "INTEGER NOT NULL DEFAULT 1"),
    ("is_active", "INTEGER NOT NULL DEFAULT 1"),
    ("kind", "TEXT NOT NULL DEFAULT 'leaf'"),
    ("parent_node_ids", "TEXT NOT NULL DEFAULT '[]'"),
    ("superseded", "INTEGER NOT NULL DEFAULT 0"),
    ("file_ids", "TEXT NOT NULL DEFAULT '[]'"),
    ("source_refs", "TEXT NOT NULL DEFAULT '[]'"),
)

LCM_MESSAGE_PART_COLUMNS: tuple[tuple[str, str], ...] = (
    ("content_id", "TEXT NOT NULL DEFAULT ''"),
    ("is_protected", "INTEGER NOT NULL DEFAULT 0"),
)

LCM_SESSION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("system_prompt", "TEXT NOT NULL DEFAULT ''"),
    ("model_id", "TEXT NOT NULL DEFAULT ''"),
    ("provider_id", "TEXT NOT NULL DEFAULT ''"),
    ("updated_at", "INTEGER NOT NULL DEFAULT 0"),
)

CONTENT_STORE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("reference_count", "INTEGER NOT NULL DEFAULT 1"),
    ("last_accessed", "REAL NOT NULL DEFAULT 0"),
)


def ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def ensure_content_store_columns(conn: sqlite3.Connection) -> None:
    for column, ddl in CONTENT_STORE_COLUMNS:
        ensure_column(conn, "content_store", column, ddl)
    conn.execute("UPDATE content_store SET last_accessed = created_at WHERE last_accessed = 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reference_count ON content_store(reference_count)")


def ensure_lcm_summary_columns(conn: sqlite3.Connection) -> None:
    for column, ddl in LCM_SUMMARY_NODE_COLUMNS:
        ensure_column(conn, "summary_nodes", column, ddl)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_summary_nodes_active "
        "ON summary_nodes(session_id, superseded)"
    )


def ensure_lcm_message_part_columns(conn: sqlite3.Connection) -> None:
    for column, ddl in LCM_MESSAGE_PART_COLUMNS:
        ensure_column(conn, "message_parts", column, ddl)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_parts_content_id ON message_parts(content_id)")


def ensure_lcm_session_columns(conn: sqlite3.Connection) -> None:
    for column, ddl in LCM_SESSION_COLUMNS:
        ensure_column(conn, "sessions", column, ddl)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_agent_updated ON sessions(agent, updated_at)"
    )


def ensure_long_term_indexes(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_long_term_agent_indexed_at "
        "ON long_term_items(agent_id, indexed_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_long_term_agent_content "
        "ON long_term_items(agent_id, content_id)"
    )


def init_memoryforge_schema(db_path: str) -> None:
    path = Path(db_path).expanduser()
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(MEMORYFORGE_SCHEMA_SQL)
        ensure_content_store_columns(conn)
        ensure_lcm_summary_columns(conn)
        ensure_lcm_message_part_columns(conn)
        ensure_lcm_session_columns(conn)
        ensure_long_term_indexes(conn)
        conn.commit()
    finally:
        conn.close()
