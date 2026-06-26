"""SQLite schema for the RLM subsystem."""

from __future__ import annotations

import sqlite3

RLM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rlm_buffers (
    buffer_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    name TEXT,
    source_path TEXT,
    content_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_type TEXT NOT NULL,
    strategy TEXT NOT NULL,
    size INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rlm_buffers_agent ON rlm_buffers(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_rlm_buffers_name ON rlm_buffers(agent_id, name);

CREATE TABLE IF NOT EXISTS rlm_chunks (
    chunk_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    buffer_id TEXT NOT NULL,
    content_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    strategy TEXT NOT NULL,
    byte_start INTEGER NOT NULL,
    byte_end INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    token_count INTEGER NOT NULL,
    has_overlap INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    FOREIGN KEY(buffer_id) REFERENCES rlm_buffers(buffer_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rlm_chunks_buffer ON rlm_chunks(buffer_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_rlm_chunks_content ON rlm_chunks(content_id);

"""


def ensure_rlm_columns(conn: sqlite3.Connection) -> None:
    return None
