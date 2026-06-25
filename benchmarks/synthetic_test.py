#!/usr/bin/env python3
"""Synthetic end-to-end smoke benchmark."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from memoryforge.api import MemoryForge


def main() -> int:
    with tempfile.TemporaryDirectory() as tempdir:
        db_path = str(Path(tempdir) / "memory.db")
        mf = MemoryForge(db_path)
        started = time.perf_counter()
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "We chose JWT authentication with SQLite."}],
            session_id="synthetic",
        )
        mf.ingest_prompt("agent", "Alice uses SQLite and prefers JWT.", session_id="synthetic-2")
        memory_hits = mf.search("agent", "JWT SQLite")
        long_term_hits = mf.recall_long_term("agent", "Alice SQLite", top_k=5)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        print(
            json.dumps(
                {
                    "memory_hits": len(memory_hits),
                    "long_term_hits": len(long_term_hits),
                    "elapsed_ms": elapsed_ms,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
