#!/usr/bin/env python3
"""Initialize or update a MemoryForge SQLite database schema."""

from __future__ import annotations

import argparse

from memoryforge.db import init_memoryforge_schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure the MemoryForge SQLite schema exists")
    parser.add_argument("--db", default="~/.memoryforge/memory.db")
    args = parser.parse_args()

    init_memoryforge_schema(args.db)
    print(f"Ensured MemoryForge schema at {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
