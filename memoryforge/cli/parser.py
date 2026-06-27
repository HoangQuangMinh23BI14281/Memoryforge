"""CLI argument parser."""

from __future__ import annotations

import argparse

PUBLIC_RUNNER_CHOICES = ["auto", "codex", "mock"]
PUBLIC_RUNNER_HELP = (
    "Sub-agent backend. Use auto/codex for real runs, mock for tests. "
    "Advanced backends remain available through MEMORYFORGE_SUBAGENT_RUNNER."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memoryforge",
        description="MemoryForge: Lossless Context Management for Recursive Language Model",
    )
    parser.add_argument("--db", default="~/.memoryforge/memory.db", help="SQLite database path")
    subcommands = parser.add_subparsers(dest="command", required=True)

    store = subcommands.add_parser("store-session", help="Store a JSON conversation session")
    store.add_argument("--agent-id", required=True)
    store.add_argument("--session-id")
    store.add_argument("--session-file", required=True)

    search = subcommands.add_parser("search", help="Search conversation memory")
    search.add_argument("--agent-id", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)
    search.add_argument(
        "--ensemble", action="store_true", help="Use long-term BM25 + vector recall"
    )

    recall_memory = subcommands.add_parser(
        "recall-memory",
        help="Recall long-term memory with BM25 and vector indexes",
    )
    recall_memory.add_argument("--agent-id", required=True)
    recall_memory.add_argument("--query", required=True)
    recall_memory.add_argument("--session-id")
    recall_memory.add_argument("--limit", type=int, default=10)
    recall_memory.add_argument("--include-content", action="store_true")

    active_recall = subcommands.add_parser(
        "active-recall",
        help="Proactively surface recent durable evidence for the active core model",
    )
    active_recall.add_argument("--agent-id", required=True)
    active_recall.add_argument("--session-id")
    active_recall.add_argument("--focus")
    active_recall.add_argument("--project-root")
    active_recall.add_argument("--limit", type=int, default=8)
    active_recall.add_argument("--include-content", action="store_true")

    record_contradiction = subcommands.add_parser(
        "record-contradiction",
        help="Record a contested memory relation without choosing a winner",
    )
    record_contradiction.add_argument("--agent-id", required=True)
    record_contradiction.add_argument("--statement", required=True)
    record_contradiction.add_argument("--conflicting-item-id", action="append", default=[])
    record_contradiction.add_argument("--conflicting-raw-ref", action="append", default=[])
    record_contradiction.add_argument("--session-id")
    record_contradiction.add_argument("--source", default="user")

    find_contradictions = subcommands.add_parser(
        "find-contradictions",
        help="List memories marked as conflicting through contradiction metadata",
    )
    find_contradictions.add_argument("--agent-id", required=True)
    find_contradictions.add_argument("--query")
    find_contradictions.add_argument("--limit", type=int, default=10)
    find_contradictions.add_argument("--include-content", action="store_true")

    runtime_context = subcommands.add_parser(
        "runtime-context",
        help="Validate active runtime delivery and build a core-model context bundle",
    )
    runtime_context.add_argument("--agent-id", required=True)
    runtime_context.add_argument("--session-id", required=True)
    runtime_context.add_argument("--query", required=True)
    runtime_context.add_argument("--project-root", default=".")
    runtime_context.add_argument("--runtime", choices=["auto", "codex"], default="auto")
    runtime_context.add_argument("--context-limit", type=int, default=200_000)
    runtime_context.add_argument("--reserved-output", type=int, default=4_000)
    runtime_context.add_argument("--compaction-buffer", type=int, default=2_000)
    runtime_context.add_argument("--top-k", type=int, default=5)
    runtime_context.add_argument("--include-content", action="store_true")
    runtime_context.add_argument(
        "--recall-content-policy",
        choices=["snippet", "champion", "full", "auto", "preview"],
        default="snippet",
    )
    runtime_context.add_argument("--long-term-token-budget", type=int, default=None)

    memory_source = subcommands.add_parser(
        "long-term-source", help="Fetch immutable raw source for an LTM item"
    )
    memory_source.add_argument("--agent-id", required=True)
    memory_source.add_argument("--item-id", required=True)

    lcm_context = subcommands.add_parser(
        "lcm-context", help="Build bounded LCM context for a session"
    )
    lcm_context.add_argument("--session-id", required=True)
    lcm_context.add_argument("--context-limit", type=int, default=200_000)
    lcm_context.add_argument("--reserved-output", type=int, default=4_000)
    lcm_context.add_argument("--compaction-buffer", type=int, default=2_000)
    lcm_context.add_argument("--agent-id")
    lcm_context.add_argument("--recall-query")
    lcm_context.add_argument("--recall-limit", type=int, default=5)

    lcm_compact = subcommands.add_parser(
        "lcm-compact", help="Run LCM soft/hard threshold compaction"
    )
    lcm_compact.add_argument("--agent-id", required=True)
    lcm_compact.add_argument("--session-id", required=True)
    lcm_compact.add_argument("--context-limit", type=int, default=200_000)
    lcm_compact.add_argument("--reserved-output", type=int, default=4_000)
    lcm_compact.add_argument("--compaction-buffer", type=int, default=2_000)
    lcm_compact.add_argument("--force", action="store_true")
    lcm_compact.add_argument(
        "--defer-soft",
        action="store_true",
        help="Record soft-threshold pressure without running compaction until hard overflow",
    )
    lcm_compact.add_argument("--runner", default=None, choices=PUBLIC_RUNNER_CHOICES)
    lcm_compact.add_argument("--model")
    lcm_compact.add_argument("--project-root", default=".")
    lcm_compact.add_argument("--base-url", help=argparse.SUPPRESS)

    lcm_maintain = subcommands.add_parser(
        "lcm-maintain",
        help="Compact sessions due for LCM maintenance outside the hot path",
    )
    lcm_maintain.add_argument("--agent-id", required=True)
    lcm_maintain.add_argument("--context-limit", type=int, default=200_000)
    lcm_maintain.add_argument("--reserved-output", type=int, default=4_000)
    lcm_maintain.add_argument("--compaction-buffer", type=int, default=2_000)
    lcm_maintain.add_argument("--limit", type=int, default=20)
    lcm_maintain.add_argument("--max-rounds", type=int, default=3)
    lcm_maintain.add_argument("--hard-only", action="store_true")
    lcm_maintain.add_argument("--runner", default=None, choices=PUBLIC_RUNNER_CHOICES)
    lcm_maintain.add_argument("--model")
    lcm_maintain.add_argument("--project-root", default=".")
    lcm_maintain.add_argument("--base-url", help=argparse.SUPPRESS)

    ingest_file = subcommands.add_parser(
        "ingest-file",
        help="Ingest a file into immutable chunks and long-term memory without adding it to LCM",
    )
    ingest_file.add_argument("path")
    ingest_file.add_argument("--agent-id", required=True)
    ingest_file.add_argument("--name")
    ingest_file.add_argument("--chunk-size", type=int, default=12_000)
    ingest_file.add_argument("--overlap", type=int, default=1_000)

    rlm_load = subcommands.add_parser(
        "rlm-load",
        help="Load a prompt/file into RLM buffers, chunks, and LTM without adding it to LCM",
    )
    rlm_load.add_argument("input")
    rlm_load.add_argument("--agent-id", required=True)
    rlm_load.add_argument("--name")
    rlm_load.add_argument("--chunk-size", type=int, default=12_000)
    rlm_load.add_argument("--overlap", type=int, default=1_000)

    rlm_search = subcommands.add_parser("rlm-search", help="Search RLM chunks by BM25/vector/RRF")
    rlm_search.add_argument("--agent-id", required=True)
    rlm_search.add_argument("--query", required=True)
    rlm_search.add_argument("--buffer-id")
    rlm_search.add_argument("--limit", type=int, default=10)
    rlm_search.add_argument(
        "--mode", choices=["hybrid", "bm25", "semantic", "vector"], default="hybrid"
    )

    rlm_chunk_get = subcommands.add_parser(
        "rlm-chunk-get", help="Fetch full RLM chunk content by ID"
    )
    rlm_chunk_get.add_argument("chunk_id")

    dispatch = subcommands.add_parser("dispatch", help="Build RLM pass-by-reference chunk batches")
    dispatch.add_argument("--agent-id", required=True)
    dispatch.add_argument("--query")
    dispatch.add_argument("--buffer-id")
    dispatch.add_argument("--limit", type=int, default=12)
    dispatch.add_argument("--batch-size", type=int, default=None)

    context_get = subcommands.add_parser(
        "context-get", help="Rehydrate an RLM ref such as rlm_chunk:<id>"
    )
    context_get.add_argument("ref")
    context_get.add_argument("--agent-id")

    rlm_record = subcommands.add_parser(
        "rlm-record", help="Store a sub-agent chunk analysis into LCM"
    )
    rlm_record.add_argument("--agent-id", required=True)
    rlm_record.add_argument("--run-id", required=True)
    rlm_record.add_argument("--chunk-id", action="append", required=True)
    rlm_record.add_argument("--batch-index", type=int)
    rlm_record.add_argument("--analysis-file", default="-", help="Text file path, or '-' for stdin")

    aggregate = subcommands.add_parser(
        "aggregate", help="Aggregate recorded RLM analyses into an LCM SummaryDAG parent"
    )
    aggregate.add_argument("--agent-id", required=True)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--summary-file", help="Optional final synthesis text file")

    rlm_run = subcommands.add_parser(
        "rlm-run", help="Run RLM with spawned sub-agents and store results in LCM"
    )
    rlm_run.add_argument(
        "input", nargs="?", help="Prompt text or file path. Omit when using --buffer-id"
    )
    rlm_run.add_argument("--agent-id", required=True)
    rlm_run.add_argument("--name")
    rlm_run.add_argument("--buffer-id")
    rlm_run.add_argument("--query")
    rlm_run.add_argument("--limit", type=int, default=20)
    rlm_run.add_argument("--batch-size", type=int, default=None)
    rlm_run.add_argument("--chunk-size", type=int, default=12_000)
    rlm_run.add_argument("--overlap", type=int, default=1_000)
    rlm_run.add_argument(
        "--runner",
        default="auto",
        choices=PUBLIC_RUNNER_CHOICES,
        help=PUBLIC_RUNNER_HELP,
    )
    rlm_run.add_argument("--model", help="Optional model name for the selected runner")
    rlm_run.add_argument("--base-url", help=argparse.SUPPRESS)
    rlm_run.add_argument("--project-root", default=".")
    rlm_run.add_argument("--timeout", type=float, default=900.0)
    rlm_run.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel RLM analysis sub-agents. DB recording remains ordered.",
    )
    rlm_run.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry transient RLM batch failures up to this many times.",
    )
    rlm_run.add_argument(
        "--allow-partial",
        action="store_true",
        help="Persist successful RLM batch records even when another batch fails.",
    )
    rlm_run.add_argument("--no-synthesis", action="store_true", help="Skip spawned final synthesis")
    rlm_run.add_argument(
        "--no-recursive", action="store_true", help="Disable recursive aggregate reduction"
    )
    rlm_run.add_argument("--max-recursive-rounds", type=int, default=2)
    rlm_run.add_argument("--recursive-token-limit", type=int)

    chunk = subcommands.add_parser(
        "chunk", help="Inspect MemoryForge chunking for text, files, or JSON turns"
    )
    chunk.add_argument("input")
    chunk.add_argument("--conversation-json", action="store_true")

    benchmark = subcommands.add_parser("benchmark", help="Run benchmark smoke tests")
    benchmark.add_argument("--dataset", choices=["synthetic", "second-brain"], default="synthetic")
    benchmark.add_argument("--agent-id", default="benchmark")
    benchmark.add_argument(
        "--mode",
        choices=["ingest-only", "context-only"],
        default="context-only",
        help="Separate ingestion from question-time context assembly",
    )

    init_cmd = subcommands.add_parser(
        "init", help="Initialize MemoryForge for this uv/Python project"
    )
    init_cmd.add_argument("path", nargs="?", default=".")
    init_cmd.add_argument("--agent-id", default="default")
    init_cmd.add_argument("--db")
    init_cmd.add_argument("--no-codex", action="store_true")
    init_cmd.add_argument("--no-index", action="store_true", help="Skip automatic Markdown indexing during init")
    init_cmd.add_argument("--force", action="store_true")

    install_codex = subcommands.add_parser(
        "install-codex", help="Install MemoryForge MCP instructions into the global Codex config"
    )
    install_codex.add_argument("--force", action="store_true")

    hook = subcommands.add_parser("hook", help="Internal Codex hook ingestion endpoint")
    hook.add_argument("event")
    hook.add_argument("--db", required=True)
    hook.add_argument("--agent-id", default="default")
    hook.add_argument("--project-root", default=".")

    subcommands.add_parser("mcp-server", help="Run the MCP server")
    return parser

