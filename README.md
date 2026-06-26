# MemoryForge

MemoryForge is a local-first memory layer for Codex CLI workflows.

It stores durable project evidence in SQLite, keeps live context bounded, and
returns source-backed context bundles to the model the user is already running.
MemoryForge is not a separate answering agent and it is not a codebase AST graph
engine. The host model answers; MemoryForge supplies memory, references, and
provenance.

```text
User -> Codex CLI -> MemoryForge MCP -> SQLite memory.db
                         |
                         +-> bounded CoreContextBundle
                         +-> optional RLM/LCM worker runs
```

## What It Is For

MemoryForge focuses on long-form project evidence that is common in real coding
workflows but too large or noisy to paste into every prompt:

- design notes and architecture documents
- requirements and implementation plans
- setup guides and decision records
- benchmark descriptions and experiment logs
- large Markdown files used while "vibe coding" or building a project over many sessions

These sources are ingested through RLM, stored as durable LTM evidence, and
recalled as bounded context when Codex needs them.

MemoryForge intentionally avoids a large AST/code graph schema in the core
release. Source files can still be stored as RLM/LTM evidence, but dedicated
code indexing is future work rather than part of the current SQLite schema.

## Memory Layers

| Layer | Role | Model worker use |
| --- | --- | --- |
| RLM | Raw Large Memory. Chunks large files/prompts and indexes them into durable memory. | Optional, for batch analysis. |
| LTM | Long-Term Memory. Recalls durable evidence across sessions and sources. | No model call. |
| LCM | Lossless Context Management Keeps the active session view bounded with summaries and refs. | Optional, for compaction. |

Important boundary: LCM compacts MemoryForge's SQLite-backed active context. It
does not directly erase Codex's own context window. Codex manages its live
thread and can compact it with `/compact`; MemoryForge hooks and MCP tools then
help preserve and rehydrate the evidence needed after compaction.

## Install

From PyPI in a project that uses `uv`:

```bash
uv add memoryforge
```

With optional local embeddings:

```bash
uv add "memoryforge[embeddings]"
```

For local development from this repository:

```bash
uv sync --extra dev --extra benchmark
```

## Initialize A Codex Project

Run this at the project root:

```bash
uv run memoryforge init . --agent-id default --force
```

This creates:

```text
.memoryforge/memory.db
.memoryforge/config.json
.memoryforge/hooks/memoryforge-hook.sh
.codex/config.toml
.codex/hooks.json
```

The Codex config registers the MemoryForge MCP server:

```toml
[mcp_servers.memoryforge]
command = "uv"
args = ["run", "memoryforge-mcp"]

[mcp_servers.memoryforge.env]
MEMORYFORGE_DB = "/absolute/path/.memoryforge/memory.db"
```

The hook file records prompt and compaction lifecycle events. Project-local
Codex hooks only run after the project `.codex/` layer is trusted by Codex.
Inspect and trust them in the CLI with `/hooks` if Codex asks.

## Prompt Submit, Cancel, And Retract

MemoryForge uses a two-phase hook flow for user prompts:

- `UserPromptSubmit` writes the latest prompt for the session to
  `.memoryforge/pending/`.
- `Stop` commits that pending prompt into SQLite, then RLM/LTM/LCM processing
  can run.
- `SessionStart` removes stale pending prompts after the local TTL.

This keeps interrupted turns out of durable memory when the Codex turn is
cancelled before `Stop` runs. MemoryForge does not infer cancellation from
free-form hook payload fields such as status or reason strings; that is too
fragile across Codex versions and locales. Explicit retract integrations should
call the internal `discard-pending` hook event for the matching session before a
later `Stop` can commit it.

## Basic Usage

Ingest a long Markdown file or project document:

```bash
uv run memoryforge --db .memoryforge/memory.db ingest-file docs/notes.md \
  --agent-id default
```

Load a large source through RLM:

```bash
uv run memoryforge --db .memoryforge/memory.db rlm-load docs/design.md \
  --agent-id default \
  --name design-notes
```

Recall durable evidence:

```bash
uv run memoryforge --db .memoryforge/memory.db recall-memory \
  --agent-id default \
  --query "why did we choose sqlite"
```

Build a runtime context bundle for the active Codex project:

```bash
uv run memoryforge --db .memoryforge/memory.db runtime-context \
  --agent-id default \
  --session-id session-1 \
  --query "what context should Codex use now" \
  --project-root .
```

Run LCM compaction over MemoryForge's stored active context:

```bash
uv run memoryforge --db .memoryforge/memory.db lcm-compact \
  --agent-id default \
  --session-id session-1 \
  --project-root . \
  --force
```

Run the MCP server directly:

```bash
uv run memoryforge-mcp
```

## Optional Vector Recall

MemoryForge works without embeddings by using lexical/FTS retrieval. To enable
semantic recall, install the embeddings extra and choose FastEmbed:

```bash
uv add "memoryforge[embeddings]"

export MEMORYFORGE_VECTOR_BACKEND=fastembed
export MEMORYFORGE_VECTOR_MODEL=BAAI/bge-small-en-v1.5
export MEMORYFORGE_REQUIRE_VECTOR_MODEL=1
```

The project intentionally keeps one vector cache table, `vec_index`, and avoids
SQLite extension backends such as `sqlite-vec` in the core release. This keeps
the package easier to install, test, and publish.

## CLI Surface

Public commands:

- Project/runtime: `init`, `mcp-server`, `runtime-context`
- Conversation memory: `store-session`, `search`, `recall-memory`, `active-recall`, `long-term-source`
- Contradictions: `record-contradiction`, `find-contradictions`
- LCM: `lcm-context`, `lcm-compact`, `lcm-maintain`
- RLM/source loading: `ingest-file`, `rlm-load`, `rlm-search`, `rlm-chunk-get`, `dispatch`, `context-get`, `rlm-record`, `aggregate`, `rlm-run`
- Diagnostics: `chunk`, `benchmark`

`memoryforge hook` is an internal Codex hook endpoint created by `memoryforge init`.
RLM/LCM sub-agents are internal MemoryForge workers. For real worker runs,
MemoryForge uses Codex CLI through `codex exec` when configured. Development-time
Codex host subagents are separate review/triage helpers and are not the
MemoryForge runtime worker API.

## Benchmarks

The current benchmark focus is long-memory behavior, not static code indexing:

- LoCoMo
- LongMemEval
- deterministic multi-session stress benchmark
- synthetic smoke benchmark

Example smoke check:

```bash
uv run python benchmarks/synthetic_test.py
```

Stress check for many real SQLite sessions:

```bash
uv run python benchmarks/stress_sessions.py \
  --sessions 100 \
  --turns-per-session 12 \
  --output benchmarks/results/stress_sessions_100x12.json
```

Real LoCoMo and LongMemEval runs require their datasets and model credentials.
See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for run modes and result fields.

## Development

Run the normal quality gate on Unix-like shells:

```bash
make check
```

Equivalent commands:

```bash
uv run ruff check memoryforge tests benchmarks
PYTHONDONTWRITEBYTECODE=1 uv run mypy memoryforge
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run pytest --ignore=tests/test_real_subagents.py --cov=memoryforge --cov-report=term-missing --cov-fail-under=77
MEMORYFORGE_REAL_SUBAGENT=1 MEMORYFORGE_REAL_PROJECT_ROOT="$PWD" MEMORYFORGE_SUBAGENT_RUNNER=codex MEMORYFORGE_MODEL=gpt-5.4 uv run pytest tests/test_real_subagents.py -vv
uv build
uv run twine check dist/*
```

On Windows PowerShell, keep pytest temp/cache paths in writable directories:

```powershell
$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'
Remove-Item Env:\MEMORYFORGE_SUBAGENT_RUNNER -ErrorAction SilentlyContinue
Remove-Item Env:\MEMORYFORGE_MODEL -ErrorAction SilentlyContinue
uv run pytest --ignore=tests/test_real_subagents.py --basetemp=C:\tmp\memoryforge-pytest-basetemp -o cache_dir=.tmp\pytest-cache

$env:MEMORYFORGE_REAL_SUBAGENT='1'; $env:MEMORYFORGE_REAL_PROJECT_ROOT=(Get-Location).Path
$env:MEMORYFORGE_SUBAGENT_RUNNER='codex'; $env:MEMORYFORGE_MODEL='gpt-5.4'
uv run pytest tests/test_real_subagents.py -vv --basetemp=C:\tmp\memoryforge-pytest-basetemp-real -o cache_dir=.tmp\pytest-cache-real
```

The real Codex sub-agent smoke tests are local-only. Run them on a machine with the Codex CLI installed and authenticated if you want to verify `runner="codex"`; they are not part of CI/CD. Mock runners are only for targeted unit tests that verify MemoryForge's own control flow.

## Release Notes For Maintainers

Before pushing or publishing:

1. Keep generated data out of the release: `.venv/`, caches, `.coverage`, `.memoryforge/`, `.codebase-memory/`, `dist/`, and `benchmarks/results/`.
2. Run the full gate on Python 3.10, 3.11, and 3.12 through CI.
3. Build the wheel and sdist with `uv build`.
4. Check distributions with `twine check`.
5. Prefer the `Publish` GitHub workflow with PyPI trusted publishing. Direct maintainer uploads may use `twine upload` with `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` supplied from the shell environment, never from a committed config file.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/API.md](docs/API.md)
- [docs/RLM.md](docs/RLM.md)
- [docs/SECOND_BRAIN_ROADMAP.md](docs/SECOND_BRAIN_ROADMAP.md)
