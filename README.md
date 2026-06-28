# MemoryForge

MemoryForge is an MCP-first, local-first memory layer for Codex CLI workflows.

It stores durable project evidence in SQLite, keeps live context bounded, and
returns source-backed context bundles to the model the user is already running.
MemoryForge is not a separate answering agent and it is not a codebase AST graph
engine. The host model answers; MemoryForge supplies memory, references, and
provenance.

```text
User -> Codex CLI -> MemoryForge MCP -> SQLite memory.db
                         |
                         +-> bounded CoreContextBundle
                         +-> RLM/LTM retrieval
                         +-> optional LCM compaction
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
release. The default auto-load path targets Markdown project knowledge
(`README`, design notes, ADRs, plans, reports, specs). Code files can still be
ingested manually, but dedicated code indexing is future work rather than part
of the current SQLite schema.

## Memory Layers

| Layer | Role | Model worker use |
| --- | --- | --- |
| RLM | Raw Large Memory. Chunks large files/prompts and indexes them into durable memory. | Optional, for batch analysis. |
| LTM | Long-Term Memory. Recalls durable evidence across sessions and sources. | No model call. |
| LCM | Lossless Context Management Keeps the active session view bounded with summaries and refs. | Optional, for compaction. |

Important boundary: LCM compacts MemoryForge's SQLite-backed active context. It
does not directly erase Codex's own context window. Codex manages its live
thread and can compact it with `/compact`; MemoryForge MCP tools then help
preserve and rehydrate the evidence needed after compaction.

## Install

From PyPI in a project that uses `uv`:

```bash
uv add memfg==6.1.1
```

MemoryForge defaults to lexical BM25/FTS recall so first run stays responsive on
fresh PyPI installs and offline machines. Semantic/vector recall is available
when explicitly enabled with `MEMORYFORGE_VECTOR_BACKEND=fastembed`.

For local development from this repository:

```bash
uv sync --extra dev --extra benchmark
```

## PyPI 6.1.1 Post-Publish Smoke Test

After publishing:

```bash
uv run twine upload dist/memfg-6.1.1*
```

use a clean project to verify the package from PyPI. The flow below is written
for Windows PowerShell, but the same commands work in any shell after adapting
path syntax.

1. Create a fresh uv project outside this repository:

```powershell
New-Item -ItemType Directory -Force C:\tmp\memoryforge-pypi-smoke | Out-Null
Set-Location C:\tmp\memoryforge-pypi-smoke
uv init --bare --name memoryforge-pypi-smoke .
```

2. Install the freshly published PyPI package:

```powershell
uv add memfg==6.1.1
uv run memoryforge --help
```

`uv run memoryforge --help` must print CLI usage and exit. Do not use
`uv run memoryforge-mcp` as a normal smoke test; that command starts the MCP
stdio server and waits for a client, so an idle terminal there is expected.

3. Register the MCP server with Codex CLI:

```powershell
codex mcp remove memoryforge
codex mcp add memoryforge -- uv run memoryforge-mcp
codex mcp get memoryforge --json
```

If `codex mcp remove memoryforge` says the server does not exist, continue with
the `codex mcp add` command. Restart Codex after changing MCP registration.

4. Initialize MemoryForge in the project:

```powershell
uv run memoryforge init . --agent-id codex --force
```

Expected behavior for `6.1.1`:

- The command exits by itself.
- `.memoryforge\memory.db` is created.
- `.memoryforge\config.json` is created.
- `AGENTS.md` is created or updated.
- The JSON output shows `"indexed": {"enabled": false, ...}`.
- The JSON output shows `"codex /init not requested"` unless you passed
  `--configure-codex`.

`init` is intentionally lightweight. It does not index Markdown by default, and
it does not call Codex CLI subprocesses by default.

5. Add a small Markdown memory file:

```powershell
New-Item -ItemType Directory -Force docs | Out-Null
@'
# Facilities telemetry

Facilities use telemetry ingress endpoint https://telemetry.facilities.example.com/v2/ingest.

The old endpoint https://telemetry.old.example.com/ingest was rejected because it bypassed tenant isolation and failed TLS pinning.
'@ | Set-Content -Encoding UTF8 docs\telemetry.md
```

6. Index the Markdown file. For a direct CLI smoke test:

```powershell
uv run memoryforge --db .memoryforge\memory.db ingest-file docs\telemetry.md --agent-id codex
```

For a project-wide Markdown refresh instead, use:

```powershell
uv run memoryforge init . --agent-id codex --force --index
```

or ask Codex to call the MCP `autoload_markdown` tool after the MCP server is
registered.

7. Verify recall without Codex:

```powershell
uv run memoryforge --db .memoryforge\memory.db recall-memory `
  --agent-id codex `
  --query "telemetry ingress endpoint facilities old value rejected" `
  --include-content
```

The output should include `https://telemetry.facilities.example.com/v2/ingest`
and the rejection reason about tenant isolation and TLS pinning.

8. Verify recall through Codex MCP. Start Codex from the same project directory:

```powershell
codex
```

Ask:

```text
What is the telemetry ingress endpoint used by facilities, and why was the old value rejected?
```

Expected Codex behavior:

- It should call `memoryforge.recall_memory`.
- The tool call should return quickly for this small project.
- The answer should cite the new endpoint and explain why the old value was
  rejected.

If Codex does not call MemoryForge, ask explicitly:

```text
Use MemoryForge MCP recall_memory first. What is the telemetry ingress endpoint used by facilities, and why was the old value rejected?
```

9. Optional semantic vector recall. Leave this off for the first smoke test. To
enable FastEmbed later, set the environment before indexing and before starting
Codex:

```powershell
$env:MEMORYFORGE_VECTOR_BACKEND='fastembed'
$env:MEMORYFORGE_VECTOR_MODEL='BAAI/bge-small-en-v1.5'
```

Without those variables, `6.1.1` uses lexical BM25/FTS recall by default so
first run stays responsive on Windows and offline machines.

## Initialize A Codex Project

First register the MemoryForge MCP server with Codex CLI. This uses Codex's own MCP manager instead of MemoryForge writing `.codex/config.toml`:

```bash
codex mcp add memoryforge -- uv run memoryforge-mcp
```

Then run MemoryForge init at the project root:

```bash
uv run memoryforge init . --agent-id codex --force
```

This creates:

```text
.memoryforge/memory.db
.memoryforge/config.json
AGENTS.md
```

During init, MemoryForge creates or updates the root `AGENTS.md` with a guarded
MemoryForge instruction block:

```text
<!-- MemoryForge instructions start -->
...
<!-- MemoryForge instructions end -->
```

MemoryForge no longer creates project-local `.codex/` files and does not install Codex hooks. It also does not call Codex CLI subprocesses unless you pass `--configure-codex`. The default workflow is MCP/tool-first:

- `recall_memory`: fast factual recall from durable RLM/LTM indexes
- `build_context_bundle`: grounded LCM/LTM context assembly for the active model
- `autoload_markdown`: explicit refresh for changed Markdown files
- `ensure_project_memory`: lightweight project-state check; it does not auto-index unless explicitly requested
- `rlm_load`, `rlm_search`, `rlm_chunk_get`, `rlm_run`: large-context RLM workflows, with `rlm_run` reserved for explicit sub-agent analysis

Init is intentionally lightweight by default. To index project Markdown during
init, add `--index`, or run `autoload_markdown` from MCP after startup.
`AGENTS.md` itself is skipped during autoload so instructions do not pollute
project memory.
## Basic Usage

Default project usage is through Codex + MCP after `memoryforge init`. The CLI
commands below are the direct/manual surface when you want to run MemoryForge
outside the normal Codex tool loop.

Ingest a long Markdown file or project document:

```bash
uv run memoryforge --db .memoryforge/memory.db ingest-file docs/notes.md \
  --agent-id codex
```

Load a large source through RLM:

```bash
uv run memoryforge --db .memoryforge/memory.db rlm-load docs/design.md \
  --agent-id codex \
  --name design-notes
```

Recall durable evidence:

```bash
uv run memoryforge --db .memoryforge/memory.db recall-memory \
  --agent-id codex \
  --query "why did we choose sqlite"
```

Build a runtime context bundle for the active Codex project:

```bash
uv run memoryforge --db .memoryforge/memory.db runtime-context \
  --agent-id codex \
  --session-id session-1 \
  --query "what context should Codex use now" \
  --project-root .
```

Run LCM compaction over MemoryForge's stored active context:

```bash
uv run memoryforge --db .memoryforge/memory.db lcm-compact \
  --agent-id codex \
  --session-id session-1 \
  --project-root . \
  --force
```

Run the MCP server directly:

```bash
uv run memoryforge-mcp
```

## Vector Recall

MemoryForge uses lexical BM25/FTS recall by default. To enable semantic vector
recall with FastEmbed and store local embeddings in `vec_index`, configure:

```bash
export MEMORYFORGE_VECTOR_BACKEND=fastembed
export MEMORYFORGE_VECTOR_MODEL=BAAI/bge-small-en-v1.5
```

For explicit lexical-only fallback, set `MEMORYFORGE_VECTOR_BACKEND=disabled`.
The project intentionally keeps one vector cache table, `vec_index`, and avoids
SQLite extension backends such as `sqlite-vec` in the core release. This keeps
the package easier to install, test, and publish.

Retrieval is hybrid by design: vector recall and lexical recall can both
contribute candidates, and MemoryForge fuses bounded evidence for the runtime
context instead of relying on a vector-only path.

## CLI Surface

Public commands:

- Project/runtime: `init`, `mcp-server`, `runtime-context`
- Conversation memory: `store-session`, `search`, `recall-memory`, `active-recall`, `long-term-source`
- Contradictions: `record-contradiction`, `find-contradictions`
- LCM: `lcm-context`, `lcm-compact`, `lcm-maintain`
- RLM/source loading: `ingest-file`, `rlm-load`, `rlm-search`, `rlm-chunk-get`, `dispatch`, `context-get`, `rlm-record`, `aggregate`, `rlm-run`
- Diagnostics: `chunk`, `benchmark`

`memoryforge hook` remains available as an internal endpoint for direct testing.
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
