# MemoryForge

MemoryForge is a Codex-CLI-first, local-first memory layer for long-running
LLM workflows.

It stores durable project evidence in SQLite, keeps live context bounded, and
returns source-backed context bundles to the model or agent the user is already
running. For Codex CLI usage on Linux/WSL, the project-local hook runner is the
required LCM auto-capture path. Codex MCP remains deliberately small and exposes
only hot-path recall/context plus the RLM analysis planner.

```text
Codex CLI -> WSL/Linux hook runner -> same SQLite memory.db
Codex CLI -> MemoryForge MCP      -> recall/context/index_analyze
Codex host subagents             -> fetch chunks -> record -> aggregate
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
recalled as bounded context when Codex needs them. The shallow path is the RLM
sub-agent analysis/summary stored in LTM; the deep path remains the lossless
`rlm_chunk:<id>` content that can be rehydrated on demand.

MemoryForge intentionally avoids a large AST/code graph schema in the core
release. The default auto-load path targets Markdown project knowledge
(`README`, design notes, ADRs, plans, reports, specs). Code files can still be
ingested manually, but dedicated code indexing is future work rather than part
of the current SQLite schema.

## Memory Layers

| Layer | Role | Model worker use |
| --- | --- | --- |
| RLM | Recursive language model. Chunks large files/prompts, prepares host-subagent analysis plans, and indexes both derived summaries and full chunks into durable memory. | Host agent/subagent executes the returned plan. |
| LTM | Long-Term Memory. Recalls durable evidence across sessions and sources. | No model call. |
| LCM | Lossless Context Management. Keeps the MemoryForge active-session view bounded with summaries, raw refs, and recoverable tool-output parts. | WSL/Linux hook runner captures Codex CLI lifecycle; optional explicit compaction worker. |

Important boundary: LCM compacts MemoryForge's SQLite-backed active context. It
does not directly erase Codex's own context window. Codex manages its live
thread and can compact it with `/compact`; MemoryForge MCP tools and the
WSL/Linux hook runner preserve and rehydrate the evidence needed after
compaction.

## Install

From PyPI in a project that uses `uv`:

```bash
uv add memfg==6.1.5
```

MemoryForge defaults to lexical BM25/FTS recall so first run stays responsive on
fresh PyPI installs and offline machines. Semantic/vector recall is available
when explicitly enabled with `MEMORYFORGE_VECTOR_BACKEND=fastembed`.

For local development from this repository:

```bash
uv sync --extra dev --extra benchmark
```

## PyPI 6.1.5 Post-Publish Smoke Test

After publishing:

```bash
uvx --from twine twine upload dist/memfg-6.1.5*
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
uv add memfg==6.1.5
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

Expected behavior for `6.1.5`:

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

6. Index the Markdown files. The default path is fast raw chunk/LTM indexing and
does not call Codex sub-agents:

```powershell
uv run memoryforge index . `
  --agent-id codex `
  --max-files 1 `
  --force
```

For RLM sub-agent analysis, do not leave Codex to run a separate CLI command.
Start Codex and ask it to use MCP `index_analyze`; the returned
`host_subagent_prompt` values are the host-subagent tasks.

`init --index` remains accepted for compatibility, but the recommended command
is `memoryforge index`.

7. Verify recall without Codex:

```powershell
uv run memoryforge --db .memoryforge\memory.db recall-memory `
  --agent-id codex `
  --query "telemetry ingress endpoint facilities old value rejected" `
  --include-content
```

The output should include `https://telemetry.facilities.example.com/v2/ingest`
and the rejection reason about tenant isolation and TLS pinning.

8. Install and trust the Codex lifecycle hook from WSL/Linux if you want LCM to
capture interactive Codex turns:

```bash
uv run memoryforge init . --agent-id codex --force --install-hooks
codex
```

Inside Codex, run `/hooks` and trust the MemoryForge project-local hooks. Without
this trust step, LCM capture is unavailable for the Codex session.

9. Optional: verify recall and RLM planning through Codex MCP. Start Codex from
the same project directory:

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

To make Codex prepare RLM host-subagent work from inside the active session, ask:

```text
Use MemoryForge MCP index_analyze for this project with analyze_min_bytes=20000 and analyze_max_files=5, then dispatch the returned host_subagent_prompt batches.
```

10. Verify LCM lifecycle capture for Codex interactive mode. Use WSL/Linux
rather than native PowerShell hooks:

```bash
uv run memoryforge init . --agent-id codex --force --install-hooks
```

Expected files:

```text
.codex/hooks.json
.memoryforge/hooks/memoryforge-hook.sh
.memoryforge/hooks/memoryforge-hook.log
```

Restart Codex from the same WSL/Linux project directory:

```bash
codex
```

Then run `/hooks` in Codex and trust the MemoryForge hook definitions. Codex
requires this review for project-local command hooks. The MemoryForge hook is
intentionally small: it calls `python -m memoryforge.cli.main hook ...` through
the project Python when available, uses `uv run --no-sync` as a fallback, and
has a 30 second timeout. It does not call `codex`, does not start a model
worker, does not use the Codex account/token, does not sync/reinstall the
project, and does not index the whole project on startup unless
`MEMORYFORGE_HOOK_AUTO_INDEX=1` is set. Diagnostics go to
`.memoryforge/hooks/memoryforge-hook.log`.

If you previously installed hooks with an older MemoryForge build and Codex
shows `SessionStart hook (failed)` or `UserPromptSubmit hook (failed)`, remove
the old native Windows hook files and regenerate from WSL/Linux:

```bash
uv run memoryforge init . --agent-id codex --force --install-hooks
```

Then reopen Codex from WSL/Linux and trust the hooks again with `/hooks`.

The hook path listens for:

- `SessionStart`: cleans stale pending hook files. It does not auto-index Markdown by default.
- `UserPromptSubmit`: stores the pending user prompt and records an LCM context snapshot.
- `PostToolUse`: stores tool output as an assistant message with a `tool` part when Codex supplies tool output in the hook payload.
- `Stop`: commits the completed turn; if Codex supplies assistant output, it stores user + tool + assistant, otherwise it still commits the user prompt so the session is not empty.
- `PreCompact` and `PostCompact`: record context snapshots around Codex `/compact`; `PostCompact` also stores a compact summary if Codex supplies one.

Ask Codex one real question, then inspect the MemoryForge LCM database:

```powershell
uv run memoryforge --db .memoryforge\memory.db lcm-sessions `
  --agent-id codex

uv run memoryforge --db .memoryforge\memory.db lcm-messages `
  --session-id <session-id-from-lcm-sessions> `
  --agent-id codex `
  --include-content

uv run memoryforge --db .memoryforge\memory.db lcm-context `
  --session-id <session-id-from-lcm-sessions>
```

If `lcm-sessions` shows `message_count: 0`, the hook was not trusted, Codex was
not restarted after installing hooks, or Codex did not run from the initialized
project directory.

11. Optional semantic vector recall. Leave this off for the first smoke test. To
enable FastEmbed later, set the environment before indexing and before starting
Codex:

```powershell
$env:MEMORYFORGE_VECTOR_BACKEND='fastembed'
$env:MEMORYFORGE_VECTOR_MODEL='BAAI/bge-small-en-v1.5'
```

Without those variables, `6.1.5` uses lexical BM25/FTS recall by default so
first run stays responsive on Windows and offline machines.

## Optional Codex MCP Adapter

For Codex CLI recall/context tools, register the MemoryForge MCP server:

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

MemoryForge does not create project-local `.codex/` files or install Codex
hooks unless `--install-hooks` is requested. It also does not call Codex CLI
subprocesses during init or project indexing. The MCP adapter intentionally
exposes only the hot-path tools:

- `recall_memory`: fast factual recall from durable RLM/LTM indexes
- `build_context_bundle`: grounded LCM/LTM context assembly for the active model
- `index_analyze`: index Markdown and return host-subagent RLM analysis plans

LCM completed-turn capture is not exposed through MCP. For Codex CLI, install
and trust the WSL/Linux hook runner so lifecycle capture is continuous instead
of relying on ad-hoc tool calls.

`index_analyze` mirrors the internal RLM planner: it chunks/indexes selected
Markdown files, writes raw `rlm_chunk:<id>` evidence into LTM, and returns
`plans[].batches[].host_subagent_prompt` for the active Codex host to dispatch.
It does not call a model or spawn `codex exec`.

## WSL/Linux Hook Auto-Capture

Use hooks when you need Codex interactive mode to auto-capture prompts, tool
outputs, stop events, and compaction snapshots into LCM. For reliability, run
Codex from WSL/Linux and install hooks there:

```bash
uv run memoryforge init . --agent-id codex --force --install-hooks
```

This creates `.codex/hooks.json` plus a tiny local runner:

```text
.memoryforge/hooks/memoryforge-hook.sh
.memoryforge/hooks/memoryforge-hook.log
```

After restarting Codex from the same WSL/Linux project directory, run `/hooks`
and trust the MemoryForge hook definitions. Without that trust step, Codex will
skip project-local command hooks.

The hook is intentionally local-only:

- It calls `python -m memoryforge.cli.main hook ...`.
- It does not call `codex`.
- It does not call a model.
- It does not run `codex exec`.
- It does not depend on the active Codex account or OAuth token.
- It exits `0` and writes diagnostics to `.memoryforge/hooks/memoryforge-hook.log`.

LCM lifecycle capture is additive. RLM/LTM ingestion and recall keep working the
same way; hooks only append active-session turns and context snapshots into the
LCM tables.

## Basic Usage

Default Codex CLI usage is WSL/Linux hook lifecycle capture plus MCP recall,
context, and RLM planning. The CLI commands below are the direct/manual surface
for indexing, recall, context inspection, and maintenance.

Ingest a long Markdown file or project document:

```bash
uv run memoryforge --db .memoryforge/memory.db ingest-file docs/notes.md \
  --agent-id codex
```

Index project Markdown quickly:

```bash
uv run memoryforge index . \
  --agent-id codex
```

Run RLM analysis only for large Markdown files from inside Codex through MCP:

```text
Use MemoryForge MCP index_analyze for this project with analyze_min_bytes=20000 and analyze_max_files=5, then dispatch the returned host_subagent_prompt batches.
```

MCP `index_analyze` does not spawn `codex exec` or any external model process.
It returns `plans[]` with:

- `host_subagent_prompt`: prompt for the active Codex host to give to a subagent
- `fetch_command_argvs`: exact chunk fetch commands
- `record_command_argv`: exact command to record that batch analysis
- `aggregate_command_argv`: final aggregation command with `--expected-batches`

The intended flow inside Codex is:

1. Call MCP `memoryforge.index_analyze`.
2. For each returned batch, spawn a Codex host subagent using `host_subagent_prompt`.
3. Each subagent fetches only its chunks and writes a concise cited analysis.
4. Record each analysis with the returned `record_command_argv`.
5. Run `aggregate_command_argv` after all planned batches are recorded.

For targeted/debug usage, `rlm-run` still exists as a legacy advanced CLI
command, but it is not the primary indexing path and is intentionally omitted
from the quickstart. Prefer MCP `index_analyze` plus host subagents.

The primary `index` path chunks sources losslessly and stores exact
`rlm_chunk:<id>` refs for deep rehydration. MCP `index_analyze` prepares
host-subagent analysis work and stores `rlm_analysis`/`rlm_summary` rows in LTM
after the returned record/aggregate commands are run.

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

Inspect the active LCM state before or after compaction:

```bash
uv run memoryforge --db .memoryforge/memory.db lcm-sessions \
  --agent-id codex

uv run memoryforge --db .memoryforge/memory.db lcm-messages \
  --session-id session-1 \
  --agent-id codex \
  --include-content

uv run memoryforge --db .memoryforge/memory.db lcm-summary \
  --session-id session-1
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
- LCM: `lcm-context`, `lcm-sessions`, `lcm-messages`, `lcm-summary`, `lcm-compact`, `lcm-maintain`
- RLM/source loading: `ingest-file`, `rlm-load`, `rlm-search`, `rlm-chunk-get`, `dispatch`, `context-get`, `rlm-record`, `aggregate`, `rlm-run`
- Diagnostics: `chunk`, `benchmark`

MCP intentionally exposes only `recall_memory`, `build_context_bundle`, and
`index_analyze`. Low-level RLM/debug commands stay on the CLI so their schemas
are not injected into every Codex turn.

`memoryforge hook` remains available as an internal endpoint for direct testing.
RLM host-subagent analysis is coordinated by MCP `index_analyze`: the active
Codex host owns subagent execution, while MemoryForge owns chunk storage,
record, aggregate, and recall.

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
