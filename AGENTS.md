# MemoryForge Codex Guide

## Project Shape

MemoryForge is a Python package for local-first memory in Codex CLI workflows.
Core package code lives under `memoryforge/`; tests live under `tests/`; benchmark
entry points live under `benchmarks/`; user docs live under `docs/`.

The package has two different "subagent" contexts:

- Product/runtime subagents: MemoryForge code uses `memoryforge/agents/` to run
  worker prompts through `codex exec` or another configured backend. This is the
  path users can run from the CLI and the path package tests should cover.
- Development-time Codex subagents: when working inside a Codex session, the host
  environment may expose explorer/worker agents for parallel review or disjoint
  implementation. This is a workflow aid for maintainers, not a MemoryForge API.

For MemoryForge product behavior, keep the CLI path simple: prefer `codex exec`
through `CodexSubAgentRunner` unless the task is explicitly about another runner.
Use development-time subagents only to help review or implement this repository.

## Development Commands

Prefer the local virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m ruff check memoryforge tests benchmarks
.\.venv\Scripts\python.exe -m mypy memoryforge
.\.venv\Scripts\python.exe -m pytest
```

The `Makefile` is Unix-oriented. In PowerShell, run the Python commands directly
or use `uv run` equivalents.

When pytest cannot write to the system temp directory, use a workspace cache
location:

```powershell
New-Item -ItemType Directory -Force .tmp\pytest-cache | Out-Null
.\.venv\Scripts\python.exe -m pytest -o cache_dir=.tmp\pytest-cache
```

Some tests use `tmp_path`; if Windows/OneDrive blocks pytest base temp creation,
that is an environment issue rather than an assertion failure.

Real Codex sub-agent tests are local-only and should run separately against `tests/test_real_subagents.py` when you need to verify `runner="codex"`. Set `MEMORYFORGE_REAL_SUBAGENT=1`, `MEMORYFORGE_REAL_PROJECT_ROOT`, `MEMORYFORGE_SUBAGENT_RUNNER=codex`, and an explicit `MEMORYFORGE_MODEL`; do not set `MEMORYFORGE_SUBAGENT_RUNNER=mock` for that local verification.

## Development-Time Subagent Workflow

Use explorer agents for bounded read-only questions such as:

- package/config/CI review
- runtime/compaction/subagent behavior review
- identifying tests that cover a proposed change

Use worker agents only for disjoint write scopes. Tell workers they are not alone
in the codebase, to avoid reverting unrelated edits, and to report changed paths.

Good splits:

- `memoryforge/init/*` plus `tests/test_init.py`
- `memoryforge/agents/*` plus `tests/test_subagents.py` and
  `tests/test_real_subagents.py`
- `memoryforge/lcm/compaction/*` plus LCM runtime tests
- docs-only updates under `README.md` and `docs/*`

Avoid sending two workers into the same module at the same time.

## Review Priorities

For code review, lead with concrete findings and file:line references. Prioritize:

- broken user setup paths for `memoryforge init`
- Codex CLI command construction and model selection
- durable memory correctness and provenance
- Windows compatibility, especially temp files and shell hooks
- tests that pass only because the mock runner hides real Codex runner behavior

Do not commit `.memoryforge/`, `.codex/`, virtualenvs, cache directories, benchmark
results, or local temp files.