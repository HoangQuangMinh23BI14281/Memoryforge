# MemoryForge Benchmarks

Benchmarks must prove the MemoryForge pipeline, not hide answering inside a
worker. Use separate modes so ingestion cost, context assembly, core answer
latency, and worker latency are visible.

## Boundary

MemoryForge has three benchmark responsibilities:

1. Load evidence through RLM when the mode requires ingestion.
2. Retrieve durable memory through LTM and build an LCM context bundle.
3. Hand that bundle to the configured core answer runner only in answer modes.

RLM and LCM worker modes test internal worker behavior only. LTM has no model
call.

```text
dataset -> RLM ingest -> LTM index
question -> LTM recall + LCM context -> CoreContextBundle
CoreContextBundle -> core answer runner only in core-answer mode
```

## Modes

| Mode | What It Tests | Uses Answer Runner? | Uses RLM Worker? | Uses LCM Worker? |
| --- | --- | --- | --- | --- |
| `ingest-only` | RLM loads sources and LTM indexes chunks. | No | No | No |
| `context-only` | Query-time recall and context bundle assembly. | No | No | No |
| `core-answer` | Accuracy after the core answer runner reads the bundle. | Yes | No | Only for special LCM probes |
| `rlm-worker` | RLM sub-agent dispatch behavior. | No | Yes | No |
| `lcm-worker` | LCM compaction sub-agent behavior. | No | No | Yes |

Every LongMemEval result includes `diagnostics.mode_contract`. If that contract
does not match the mode, the result is not useful.

## Real LongMemEval

Real answer run, with a real vector model and no hashing fallback:

```bash
MEMORYFORGE_MODEL=gpt-5.2 \
MEMORYFORGE_SUBAGENT_RUNNER=codex \
MEMORYFORGE_VECTOR_BACKEND=fastembed \
MEMORYFORGE_VECTOR_MODEL=BAAI/bge-small-en-v1.5 \
MEMORYFORGE_REQUIRE_VECTOR_MODEL=1 \
uv run python benchmarks/longmemeval_benchmark.py \
  /mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource/longmemeval_s_cleaned.json \
  --mode core-answer \
  --limit 40 \
  --special-lcm-probes 10 \
  --clean-output \
  --output benchmarks/results/longmemeval_rlm_ltm_lcm_gpt52_50.json \
  --jsonl-output benchmarks/results/longmemeval_rlm_ltm_lcm_gpt52_50.jsonl
```

Context-only run for MemoryForge latency without answer model latency. Run
`ingest-only` first on the same DB; `context-only` refuses to ingest missing
sources.

```bash
MEMORYFORGE_VECTOR_BACKEND=fastembed \
MEMORYFORGE_VECTOR_MODEL=BAAI/bge-small-en-v1.5 \
MEMORYFORGE_REQUIRE_VECTOR_MODEL=1 \
uv run python benchmarks/longmemeval_benchmark.py \
  /mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource/longmemeval_s_cleaned.json \
  --mode ingest-only \
  --limit 40 \
  --special-lcm-probes 10 \
  --db /tmp/memoryforge_longmemeval_context_fastembed.db

MEMORYFORGE_VECTOR_BACKEND=fastembed \
MEMORYFORGE_VECTOR_MODEL=BAAI/bge-small-en-v1.5 \
MEMORYFORGE_REQUIRE_VECTOR_MODEL=1 \
uv run python benchmarks/longmemeval_benchmark.py \
  /mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource/longmemeval_s_cleaned.json \
  --mode context-only \
  --limit 40 \
  --special-lcm-probes 10 \
  --db /tmp/memoryforge_longmemeval_context_fastembed.db \
  --output benchmarks/results/longmemeval_context_fastembed_50.json \
  --jsonl-output benchmarks/results/longmemeval_context_fastembed_50.jsonl
```

RLM worker run for sub-agent behavior only:

```bash
MEMORYFORGE_MODEL=gpt-5.2 \
MEMORYFORGE_SUBAGENT_RUNNER=codex \
MEMORYFORGE_VECTOR_BACKEND=fastembed \
MEMORYFORGE_VECTOR_MODEL=BAAI/bge-small-en-v1.5 \
MEMORYFORGE_REQUIRE_VECTOR_MODEL=1 \
uv run python benchmarks/longmemeval_benchmark.py \
  /mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource/longmemeval_s_cleaned.json \
  --mode rlm-worker \
  --limit 10 \
  --special-lcm-probes 0 \
  --rlm-max-workers 4
```

The script refuses mock runners in modes that require a runner and requires a
real vector model; hashing is not accepted as a semantic vector backend.

## LoCoMo

```bash
uv run python benchmarks/locomo_benchmark.py \
  /mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource/hindsight/hindsight-dev/benchmarks/locomo/datasets/locomo10.json \
  --mode ingest-only \
  --limit 25 \
  --top-k 10 \
  --vector-backend fastembed \
  --vector-model BAAI/bge-small-en-v1.5 \
  --require-vector-model \
  --db /tmp/memoryforge_locomo_context_fastembed.db

uv run python benchmarks/locomo_benchmark.py \
  /mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource/hindsight/hindsight-dev/benchmarks/locomo/datasets/locomo10.json \
  --mode context-only \
  --limit 25 \
  --top-k 10 \
  --vector-backend fastembed \
  --vector-model BAAI/bge-small-en-v1.5 \
  --require-vector-model \
  --db /tmp/memoryforge_locomo_context_fastembed.db \
  --output benchmarks/results/locomo_context_fastembed_25.json \
  --jsonl-output benchmarks/results/locomo_context_fastembed_25.jsonl
```

LoCoMo also supports `--mode adapter-search` for the older retrieval smoke path.
Use LongMemEval for the stricter RLM/LTM/LCM mode contract.

## Deterministic Second-Brain Check

This benchmark does not use an answer model. It checks typed durable memory,
active recall, corrections, contradictions, provenance, and core-model context
injection.

```bash
uv run memoryforge --db .memoryforge/memory.db benchmark \
  --dataset second-brain \
  --mode context-only
```

Use `--mode ingest-only` to verify durable memory ingestion without building a
context bundle.

## Multi-Session Stress Check

Use this before release to exercise many SQLite sessions, LTM rows, RLM chunks,
context bundle builds, and query latency without calling an answer model:

```bash
uv run python benchmarks/stress_sessions.py \
  --sessions 100 \
  --turns-per-session 12 \
  --output benchmarks/results/stress_sessions_100x12.json
```

The output reports table counts, ingest latency, query latency, context tokens,
raw refs, and long-term hits per query. It is a stress and regression gate, not
an accuracy benchmark.

## What To Inspect

For real LongMemEval or LoCoMo context output, inspect:

- `summary.mode_contract`
- `summary.vector_model` and `summary.vector_backend`
- `summary.performance`
- `results[].diagnostics.ingestion`
- `results[].diagnostics.ingestion_manifest`
- `results[].diagnostics.context_bundle`
- `results[].diagnostics.context_bundle.raw_refs`
- `results[].diagnostics.context_bundle.provenance`
- `results[].diagnostics.retrieval`
- `results[].diagnostics.dag`
- `results[].core_answer_runner`

The important latency split is:

- ingestion/setup latency
- retrieval and context assembly latency
- LCM compaction worker latency, when used
- RLM worker latency, when used
- core answer runner latency

Do not use a single end-to-end latency number to judge MemoryForge speed.

## Current Local Evidence

Generated on 2026-06-22 from real dataset files under
`/mnt/c/Users/ADMIN/OneDrive/Desktop/Intership/outsource`.

| Result file | Scope | Vector | Key evidence |
| --- | --- | --- | --- |
| `benchmarks/results/longmemeval_core_gpt52_promptfix_5.json` | LongMemEval `core-answer`, 5 real dataset cases, real Codex runner `gpt-5.2` | `fastembed`, `BAAI/bge-small-en-v1.5` | Full RLM -> LTM -> LCM -> core-answer path with no mock runner. Exact score: 5/5; true misses: 0. MemoryForge query/context avg: 208.85ms, p50: 167.97ms; retrieval avg: 67.69ms; context build avg: 4.70ms. Core answer latency is separated: avg 24.57s. Context tokens avg: 2995.8; raw refs avg: 17.2. |
| `benchmarks/results/longmemeval_core_gpt52_answer_range_6_10.json` | LongMemEval `core-answer`, dataset cases 6-10, real Codex runner `gpt-5.2` | `fastembed`, `BAAI/bge-small-en-v1.5` | Full RLM -> LTM -> LCM -> core-answer path with answer-evidence snippets centered on provenance ranges. Exact score: 5/5; true misses: 0. MemoryForge query/context avg: 411.49ms, p50: 301.81ms; retrieval avg: 207.55ms; context build avg: 4.54ms. Core answer latency is separated: avg 18.00s. Context tokens avg: 3526.2; raw refs avg: 17.4. |
| `benchmarks/results/longmemeval_core_gpt52_promptfix_case3.json` | LongMemEval `core-answer` regression case `51a45a95`, real Codex runner `gpt-5.2` | `fastembed`, `BAAI/bge-small-en-v1.5` | Verifies the core-answer runner uses adjacent same-session user context for implicit facts instead of incidental storage/location mentions. Exact score: 1/1; expected `Target`; MemoryForge query/context: 418.24ms; answer latency: 23.25s. |
| `benchmarks/results/longmemeval_core_gpt52_after_annotation_1.json` | LongMemEval `core-answer` regression case `e47becba`, real Codex runner `gpt-5.2` | `fastembed`, `BAAI/bge-small-en-v1.5` | Verifies LongMemEval RLM chunks are annotated from dataset session provenance and the previously missed degree answer is recovered. Exact score: 1/1; expected `Business Administration`; MemoryForge query/context: 330.50ms; answer latency: 28.63s. |
| `benchmarks/results/longmemeval_context_fastembed_5.json` | LongMemEval `context-only`, 5 cases, after separate `ingest-only` on a fresh DB | `fastembed`, `BAAI/bge-small-en-v1.5` | `mode_contract.ingests=false`; every case reports `ingestion.performed=false`, `preingested_ltm_count>=187`, raw refs and provenance, no answer runner/model. MemoryForge query/context avg: 256.48ms, p50: 270.85ms; retrieval avg: 102.66ms; context build avg: 6.36ms. Context tokens avg: 2883.8; raw refs avg: 14.8. |
| `benchmarks/results/locomo_context_fastembed_5.json` | LoCoMo `context-only`, 5 QA cases, after separate `ingest-only` on a fresh DB | `fastembed`, `BAAI/bge-small-en-v1.5` | `mode_contract.ingests=false`; every case reports `ingestion.performed=false`, `preingested=true`, `ingestion.latency_ms=0`, raw refs and provenance, no answer runner/model. MemoryForge query/context avg: 151.66ms; retrieval avg: 89.80ms; context build avg: 4.77ms. Context tokens avg: 776.8; raw refs avg: 7.0. |
| `benchmarks/results/locomo_fastembed_5.json` | LoCoMo adapter smoke, 5 cases | `fastembed`, `BAAI/bge-small-en-v1.5` | Adapter-search latency avg: 42.79ms; 1/5 exact retrieval contains the expected answer string. |

These are smoke/subset runs, not the full benchmark refresh. They prove the
current result schema, no-ingest context-only contract, real-vector path, and a
real LongMemEval core-answer path are working across the first 10 dataset cases.
They do not complete the roadmap requirement for full current LongMemEval/LoCoMo
evidence.

## Invariants

- Re-running unchanged ingestion should report dedupe rather than creating new
  RLM buffers, chunks, LTM items, or vector rows.
- `context-only` must require pre-ingested sources and report
  `ingestion.performed=false`.
- `context-only` must not call a core answer runner.
- `context-only` must not force RLM or LCM workers.
- `core-answer` must report the answer runner separately from MemoryForge
  retrieval/context time.
- `rlm-worker` and `lcm-worker` are worker tests, not accuracy tests.
- Hashing vectors are allowed only for local development, never for serious
  retrieval evaluation.
