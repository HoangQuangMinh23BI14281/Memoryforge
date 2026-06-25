# MemoryForge Second Brain Roadmap

This document captures the optimization plan and the architecture direction needed
to move MemoryForge beyond Agentic RAG.

## Core Principle

MemoryForge should not be the answer model.

The core model is the model the user is already working with in the active runtime,
for example Codex CLI, Claude Code, Cursor, or another agent shell. MemoryForge
should prepare durable memory, working context, summaries, and citations for that
core model.

Sub-agents are allowed only where they are part of the memory machinery:

- RLM: batch analysis, recursive reduction, synthesis over large source buffers.
- LCM: context compaction, summary DAG creation, stale context compression.
- LTM: no model call. It indexes, retrieves, locally selects, and formats memory
  for the core model.

The correct question-time flow is:

```text
user asks core model
  -> MemoryForge retrieves from LTM
  -> MemoryForge builds LCM context with summaries and refs
  -> core model reads that context
  -> core model answers the user
```

If a benchmark uses a model to produce an answer, that runner must be named as a
benchmark/core answer runner, not as an LTM or RLM/LCM sub-agent.

## Schema Discipline

Prefer larger logic layers over larger database schema.

The current schema should remain the source of truth for the near term:

- `content_store`
- `sessions`
- `messages`
- `context_items`
- `summary_nodes`
- `rlm_buffers`
- `rlm_chunks`
- `long_term_items`
- `vec_index`
- `search_fts`

New second-brain behavior should start as code and metadata conventions, not new
tables. Use existing `metadata` JSON columns where possible.

Examples:

- `metadata.kind`: `fact`, `preference`, `decision`, `task`, `correction`,
  `entity_profile`, `project_state`.
- `metadata.confidence`: confidence score or bucket.
- `metadata.freshness`: freshness label or timestamp.
- `metadata.raw_refs`: raw immutable refs backing the memory.
- `metadata.supersedes`: refs or item IDs this memory replaces.
- `metadata.contradicts`: refs or item IDs this memory conflicts with.
- `metadata.valid_from` / `metadata.valid_to`: temporal validity.
- `metadata.answer_evidence`: benchmark/adapter evidence marker when the source
  format explicitly exposes answer-bearing turns or dialogue IDs.

Only promote metadata into new indexed columns or tables when a measured hot path
requires it.

Acceptable future exceptions:

- generated columns or indexes for heavily queried metadata keys
- rebuildable FTS/vector indexes that do not become source-of-truth tables
- a relation table only if contradiction/entity/task graph traversal becomes a
  proven bottleneck

Rebuildable indexes are allowed. They are not source-of-truth schema.

## Core Context Bundle Contract

The primary question-time output of MemoryForge should be a context bundle for the
active core model.

Suggested bundle fields:

- `messages`: LLM-ready messages to inject or prepend.
- `active_recall`: proactive typed memory surfaced for the current session and
  project focus.
- `long_term_recall`: ranked LTM hits with raw refs.
- `summary_nodes`: active LCM summary DAG nodes included.
- `raw_refs`: immutable source refs required to audit the answer.
- `token_estimate`: estimated context cost.
- `budget`: budget settings used to build the context.
- `provenance`: source type, source ID, timestamp, confidence, freshness.
- `diagnostics`: BM25/vector counts, local selection signals, truncation,
  latency.

The bundle is not an answer. The active core model reads the bundle and answers.

## Implementation Status

Status as of 2026-06-21: MemoryForge has moved beyond a plain RAG loop in its
architecture and local memory lifecycle, but it is not yet fully validated as a
finished second brain. The remaining work is mostly validation, performance
proof, and API cleanup rather than adding large new tables.

Current state:

| Area | Status | Evidence in the current design |
| --- | --- | --- |
| Core model boundary | Implemented | MemoryForge returns `CoreContextBundle`; LTM does not answer with a model. |
| RLM/LCM worker boundary | Implemented | RLM and LCM can use bounded sub-agent workers; benchmark answer runners are named separately. |
| Stable schema discipline | Implemented | New behavior uses `long_term_items.metadata`, rebuildable FTS/vector indexes, and internal LCM summary refs rather than new source-of-truth tables. |
| RLM -> LTM ingestion | Implemented | `rlm_load` chunks sources once, indexes chunks into LTM, and reports ingestion manifests/dedupe/vector stats. |
| Vector cache | Implemented | `vec_index.cache_key` is model-qualified and content-hash based when text is available. |
| Vector path | Implemented | `vec_index` is the single vector cache. Optional ANN extension storage was removed to keep schema, CI, and release behavior simpler. |
| LTM retrieval selection | Implemented | BM25/vector RRF, stream champions, and local model-free selection signals are in place. |
| Context budgeting | Implemented | Snippet-first injection, full-text escalation for champions, raw refs preserved separately. |
| LCM off hot path | Implemented | Soft compaction can be deferred; compaction output is input-signature cached through internal summary refs. |
| Memory controller | Partial | Typed metadata, metadata-first turn classification, corrections, contradictions, open tasks, project state, entity profile, procedural, and identity labels exist. Broader transient-noise policy and real-run validation still need hardening. |
| Active recall | Implemented | Context bundle injects proactive typed memory before normal LTM recall. |
| Runtime integration | Implemented for Codex-style MCP path | Runtime bundle validates `.memoryforge/config.json`, MCP delivery, and DB path alignment before returning context. |
| Second-brain benchmark | Partial | Deterministic lifecycle benchmark exists; current real LongMemEval core-answer subsets cover the first 10 dataset cases (`10/10` exact across two `gpt-5.2` gates) and LongMemEval/LoCoMo no-ingest context-only subset runs are recorded under `benchmarks/results`; full refresh still needs to be repeated. |
| Performance targets | Partial | Current subset result files report latency splits and meet no-ingest context-only targets; LongMemEval core-answer gate 1-5 is sub-300ms query/context avg (`208.85ms`), while gate 6-10 is slower (`411.49ms` avg), so full-scale targets still need current real-data evidence and tuning. |

What is still not done:

- Re-run full real LongMemEval/LoCoMo after the latest retrieval/context changes.
  Current subset LongMemEval core-answer and no-ingest context-only runs are
  stored under `benchmarks/results`.
- Produce a current latency table for ingest, retrieval, context assembly, and
  answer model latency.
- Tighten memory-controller policy for transient noise, task updates, and entity
  profile merging without growing schema.
- Reduce API/mixin surface where implementation details still leak across
  module boundaries.
- Keep public diagnostics on `selection` terminology. The historical `rerank`
  stream key is retained only as a compatibility alias; the underlying behavior
  remains local model-free selection.

### Definition of Done

Call the roadmap implementation complete only when all of the following are true:

1. `make check` passes on the current tree.
2. Context-only benchmark mode proves question-time work is retrieval/context
   assembly only: no ingestion, no answer model, no RLM/LCM worker call.
3. Real LongMemEval and LoCoMo runs have been repeated after the latest
   retrieval/context changes, with result files kept under `benchmarks/results`.
4. The benchmark summary separates ingestion/setup latency, MemoryForge
   query/context latency, and core answer latency.
5. The real-run diagnostics show BM25/vector participation, selected refs,
   context token counts, raw refs, and provenance for each case.
6. A repeated-ingest gate proves unchanged sources create zero new RLM chunks,
   LTM rows, and embeddings.
7. The memory-controller behavior for corrections, contradictions, open tasks,
   preferences, project state, entity profile, procedural memory, and identity
   memory is covered by tests without adding type-specific tables.
8. Runtime integration fails loudly on MCP/config/DB mismatch and never silently
   swaps in a separate answer model.
9. Public docs and diagnostics consistently describe Stage 2 as local selection,
   not an LLM reranker or sub-agent.
10. Remaining API/mixin boundaries are documented or simplified enough that RLM,
    LCM, LTM, runtime, and benchmark responsibilities are easy to audit.

### Remaining Execution Slices

These are the remaining practical slices before the roadmap can be called done:

| Slice | Purpose | Expected effort |
| --- | --- | --- |
| Diagnostics naming cleanup | Implemented: public diagnostics expose local `selection`; historical `rerank` is compatibility-only. | Done |
| Memory-controller hardening | Partially implemented: task updates, entity-profile merge policy, and metadata-first turn classification use metadata only. Broader transient-noise policy still needs hardening. | Medium |
| API boundary cleanup | Reduce implicit mixin surface and document RLM/LCM/LTM boundaries. | Medium |
| Real benchmark refresh | Re-run current full LongMemEval/LoCoMo after recent retrieval/context changes with real model settings. Subset no-ingest context-only and LongMemEval core-answer runs are refreshed. | Medium to large, mostly runtime latency |
| Performance report | Add/update latency table for ingest, retrieval, context assembly, answer model, token counts, and refs. | Small after benchmark data exists |
The shortest path to a credible “finished” claim is: finish docs/API cleanup,
then run the real benchmark refresh, then write the performance report from those
results. Without that evidence, the implementation is promising but not yet
proven.

## Speed Optimization Plan

### 1. Remove Re-ingestion From the Hot Path

The question path must never reprocess the whole dataset or project.

Target behavior:

- RLM loads and chunks sources once.
- LTM persists BM25, vector embeddings, metadata, and raw refs.
- LCM stores live session context and summary nodes incrementally.
- A question performs only retrieval, context assembly, and core-model answer.

Key tasks:

- Add benchmark modes that separate ingestion time from query time.
- Persist ingestion manifests by content hash, model, chunking config, and source.
- Skip unchanged RLM buffers and unchanged embeddings.

Current `rlm_load` and LongMemEval diagnostics expose `ingestion_manifest` with
content hash, chunk config, RLM/LTM dedupe flags, and vector model/cache stats.
The repeat-ingestion invariant is `rlm_deduped=true`, `ltm_deduped=true`, and
`vector.add_stats.encoded=0` for unchanged input.
`vec_index.cache_key` is now `embedding_backend:vector_model + content_hash`
when content text is available, with legacy `content_id` lookups kept only for
migration fallback.

Gate:

- Running the same benchmark query twice must not create new RLM buffers, chunks,
  LTM items, or embeddings unless the source changed.

### 2. Batch Embeddings and Cache by Content Hash

Current per-item embedding is too slow for large ingestion.

Target behavior:

- Batch FastEmbed calls.
- Cache embedding rows by `content_hash + vector_backend + vector_model`.
- Reuse embeddings across RLM, LTM, and conversation memory when content is
  identical.
- Store embedding dimensions and model identity in `vec_index`.

Candidate models:

- `BAAI/bge-small-en-v1.5`: FastEmbed-supported, 384 dimensions.
- `sentence-transformers/all-MiniLM-L6-v2`: FastEmbed-supported, 384 dimensions.
- `BAAI/bge-m3`: 1024 dimensions conceptually, but not supported by the tested
  FastEmbed `TextEmbedding` backend at the time of this note.

### 3. Keep Vector Search Simple

The current local vector path is the release path. It favors predictable
installation, schema simplicity, and CI stability over optional ANN extensions.

Target behavior:

- Keep `vec_index` as the single persistent vector cache.
- Keep the in-process HNSW-compatible index loaded from `vec_index`.
- Fall back to brute-force search over `vec_index` if needed.
- Do not add a second optional ANN table unless full benchmark evidence proves
  the extra release complexity is worth it.

Current split:

- Embedding backend: `fastembed`, or disabled lexical-only failover.
- Vector cache: `vec_index`.
- Runtime search: local HNSW-compatible index plus brute-force fallback.

### 4. Two-Stage Retrieval

Question-time retrieval should be deterministic and cheap by default.

Stage 1:

- BM25 overfetch.
- Vector overfetch.
- RRF fusion by rank, not raw score mixing.
- Keep each retrieval stream champion so BM25 and vector cannot erase each other.

Stage 2 local selection:

- Apply local selection signals without calling an LLM:
  - exact answer-bearing terms
  - session/date proximity
  - source type priority
  - repeated fact reinforcement
  - entity match
  - answer-session metadata when benchmark data exposes it

LLM reranking should be optional and off the hot path. Current LTM behavior does
not call an LLM here. Diagnostics expose this stage as `selection`; the
historical `rerank` stream key is a compatibility alias for deterministic local
candidate selection.

Current local selection diagnostics include exact token overlap, entity overlap,
same-session matches, normalized time-token matches, correction/staleness
metadata, source-type priority, durable kind priority, answer-evidence metadata,
and repeated fact reinforcement. Core context bundles, LCM recall injection,
CLI, and MCP can pass `session_id` into LTM recall, so session proximity is
handled in the retrieval boundary instead of through a new table. Source-type
priority is a small local tie-break for direct evidence sources, not a model
call. Reinforcement is query-relative and entity-aware so conflicting entity
memories do not reinforce each other through shared project/query anchors.
LongMemEval `has_answer` and LoCoMo `evidence` IDs are normalized by their
adapters into metadata before retrieval; LTM does not know those
dataset-specific field names.

Gate:

- Retrieval must report separate BM25, vector, fused, and final selected refs.
- At least one champion from each active retrieval stream should survive into the
  candidate set unless explicitly filtered by policy.

### 5. Context Budgeting

Injecting full chunks improved accuracy but made benchmark queries slow.

Target behavior:

- Use snippets by default.
- Escalate to full chunk only when:
  - the evidence window is ambiguous
  - the query asks for a relation spanning multiple turns
  - the hit is a top stream champion
  - the benchmark/evaluator requires exact phrasing
- Include neighboring chunk refs without always including full neighboring text.

Goal:

- Bring typical query context from roughly 9k tokens down to 2k-4k tokens.
- Preserve answer recall and raw refs.

Policy order:

1. Include concise snippets around the best evidence window.
2. Include full chunks only for stream champions or ambiguous evidence.
3. Include neighboring refs before neighboring full text.
4. Include summary nodes only when they add non-duplicated state.
5. Always preserve raw refs even when text is truncated.

Current context bundles default to `snippet`, expose `retrieval.recall_text`
diagnostics, and make `auto` snippet-first with full-text escalation only for
stream champions. Raw refs are preserved separately from injected text.

### 6. LCM Compaction Off the Critical Path

LCM compaction is allowed to use a sub-agent, but it should not block every user
question unless context is actually over budget.

Target behavior:

- Background or lazy compaction.
- Input-hash cache for compaction outputs. Current implementation stores the
  cache signature as an internal `summary_nodes.source_refs` entry and filters
  that internal ref out of public raw refs/provenance.
- Incremental summary DAG updates.
- Avoid forced compaction in normal query benchmarks except special LCM/DAG tests.

### 7. RLM Parallelism

RLM is the right place to use sub-agents heavily, but it must be bounded.

Target behavior:

- Parallel batch analysis with a concurrency limit.
- Retry only transient failures.
- Store partial records.
- Recursive reduction only when output still exceeds budget.
- Track cost, tokens, latency, and chunk coverage per run.

### 8. Benchmark Modes

Benchmarks must not blur architectural boundaries.

Required modes:

- `ingest-only`: RLM -> LTM index build.
- `context-only`: question -> LTM recall -> LCM context bundle, no answer model.
- `core-answer`: context bundle -> configured core answer runner.
- `rlm-worker`: tests RLM sub-agent behavior only.
- `lcm-worker`: tests LCM compaction sub-agent behavior only.

Current benchmark results include a `diagnostics.mode_contract` payload so mode
boundaries are auditable per case. `context-only` remains bundle-only even for
special probe records; LCM worker compaction is isolated to `lcm-worker` or
explicit core-answer special-probe runs.

Metrics:

- ingestion latency
- query retrieval latency
- context build latency
- answer latency
- exact score
- semantic score
- true miss count
- token budget per answer
- refs included per answer

Suggested performance targets:

- Ingest unchanged source: zero new chunks, embeddings, and LTM rows.
- LTM retrieval after warmup: sub-100ms for ordinary project-sized indexes.
- Context assembly excluding answer model: sub-300ms target.
- Typical injected context: 2k-4k tokens.
- Full answer benchmark should report model-answer latency separately from
  MemoryForge retrieval/context latency.

## How to Move Beyond Agentic RAG

MemoryForge becomes only an improved RAG system if its main loop is:

```text
retrieve chunks -> ask a model -> return answer
```

To become a second brain, MemoryForge needs a continuous memory lifecycle:

```text
observe -> store -> consolidate -> retrieve -> inject -> act -> learn correction
```

The difference is not only better retrieval. The difference is persistent,
structured, self-maintaining memory around the core agent.

### 1. Add a Memory Controller

Add a policy layer that decides what to do with new information:

- store as raw episode
- extract durable fact
- update an entity profile
- create or update a task
- link to a project
- mark as contradiction
- mark as preference
- ignore as transient noise

This controller should not replace RLM/LTM/LCM. It coordinates them.

Start with a logic-only controller. Store controller decisions in existing
metadata fields until query performance proves a new index is necessary.

### 2. Separate Memory Types

A second brain needs more than chunks.

Suggested memory types:

- Episodic memory: what happened, when, in which session.
- Semantic memory: stable facts, preferences, decisions.
- Procedural memory: how the user likes work to be done.
- Project memory: goals, constraints, active files, milestones.
- Task memory: open loops, deadlines, blockers, promises.
- Identity memory: user preferences, style, recurring patterns.
- Source memory: raw immutable refs, chunks, files, transcripts.

RLM preserves raw and large source memory.
LTM retrieves durable memory.
LCM manages active working memory.
The memory controller decides what belongs where.

Initial implementation can encode these types in `long_term_items.metadata.kind`.
Do not add one table per memory type during the first implementation pass.

Current metadata kind constants include `project_state`, `entity_profile`,
`procedural`, and `identity` alongside task, correction, preference, decision,
episodic, source, and fact. These remain logic-only labels in existing metadata.

### 3. Add Entity and Time Awareness

Agentic RAG often fails because all chunks are treated as equal.

A second brain should track:

- people
- projects
- files
- dates
- locations
- decisions
- commitments
- unresolved questions
- corrections
- stale facts

Retrieval should be able to answer:

- What did we decide last week?
- What changed since the last run?
- What does the user usually prefer?
- Which facts conflict?
- Which task is still open?

### 4. Learn From Corrections

When the core model answers incorrectly and the user corrects it, MemoryForge
should not just store another chat line.

It should:

- attach the correction to the wrong answer
- mark the corrected fact as high confidence
- lower confidence of the contradicted memory
- preserve source refs
- make future retrieval prefer the correction

This is a major step beyond RAG.

### 5. Active Recall, Not Passive Search

RAG waits for a query. A second brain should proactively surface useful memory.

Examples:

- The user is editing a benchmark: recall previous benchmark failures.
- The user asks for speed: recall existing latency bottlenecks.
- The user reopens a project: recall open tasks and last known state.
- The user repeats a correction: promote it to a durable preference.

This requires a context policy, not just a search API.

### 6. Trust and Provenance

Every memory injected into the core model should carry:

- raw source ref
- source type
- timestamp
- confidence
- freshness
- whether it came from user, assistant, file, benchmark, or synthesis
- whether it is direct evidence or derived summary

The core model should be able to distinguish raw truth from generated summary.

### 7. Core Runtime Integration

When MemoryForge is added to a project, it should detect or integrate with the
active agent runtime.

For Codex CLI-like usage:

- expose MCP tools
- provide context bundle commands
- read project root and config
- avoid silently choosing a different answer model
- let the active core model consume MemoryForge context

Sub-agents remain internal workers for RLM and LCM.

The runtime adapter should fail loudly if it cannot identify the active runtime
or cannot deliver context to the core model. It should not silently spawn a
different answer model and call that success.

Current runtime validation checks Codex MCP delivery, `.memoryforge/config.json`,
and database path alignment before returning a runtime context bundle. The MCP
surface includes `build_runtime_context_bundle` for the same fail-loud path.

### 8. Evaluation for Second Brain Behavior

LongMemEval and LoCoMo are useful, but not enough.

Add evaluations for:

- cross-session continuity
- remembering user preferences
- preserving unresolved tasks
- correcting stale facts
- detecting contradictions
- project restart quality
- answer provenance
- context budget efficiency
- whether the core model, not a worker sub-agent, answered using injected memory

Current deterministic `second-brain` benchmark reports `dimension_coverage` for
these categories in `context-only` mode and verifies ingestion-only metadata for
typed memory, corrections, and contradictions without using an answer model.

## Priority Order

1. Fix architecture boundaries:
   - LTM has no model call.
   - benchmark answer runner is not called a sub-agent.
   - core context bundle is the primary output.
   - schema remains stable; new behavior starts as logic plus metadata.

2. Split benchmark modes:
   - ingest-only
   - context-only
   - core-answer
   - RLM worker
   - LCM worker

3. Optimize retrieval and context:
   - batch embeddings
   - embedding cache
   - snippet-first context
   - full chunk escalation
   - session/date/entity selection

4. Add memory controller:
   - memory type classification
   - entity extraction
   - task/preference/decision tracking
   - correction learning

5. Add runtime integration:
   - Codex CLI adapter
   - MCP tools
   - core context injection
   - no silent model replacement

6. Add second-brain benchmark suite:
   - continuity
   - preferences
   - tasks
   - corrections
   - contradictions
   - provenance

## Success Criteria

MemoryForge is beyond Agentic RAG when:

- The active core model answers with MemoryForge context.
- LTM does not generate answers.
- RLM and LCM sub-agents are bounded internal workers.
- Memory is durable, typed, corrected, and provenance-aware.
- The system recalls useful context proactively, not only by query string.
- It improves across sessions and user corrections.
- It can restart a project and reconstruct the working state with refs.
- Query-time latency is dominated by retrieval/context assembly, not ingestion.
