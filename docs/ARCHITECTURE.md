# MemoryForge Architecture

This document describes the current implementation. It is not the full
second-brain roadmap.

MemoryForge is a local memory layer for the model the user is already working
with. The core model answers. MemoryForge stores evidence, retrieves memory, and
builds a context bundle for that core model.

## Runtime Boundary

```text
Host/core model
  -> MemoryForge API
  -> CoreContextBundle
  -> Host/core model

MemoryForge API
  -> SQLite memory.db
  -> optional RLM/LCM workers
```

Rules:

- The core model is outside MemoryForge.
- `build_core_context_bundle(...)` retrieves memory and assembles context only.
- LTM does not call a model.
- RLM may call a worker in `rlm_run(...)`.
- LCM may call a worker during compaction.
- Benchmark answer runners are benchmark/core answer runners, not LTM.

## Main Code Boundaries

| Area | Current Code | Owns | Model Call? |
| --- | --- | --- | --- |
| API facade | `memoryforge/api/app.py` | `MemoryForge` orchestration | No final answer call |
| LCM | `memoryforge/lcm/...` | live session context and summaries | Optional compaction worker |
| RLM | `memoryforge/rlm/...` | large source loading, chunking, dispatch | Optional RLM worker |
| LTM | `memoryforge/memory/longterm/...` | durable memory indexing and recall | No |
| Vector | `memoryforge/search/vector.py` | embedding cache and local vector search | Embedding model only |
| FTS | `memoryforge/search/fts.py` | lexical search index or fallback table | No |
| Long-term memory metadata | `memoryforge/memory/longterm/models.py`, `indexing.py`, `retrieval.py` | durable memory metadata fields and local selection policy | No |

Current debt: `RLMEngine` is still composed from mixins. That is an internal
implementation detail, not a clean public boundary.

## Storage Map

```text
content_store
  -> LCM tables
  -> RLM tables
  -> long_term_items
       -> search_fts
  -> vec_index
```

Actual schema sources:

- `memoryforge/db/schema.py`: shared schema for `content_store`, LCM tables,
  `vec_index`, and `long_term_items`.
- `memoryforge/rlm/schema.py`: `rlm_buffers` and `rlm_chunks`.
- `memoryforge/search/fts.py`: shared lexical search table.
- `memoryforge/search/vector.py`: vector cache and local vector search.

## Tables

Source-of-truth tables:

| Table | Meaning |
| --- | --- |
| `content_store` | Exact stored text, content hash, reference count, timestamps. |
| `sessions` | LCM session root. This table does exist. |
| `messages` | Immutable raw session messages and summary messages. |
| `message_parts` | Message parts with `content_id` back to `content_store`. |
| `context_items` | Ordered active context view for a session. |
| `summary_nodes` | LCM compression nodes used by compaction. |
| `rlm_buffers` | Whole large sources loaded by RLM. |
| `rlm_chunks` | Pass-by-reference chunks for RLM/LTM retrieval. |
| `long_term_items` | Durable memory records pointing to source content. |

Indexes and caches:

| Table | Meaning |
| --- | --- |
| `search_fts` | Shared lexical index for conversation, RLM chunks, and LTM. Uses FTS5 when available, otherwise a plain fallback table. |
| `vec_index` | Model-qualified embedding cache. It is not the durable memory table. |

## Session Invariant

`sessions` is the root. `context_items` is only the active view.

Current invariant:

```text
sessions.id
  -> messages.session_id
  -> message_parts.message_id

context_items.session_id
  -> item_type = message, item_id = messages.id
  -> item_type = summary, item_id = summary_nodes.id

summary_nodes.session_id
  -> parent_node_ids / parent_node_id for LCM compaction lineage
```

Compaction changes the active view, not the raw record:

1. Raw messages stay in `messages`.
2. Exact text stays in `content_store`.
3. LCM creates a row in `summary_nodes`.
4. LCM swaps old `context_items` rows for one summary row.

This is why `context_items` cannot replace `sessions`. It is a per-session
window, not the session identity.

## No Knowledge Graph

The current implementation has no graph store, no entity-relation table, and no
temporal knowledge graph.

`summary_nodes` forms a compaction DAG for LCM summaries. That DAG is internal
compression lineage, not a knowledge graph.

LTM metadata is stored in `long_term_items.metadata` and uses the lightweight
field contract in `memoryforge/memory/longterm/models.py`. It is not a fixed
memory-kind ontology. Caller-provided labels such as `kind` may be preserved
for provenance, benchmark evidence, or lifecycle annotations, but recall does
not require a knowledge-graph schema.

## Write Paths

Conversation ingestion:

1. `MemoryForge.store_conversation(...)`
2. `ConversationStore.store_session(...)`
3. `ImmutableMessageStore.append_text_message(...)`
4. `content_store`, `sessions`, `messages`, `message_parts`, `context_items`
5. BM25 row in `search_fts` with `scope = conversation`
6. embedding row in `vec_index`
7. `LongTermMemoryIndex.index_messages(...)` creates `long_term_items`

Large source ingestion:

1. `MemoryForge.ingest_file(...)` or `MemoryForge.rlm_load(...)`
2. `RLMEngine.load(...)`
3. full source row in `content_store`
4. source owner row in `rlm_buffers`
5. chunk rows in `rlm_chunks`
6. chunk text rows in `content_store`
7. `LongTermMemoryIndex.index_rlm_buffer(...)` creates `long_term_items`
8. lexical and vector indexes are updated

Question-time context:

1. `MemoryForge.build_core_context_bundle(...)`
2. LCM builds bounded session context from `context_items`
3. LTM runs active recall and query recall
4. LTM combines BM25 and vector hits with local selection
5. MemoryForge injects active recall, LTM snippets, refs, provenance, and
   diagnostics into `CoreContextBundle`
6. The host/core model reads that bundle and answers

At question time, RLM should not reprocess the whole original file. RLM already
loaded and indexed source chunks earlier.

## Retrieval

Current LTM retrieval uses:

- lexical search through `search_fts`
- semantic search through `vec_index`
- reciprocal-rank fusion
- stream champions so BM25 and vector hits both survive
- deterministic local selection signals

Public diagnostics expose Stage 2 as `selection`, a deterministic local pass over
retrieved candidates. The older `rerank` stream key remains as a compatibility
alias only; it is not an LLM reranker.

## Vector Storage

`vec_index.cache_key` is model-qualified and content-hash based when content is
available:

```text
embedding_backend:vector_model + content_hash
```

Different embedding models can coexist. Recommended serious retrieval settings:

```bash
export MEMORYFORGE_VECTOR_BACKEND=fastembed
export MEMORYFORGE_VECTOR_MODEL=BAAI/bge-small-en-v1.5
export MEMORYFORGE_REQUIRE_VECTOR_MODEL=1
```

MemoryForge intentionally keeps a single canonical vector cache, `vec_index`.
There is no optional ANN extension table in the release schema, which keeps the
schema and release surface smaller.
