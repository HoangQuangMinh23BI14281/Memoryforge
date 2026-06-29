# Recursive Language Model Integration

RLM is MemoryForge's oversized-input path. It lets a root agent work with a
large prompt or file through chunk references instead of copying the full source
into the active context.

## Mapping

| RLM idea | MemoryForge implementation |
| --- | --- |
| External environment | `.memoryforge/memory.db` |
| Load large context | `memoryforge rlm-load` |
| Chunk references | `rlm_chunk:<id>` |
| Chunk retrieval | `memoryforge rlm-chunk-get <id>` |
| Hybrid retrieval | BM25 plus vector search with RRF fusion |
| Worker findings | Host-subagent analyses recorded as LCM messages, SummaryDAG leaves, and derived LTM items |
| Aggregation | SummaryDAG parent nodes plus `rlm_summary` LTM rows |

## Lossless Contract

- Full sources are stored in `content_store`.
- Chunks store byte, char, and line ranges.
- `rlm-chunk-get` can recover exact chunk content.
- LTM indexes RLM chunks with BM25 and vector search.
- `memoryforge index --analyze` and MCP `index_analyze` return host-subagent batch prompts and record/aggregate commands.
- Recorded host-subagent analyses become `rlm_analysis` and final aggregates become `rlm_summary`.
- Sub-agent outputs keep source chunk refs, so shallow recall can rehydrate exact chunks later.

## Runtime Shape

The low-level tools support a root LLM that dispatches workers manually:

- `rlm-load`
- `rlm-search`
- `rlm-chunk-get`
- `dispatch`
- `rlm-record`
- `aggregate`

`rlm-load` is ingestion-only: it loads/chunks/indexes lossless source chunks and
does not spawn a worker. `memoryforge index --analyze` and MCP `index_analyze`
follow the same principle: they prepare host-subagent work but do not call `codex exec` or any
external model process. The active host agent runs subagents, then records each
batch with `rlm-record` and finalizes with `aggregate`.

## Relation To LCM

RLM processes large sources. LCM decides what enters the next active model
context. LTM recall can surface RLM-derived summaries first, while full file
bodies and exact chunks stay recoverable through `rlm_chunk:<id>` refs unless a
caller explicitly rehydrates them.
