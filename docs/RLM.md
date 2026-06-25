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
| Worker findings | LCM messages and SummaryDAG leaves |
| Aggregation | SummaryDAG parent nodes |

## Lossless Contract

- Full sources are stored in `content_store`.
- Chunks store byte, char, and line ranges.
- `rlm-chunk-get` can recover exact chunk content.
- LTM indexes RLM chunks with BM25 and vector search.
- Sub-agent outputs keep source chunk refs.

## Runtime Shape

The low-level tools support a root LLM that dispatches workers manually:

- `rlm-load`
- `rlm-search`
- `rlm-chunk-get`
- `dispatch`
- `rlm-record`
- `aggregate`

`rlm-run` can also run configured sub-agents directly and record their outputs
into the local LCM store and summary DAG.

## Relation To LCM

RLM processes large sources. LCM decides what enters the next active model
context. RLM findings can become part of LCM, but full file bodies stay in
RLM/LTM unless a caller explicitly rehydrates them.
