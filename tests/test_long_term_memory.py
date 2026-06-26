import sys
import types
from datetime import datetime, timezone

from memoryforge import MemoryForge
from memoryforge.lcm import ContextBudget
from memoryforge.memory.longterm.models import LongTermRecallResult
from memoryforge.memory.longterm.retrieval import LongTermRetrievalMixin


def _install_fake_fastembed(monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for text in texts:
                vector = [0.0] * 8
                lowered = text.lower()
                if "semantic" in lowered:
                    vector[0] = 1.0
                if "embedding" in lowered:
                    vector[1] = 1.0
                if "memory" in lowered:
                    vector[2] = 1.0
                if not any(vector):
                    vector[3] = 1.0
                yield vector

    fake_fastembed = types.ModuleType("fastembed")
    fake_fastembed.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)


def test_ltm_indexes_raw_message_refs_across_bm25_and_vector(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turn_ids = mf.store_conversation(
            "agent",
            [
                {
                    "role": "user",
                    "content": "Alice uses SQLite because local-first storage is simple.",
                }
            ],
            session_id="session-1",
        )

        hits = mf.recall_long_term("agent", "Alice SQLite storage", top_k=5, include_content=True)

        assert hits
        assert hits[0]["source_type"] == "message"
        assert hits[0]["source_id"] == turn_ids[0]
        assert hits[0]["raw_ref"] == f"message:{turn_ids[0]}"
        assert "content_id" in hits[0]
        assert "Alice uses SQLite" in hits[0]["content"]
        assert {"bm25", "vector"} & set(hits[0]["streams"])
    finally:
        mf.close()


def test_lcm_context_can_include_ltm_recall_without_replacing_raw_turns(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Alice prefers SQLite for offline memory."}],
            session_id="session-1",
        )

        result = mf.lcm_build_context_with_recall(
            "session-1",
            "agent",
            "What did Alice choose?",
            budget=ContextBudget(model_context_limit=10_000),
        )
        context = result["context"]

        assert result["long_term_recall"]
        assert any(message.source == "long_term" for message in context.messages)
        assert any(message.source == "message" for message in context.messages)
        assert "message:" in next(
            message.content for message in context.messages if message.source == "long_term"
        )
    finally:
        mf.close()


def test_lcm_context_uses_full_ltm_recall_content_not_preview(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        tail_fact = "TAIL_FACT: Alice graduated with Business Administration."
        mf.store_conversation(
            "agent",
            [
                {
                    "role": "user",
                    "content": "filler text " * 80 + tail_fact,
                }
            ],
            session_id="session-1",
        )

        result = mf.lcm_build_context_with_recall(
            "session-1",
            "agent",
            "What degree did Alice graduate with?",
            budget=ContextBudget(model_context_limit=10_000),
        )
        recall_message = next(
            message for message in result["context"].messages if message.source == "long_term"
        )

        assert tail_fact in recall_message.content
        assert result["long_term_recall"][0]["content"].endswith(tail_fact)
    finally:
        mf.close()


def test_core_context_bundle_recall_policy_controls_injected_text(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        tail_fact = "TAIL_FACT: Atlas retention key is amber-17."
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "filler text " * 100 + tail_fact}],
            session_id="session-1",
        )

        snippet_bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Atlas retention key",
            include_content=True,
            recall_content_policy="snippet",
        )
        champion_bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Atlas retention key",
            include_content=True,
            recall_content_policy="champion",
        )
        default_bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Atlas retention key",
            include_content=True,
        )
        snippet_recall = next(
            message for message in snippet_bundle.messages if message["source"] == "long_term"
        )
        champion_recall = next(
            message for message in champion_bundle.messages if message["source"] == "long_term"
        )
        default_recall = next(
            message for message in default_bundle.messages if message["source"] == "long_term"
        )

        assert snippet_bundle.long_term_recall[0]["content"].endswith(tail_fact)
        assert tail_fact not in snippet_recall["content"]
        assert tail_fact in champion_recall["content"]
        assert len(snippet_recall["content"]) < len(champion_recall["content"])
        assert default_bundle.diagnostics["context"]["recall_content_policy"] == "snippet"
        assert tail_fact not in default_recall["content"]
        assert len(default_recall["content"]) < len(champion_recall["content"])
        assert (
            snippet_bundle.diagnostics["context"]["recall_content_policy"] == "snippet"
        )
        recall_text = snippet_bundle.diagnostics["retrieval"]["recall_text"]
        assert recall_text[0]["raw_ref"] == snippet_bundle.long_term_recall[0]["raw_ref"]
        assert recall_text[0]["requested_policy"] == "snippet"
        assert recall_text[0]["truncated"] is True
        assert recall_text[0]["injected_chars"] < recall_text[0]["source_chars"]
        assert snippet_bundle.long_term_recall[0]["raw_ref"] in snippet_bundle.raw_refs
    finally:
        mf.close()


def test_core_context_bundle_enforces_ltm_injection_token_budget(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        hidden_tail = "FULL_ONLY_TAIL: this should not fit the rendered LTM block."
        text = "Atlas budget needle. " + ("filler text " * 220) + hidden_tail
        mf.long_term.index_raw_item("agent", "note", "oversized-note", text)

        bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Atlas budget needle",
            budget=ContextBudget(
                model_context_limit=1_200,
                reserved_output_tokens=100,
                compaction_buffer=100,
            ),
            top_k=1,
            include_content=True,
            recall_content_policy="full",
            long_term_token_budget=170,
        )
        injection = bundle.diagnostics["context"]["long_term_recall_injection"]
        ltm_message = next(
            message for message in bundle.messages if message["source"] == "long_term"
        )

        assert injection["requested_policy"] == "full"
        assert injection["effective_policy"] in {"snippet", "preview"}
        assert injection["token_estimate"] <= injection["token_budget"]
        assert bundle.token_estimate <= bundle.budget["hard_limit"]
        assert hidden_tail not in ltm_message["content"]
    finally:
        mf.close()


def test_ltm_recall_auto_policy_uses_full_only_for_stream_champions():
    content = "intro " * 120 + "Atlas retention amber-17 " + "tail " * 120
    champion = LongTermRecallResult(
        item_id="champion",
        source_type="note",
        source_id="champion",
        content_id="content-champion",
        raw_ref="note:champion",
        preview="Atlas retention amber-17",
        score=0.9,
        streams={"bm25": {"rank": 1, "score": 10.0}},
        metadata={},
        content=content,
    )
    non_champion = LongTermRecallResult(
        item_id="non-champion",
        source_type="note",
        source_id="non-champion",
        content_id="content-non-champion",
        raw_ref="note:non-champion",
        preview="Atlas retention amber-17",
        score=0.5,
        streams={"bm25": {"rank": 2, "score": 5.0}},
        metadata={},
        content=content,
    )

    diagnostics = LongTermRetrievalMixin().recall_injection_diagnostics(
        [champion, non_champion],
        content_policy="auto",
        query="Atlas retention amber-17",
    )

    assert diagnostics[0]["effective_policy"] == "full"
    assert diagnostics[0]["truncated"] is False
    assert diagnostics[1]["effective_policy"] == "preview"
    assert diagnostics[1]["truncated"] is True


def test_ltm_vector_recall_maps_content_id_back_to_item_id(tmp_path, monkeypatch):
    _install_fake_fastembed(monkeypatch)
    monkeypatch.setenv("MEMORYFORGE_VECTOR_BACKEND", "fastembed")
    monkeypatch.setenv("MEMORYFORGE_VECTOR_MODEL", "fake-semantic-model")
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        item_id = mf.long_term.index_raw_item(
            "agent",
            "note",
            "semantic-only-note",
            "vector embeddings retrieve durable semantic memory",
        )
        mf.long_term.conn.execute("DELETE FROM search_fts WHERE scope = 'long_term'")
        mf.long_term.conn.commit()

        hits = mf.recall_long_term("agent", "embeddings semantic memory", top_k=3)

        assert hits
        assert hits[0]["item_id"] == item_id
        assert hits[0]["source_type"] == "note"
        assert "vector" in hits[0]["streams"]
    finally:
        mf.close()


def test_ltm_bm25_ranks_more_negative_fts_rank_higher(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        relevant_id = mf.long_term.index_raw_item(
            "agent",
            "note",
            "degree-note",
            "degree degree degree Business Administration.",
        )
        mf.long_term.index_raw_item(
            "agent",
            "note",
            "noise-note",
            "degree routine paperwork.",
        )

        hits = mf.long_term._bm25_search("agent", "degree", limit=3)

        assert hits
        assert hits[0][0] == relevant_id
        assert hits[0][1] > 0
    finally:
        mf.close()


def test_recorded_correction_supersedes_wrong_memory_in_recall(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        wrong_item_id = mf.long_term.index_raw_item(
            "agent",
            "note",
            "atlas-old-key",
            "Project Atlas retention key is blue-12.",
            metadata={"kind": "fact", "confidence": "normal"},
        )

        correction = mf.record_correction(
            "agent",
            "Project Atlas retention key is amber-17.",
            wrong_item_id=wrong_item_id,
            session_id="session-1",
        )
        hits = mf.recall_long_term(
            "agent",
            "Project Atlas retention key",
            top_k=2,
            include_content=True,
        )
        wrong_source = mf.long_term_source("agent", wrong_item_id)

        assert hits[0]["item_id"] == correction["item_id"]
        assert hits[0]["source_type"] == "correction"
        assert hits[0]["metadata"]["kind"] == "correction"
        assert hits[0]["metadata"]["confidence"] == "high"
        assert "amber-17" in hits[0]["content"]
        assert wrong_source is not None
        assert wrong_source["metadata"]["confidence"] == "low"
        assert correction["item_id"] in wrong_source["metadata"]["superseded_by"]
    finally:
        mf.close()


def test_ltm_ensemble_keeps_each_retrieval_stream_champion():
    selected = LongTermRetrievalMixin._select_ensemble_ids(
        [("shared-noise", 0.9), ("lexical-answer", 0.3), ("semantic-hit", 0.2)],
        {
            "bm25": [("lexical-answer", 10.0), ("shared-noise", 1.0)],
            "vector": [("semantic-hit", 0.8), ("shared-noise", 0.7)],
        },
        limit=3,
    )

    assert selected == ["lexical-answer", "semantic-hit", "shared-noise"]


def test_ltm_rerank_does_not_use_exact_entity_or_date_signals():
    noise = LongTermRecallResult(
        item_id="noise",
        source_type="note",
        source_id="noise",
        content_id="cnt_noise",
        raw_ref="note:noise",
        preview="General benchmark notes without the owner.",
        score=0.9,
        streams={"bm25": {"rank": 1, "score": 10.0}},
        metadata={"kind": "source", "confidence": "normal"},
    )
    relevant = LongTermRecallResult(
        item_id="atlas-owner",
        source_type="note",
        source_id="atlas-owner",
        content_id="cnt_atlas",
        raw_ref="note:atlas-owner",
        preview="Project Atlas migration owner is Mina on 2026-06-01.",
        score=0.1,
        streams={"vector": {"rank": 1, "score": 0.8}},
        metadata={
            "kind": "fact",
            "confidence": "normal",
            "entities": ["Project Atlas", "Mina"],
            "event_date": "2026-06-01",
            "time_tokens": ["2026", "06", "01"],
        },
    )

    reranked = LongTermRetrievalMixin._rerank_results(
        [noise, relevant],
        query="Project Atlas owner Mina 2026-06-01",
    )

    assert reranked[0].item_id == "noise"
    assert reranked[1].item_id == "atlas-owner"
    assert reranked[1].streams["fusion"]["score"] == 0.1
    assert "exact_token_overlap" not in reranked[1].streams["rerank"]
    assert "entity_overlap" not in reranked[1].streams["rerank"]
    assert "date_or_freshness_bonus" not in reranked[1].streams["rerank"]


def test_ltm_rerank_does_not_reinforce_repeated_token_facts():
    contested = LongTermRecallResult(
        item_id="rina",
        source_type="note",
        source_id="rina",
        content_id="cnt_rina",
        raw_ref="note:rina",
        preview="Project Atlas owner is Rina.",
        score=0.64,
        streams={"bm25": {"rank": 1, "score": 10.0}},
        metadata={
            "kind": "fact",
            "confidence": "normal",
            "entities": ["Project Atlas", "Rina"],
        },
    )
    support_a = LongTermRecallResult(
        item_id="mina-a",
        source_type="note",
        source_id="mina-a",
        content_id="cnt_mina_a",
        raw_ref="note:mina-a",
        preview="Project Atlas owner is Mina.",
        score=0.4,
        streams={"vector": {"rank": 1, "score": 0.8}},
        metadata={
            "kind": "fact",
            "confidence": "normal",
            "entities": ["Project Atlas", "Mina"],
        },
    )
    support_b = LongTermRecallResult(
        item_id="mina-b",
        source_type="note",
        source_id="mina-b",
        content_id="cnt_mina_b",
        raw_ref="note:mina-b",
        preview="Mina owns Project Atlas.",
        score=0.39,
        streams={"bm25": {"rank": 2, "score": 8.0}},
        metadata={
            "kind": "fact",
            "confidence": "normal",
            "entities": ["Project Atlas", "Mina"],
        },
    )

    reranked = LongTermRetrievalMixin._rerank_results(
        [contested, support_a, support_b],
        query="Who owns Project Atlas?",
    )
    by_id = {result.item_id: result for result in reranked}

    assert reranked[0].item_id == "rina"
    assert "reinforcement_support" not in by_id["mina-a"].streams["rerank"]
    assert "reinforcement_support" not in by_id["mina-b"].streams["rerank"]
    assert "reinforcement_support" not in by_id["rina"].streams["rerank"]


def test_ltm_stage2_rerank_requires_normalized_time_tokens_for_date_bonus():
    result = LongTermRecallResult(
        item_id="atlas-owner",
        source_type="note",
        source_id="atlas-owner",
        content_id="cnt_atlas",
        raw_ref="note:atlas-owner",
        preview="Project Atlas owner is Mina.",
        score=0.1,
        streams={"bm25": {"rank": 1, "score": 10.0}},
        metadata={
            "kind": "fact",
            "confidence": "normal",
            "entities": ["Project Atlas", "Mina"],
            "event_date": "2026-06-01",
        },
    )

    reranked = LongTermRetrievalMixin._rerank_results(
        [result],
        query="Project Atlas owner Mina 2026-06-01",
    )

    assert "date_or_freshness_bonus" not in reranked[0].streams["rerank"]


def test_ltm_stage2_rerank_uses_same_session_signal():
    older = LongTermRecallResult(
        item_id="older",
        source_type="message",
        source_id="older",
        content_id="cnt_older",
        raw_ref="message:older",
        preview="Project Atlas endpoint is alpha.",
        score=0.5,
        streams={"bm25": {"rank": 1, "score": 10.0}},
        metadata={"kind": "fact", "confidence": "normal", "session_id": "session-old"},
    )
    current = LongTermRecallResult(
        item_id="current",
        source_type="message",
        source_id="current",
        content_id="cnt_current",
        raw_ref="message:current",
        preview="Project Atlas endpoint is beta.",
        score=0.4,
        streams={"vector": {"rank": 1, "score": 0.8}},
        metadata={"kind": "fact", "confidence": "normal", "session_id": "session-current"},
    )

    reranked = LongTermRetrievalMixin._rerank_results(
        [older, current],
        query="Project Atlas endpoint",
        session_id="session-current",
    )

    assert reranked[0].item_id == "current"
    assert reranked[0].streams["rerank"]["same_session_bonus"] == 1


def test_ltm_rerank_does_not_use_source_type_priority_signal():
    generic_note = LongTermRecallResult(
        item_id="note",
        source_type="note",
        source_id="note",
        content_id="cnt_note",
        raw_ref="note:note",
        preview="Project Atlas endpoint is beta.",
        score=0.5,
        streams={"bm25": {"rank": 1, "score": 10.0}},
        metadata={"kind": "fact", "confidence": "normal"},
    )
    direct_message = LongTermRecallResult(
        item_id="message",
        source_type="message",
        source_id="message",
        content_id="cnt_message",
        raw_ref="message:message",
        preview="Project Atlas endpoint is beta.",
        score=0.45,
        streams={"vector": {"rank": 1, "score": 0.8}},
        metadata={"kind": "fact", "confidence": "normal"},
    )

    reranked = LongTermRetrievalMixin._rerank_results(
        [generic_note, direct_message],
        query="Project Atlas endpoint beta",
    )

    assert reranked[0].item_id == "note"
    assert "source_type_priority" not in reranked[0].streams["rerank"]


def test_ltm_recall_preserves_message_event_date_without_date_rerank(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        older_date = datetime(2026, 6, 20, tzinfo=timezone.utc).timestamp()
        newer_date = datetime(2026, 6, 21, tzinfo=timezone.utc).timestamp()
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Project Atlas owner is Mina."}],
            session_id="older",
            event_date=older_date,
        )
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Project Atlas owner is Rina."}],
            session_id="newer",
            event_date=newer_date,
        )

        hits = mf.recall_long_term(
            "agent",
            "Project Atlas owner 2026 06 21",
            top_k=2,
        )

        by_session = {hit["metadata"]["session_id"]: hit for hit in hits}
        assert by_session["newer"]["metadata"]["event_date"] == newer_date
        assert by_session["newer"]["metadata"]["timestamp"].startswith("2026-06-21T")
        assert "date_or_freshness_bonus" not in by_session["newer"]["streams"]["rerank"]
    finally:
        mf.close()


def test_core_context_bundle_passes_session_to_ltm_recall(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Project Atlas endpoint is alpha."}],
            session_id="session-old",
        )
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Project Atlas endpoint is beta."}],
            session_id="session-current",
        )

        bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-current",
            query="Project Atlas endpoint",
            top_k=2,
        )

        assert bundle.long_term_recall[0]["metadata"]["session_id"] == "session-current"
        assert bundle.long_term_recall[0]["streams"]["rerank"]["same_session_bonus"] == 1
    finally:
        mf.close()


def test_ltm_stage2_rerank_penalizes_superseded_low_confidence_memory():
    stale = LongTermRecallResult(
        item_id="old",
        source_type="note",
        source_id="old",
        content_id="cnt_old",
        raw_ref="note:old",
        preview="Project Atlas retention key is blue-12.",
        score=0.9,
        streams={"bm25": {"rank": 1, "score": 10.0}},
        metadata={"kind": "fact", "confidence": "low", "superseded_by": ["new"]},
    )
    current = LongTermRecallResult(
        item_id="new",
        source_type="correction",
        source_id="new",
        content_id="cnt_new",
        raw_ref="correction:new",
        preview="Project Atlas retention key is amber-17.",
        score=0.2,
        streams={"vector": {"rank": 1, "score": 0.8}},
        metadata={"kind": "correction", "confidence": "high", "supersedes": ["old"]},
    )

    reranked = LongTermRetrievalMixin._rerank_results(
        [stale, current],
        query="Project Atlas retention key",
    )

    assert reranked[0].item_id == "new"
    assert reranked[1].streams["rerank"]["low_confidence_penalty"] == 1
    assert reranked[1].streams["rerank"]["stale_or_superseded_penalty"] == 1


def test_recall_results_expose_fusion_and_selection_diagnostics(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.long_term.index_raw_item(
            "agent",
            "note",
            "atlas-owner",
            "Project Atlas migration owner is Mina on 2026-06-01.",
            metadata={"entities": ["Project Atlas", "Mina"], "event_date": "2026-06-01"},
        )

        hits = mf.recall_long_term("agent", "Project Atlas Mina 2026-06-01", top_k=1)

        assert hits
        assert hits[0]["streams"]["fusion"]["score"] >= 0
        assert hits[0]["streams"]["selection"]["final_score"] == round(hits[0]["score"], 6)
        assert hits[0]["streams"]["selection"]["bonus"] == 0
        assert hits[0]["streams"]["rerank"] == hits[0]["streams"]["selection"]

        bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Project Atlas Mina 2026-06-01",
        ).to_dict()
        retrieval = bundle["diagnostics"]["retrieval"]
        assert retrieval["selection"] == "local_model_free"
        assert retrieval["llm_selection_used"] is False
        assert retrieval["llm_rerank_used"] is False
    finally:
        mf.close()


def test_rlm_chunks_are_indexed_as_ltm_raw_refs(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        loaded = mf.rlm_load(
            "agent",
            "Alice owns authentication.\n\nBob owns billing.\n\nCarol owns search.",
            name="ownership-notes",
            chunk_size=1000,
            runner="mock",
        )

        hits = mf.recall_long_term("agent", "Alice authentication", top_k=5)

        assert loaded["long_term_item_ids"]
        assert hits
        assert any(hit["source_type"] == "rlm_chunk" for hit in hits)
    finally:
        mf.close()


def test_rlm_does_not_duplicate_ltm_when_file_already_indexed(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORYFORGE_SUBAGENT_RUNNER", "mock")
    db_path = str(tmp_path / "memory.db")
    path = tmp_path / "notes.md"
    path.write_text("Alice owns authentication. Bob owns billing. " * 80, encoding="utf-8")
    mf = MemoryForge(db_path)
    try:
        file_ingest = mf.ingest_file("agent", path, chunk_size=1_000)
        rlm_load = mf.rlm_load("agent", path, chunk_size=1_000, runner="mock")

        counts = {
            row[0]: row[1]
            for row in mf.long_term.conn.execute(
                """
                SELECT source_type, COUNT(*)
                FROM long_term_items
                WHERE agent_id = ?
                GROUP BY source_type
                """,
                ("agent",),
            ).fetchall()
        }

        assert file_ingest["source_type"] == "rlm_chunk"
        assert rlm_load["long_term_item_ids"] == file_ingest["long_term_item_ids"]
        assert counts.get("rlm_chunk", 0) > 0
        assert counts.get("file_chunk", 0) == 0
    finally:
        mf.close()


def test_file_ingest_does_not_duplicate_ltm_when_rlm_already_indexed(tmp_path):
    db_path = str(tmp_path / "memory.db")
    path = tmp_path / "notes.md"
    path.write_text("Carol owns observability. Dana owns reliability. " * 80, encoding="utf-8")
    mf = MemoryForge(db_path)
    try:
        rlm_load = mf.rlm_load("agent", path, chunk_size=1_000, runner="mock")
        file_ingest = mf.ingest_file("agent", path, chunk_size=1_000)

        counts = {
            row[0]: row[1]
            for row in mf.long_term.conn.execute(
                """
                SELECT source_type, COUNT(*)
                FROM long_term_items
                WHERE agent_id = ?
                GROUP BY source_type
                """,
                ("agent",),
            ).fetchall()
        }

        assert rlm_load["long_term_item_ids"]
        assert file_ingest["deduped"] is True
        assert file_ingest["long_term_item_ids"] == rlm_load["long_term_item_ids"]
        assert counts.get("rlm_chunk", 0) > 0
        assert counts.get("file_chunk", 0) == 0
    finally:
        mf.close()


def test_ltm_source_fetch_rehydrates_raw_content_by_item_id(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Alice selected SQLite on Tuesday."}],
            session_id="session-1",
        )
        hit = mf.recall_long_term("agent", "Alice SQLite Tuesday", top_k=1)[0]
        source = mf.long_term_source("agent", hit["item_id"])

        assert source is not None
        assert source["raw_ref"] == hit["raw_ref"]
        assert source["content"] == "Alice selected SQLite on Tuesday."
    finally:
        mf.close()
