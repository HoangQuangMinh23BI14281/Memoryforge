import json
import sqlite3
from datetime import datetime, timezone

from memoryforge import MemoryForge
from memoryforge.memory.longterm.models import (
    MemoryConfidence,
    MetadataField,
)


def test_conversation_indexes_ltm_bm25_and_vector(tmp_path, real_data_text):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        result = mf.ingest_prompt(
            "agent",
            real_data_text,
            session_id="session-1",
        )

        recall = mf.recall_long_term(
            "agent", "basal cell skin cancer", top_k=5, include_content=True
        )

        assert result["turn_ids"]
        assert recall
        assert any("basal cell" in item["content"].lower() for item in recall)
        assert len(mf.conversations.get_session("agent", "session-1")) == 1
    finally:
        mf.close()


def test_long_term_source_restores_full_conversation_message(tmp_path, real_data_text):
    db_path = str(tmp_path / "memory.db")
    prompt = real_data_text
    mf = MemoryForge(db_path)
    try:
        result = mf.ingest_prompt("agent", prompt)
        hit = mf.recall_long_term("agent", "basal cell skin cancer", top_k=1, include_content=True)[
            0
        ]
        source = mf.long_term_source("agent", hit["item_id"])

        assert result["session_id"]
        assert result["turn_ids"]
        assert source is not None
        assert source["raw_ref"] == hit["raw_ref"]
        assert source["content"] == prompt
        assert len(source["content"]) > len(hit["preview"])
    finally:
        mf.close()


def test_core_context_bundle_is_bundle_not_answer(tmp_path, real_data_text):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.ingest_prompt("agent", real_data_text, session_id="session-1")

        bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="basal cell skin cancer",
            top_k=3,
        )
        payload = bundle.to_dict()

        assert payload["messages"][0]["source"] == "active_recall"
        assert any(message["source"] == "long_term" for message in payload["messages"])
        assert payload["active_recall"]
        assert payload["long_term_recall"]
        assert payload["raw_refs"]
        assert payload["provenance"]
        assert all(item.get("source_origin") for item in payload["provenance"])
        assert all(item.get("raw_refs") for item in payload["provenance"])
        assert any(item.get("source_origin") == "user" for item in payload["provenance"])
        assert any(item.get("surface") == "active_recall" for item in payload["provenance"])
        assert payload["diagnostics"]["bundle_only"] is True
        assert payload["diagnostics"]["answer_model_used"] is False
        assert payload["diagnostics"]["active_recall"]["count"] == len(
            payload["active_recall"]
        )
        assert payload["diagnostics"]["active_recall"]["query_required"] is False
        assert payload["diagnostics"]["active_recall"]["answer_model_used"] is False
        assert payload["diagnostics"]["retrieval"]["long_term_count"] == len(
            payload["long_term_recall"]
        )
        assert "content" not in payload["long_term_recall"][0]
    finally:
        mf.close()


def test_core_context_bundle_injects_active_recall_for_core_model(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [
                {"role": "user", "content": "I prefer SQLite for Project Atlas memory."},
                {"role": "assistant", "content": "We decided to keep schema changes small."},
                {"role": "user", "content": "TODO follow up on context bundle latency."},
            ],
            session_id="session-1",
        )

        bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session-1",
            query="Continue Project Atlas",
            top_k=5,
        ).to_dict()
        sources = [message["source"] for message in bundle["messages"]]
        active_message = next(
            message for message in bundle["messages"] if message["source"] == "active_recall"
        )

        assert sources.index("active_recall") < sources.index("long_term")
        assert "MemoryForge active recall" in active_message["content"]
        assert any("recent_evidence" in item["reasons"] for item in bundle["active_recall"])
        assert bundle["diagnostics"]["active_recall"]["semantic_focus_used"] is False
        assert bundle["diagnostics"]["answer_model_used"] is False
    finally:
        mf.close()


def test_ltm_writes_provenance_without_inferred_typed_metadata(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        event_date = datetime(2026, 6, 20, tzinfo=timezone.utc).timestamp()
        mf.store_conversation(
            "agent",
            [
                {"role": "user", "content": "I prefer SQLite for Project Atlas memory."},
                {"role": "assistant", "content": "We decided to keep schema changes small."},
                {"role": "user", "content": "TODO follow up on context bundle latency."},
            ],
            session_id="session-1",
            event_date=event_date,
        )

        hit = mf.recall_long_term("agent", "SQLite Project Atlas", top_k=1)[0]

        assert MetadataField.KIND not in hit["metadata"]
        assert "controller" not in hit["metadata"]
        assert "entities" not in hit["metadata"]
        assert "time_tokens" not in hit["metadata"]
        assert hit["metadata"]["source_origin"] == "user"
        assert hit["metadata"]["raw_refs"]
        assert hit["metadata"]["event_date"] == event_date
        assert hit["metadata"]["timestamp"].startswith("2026-06-20T")
    finally:
        mf.close()


def test_turn_metadata_preserves_explicit_metadata_without_deriving_types(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        message_ids = mf.store_conversation(
            "agent",
            [
                {
                    "role": "user",
                    "content": "SQLite is the local durable store for Atlas.",
                    "metadata": {MetadataField.KIND: "preference"},
                },
                {
                    "role": "assistant",
                    "content": "Run latency smoke before release signoff.",
                    "metadata": {MetadataField.KIND: "procedural"},
                },
                {
                    "role": "user",
                    "content": "Short direct responses match the collaboration contract.",
                    "metadata": {MetadataField.KIND: "identity"},
                },
                {
                    "role": "assistant",
                    "content": "Atlas is waiting on vector latency evidence.",
                    "metadata": {MetadataField.KIND: "project_state"},
                },
                {
                    "role": "assistant",
                    "content": "Mina handles Calypso billing ownership.",
                    "metadata": {
                        MetadataField.KIND: "entity_profile",
                        "profile_key": "Mina",
                    },
                },
                {
                    "role": "assistant",
                    "content": "Release checklist needs benchmark evidence.",
                    "metadata": {MetadataField.KIND: "task"},
                },
            ],
            session_id="session-1",
        )

        rows = mf.long_term.conn.execute(
            """
            SELECT source_id, metadata
            FROM long_term_items
            WHERE agent_id = ? AND source_type = 'message'
            """,
            ("agent",),
        ).fetchall()
        metadata_by_source_id = {str(row[0]): json.loads(row[1]) for row in rows}

        preference = metadata_by_source_id[message_ids[0]]
        procedural = metadata_by_source_id[message_ids[1]]
        identity = metadata_by_source_id[message_ids[2]]
        project_state = metadata_by_source_id[message_ids[3]]
        entity_profile = metadata_by_source_id[message_ids[4]]
        task = metadata_by_source_id[message_ids[5]]

        assert preference[MetadataField.KIND] == "preference"
        assert procedural[MetadataField.KIND] == "procedural"
        assert identity[MetadataField.KIND] == "identity"
        assert project_state[MetadataField.KIND] == "project_state"
        assert entity_profile[MetadataField.KIND] == "entity_profile"
        assert entity_profile["profile_key"] == "Mina"
        assert MetadataField.MERGE_KEY not in entity_profile
        assert task[MetadataField.KIND] == "task"
        assert "task_status" not in task
    finally:
        mf.close()


def test_marker_text_does_not_create_project_state_or_entity_profile_metadata(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [
                {
                    "role": "assistant",
                    "content": "Project state: Project Atlas is blocked on vector latency.",
                },
                {
                    "role": "assistant",
                    "content": "Entity profile: Mina owns the Calypso billing migration.",
                },
            ],
            session_id="session-1",
        )

        project_state = mf.recall_long_term("agent", "Project Atlas blocked latency", top_k=1)[0]
        entity_profile = mf.recall_long_term("agent", "Mina Calypso billing", top_k=1)[0]
        active = mf.active_recall("agent", session_id="session-1", focus="Project Atlas Mina", limit=5)
        active_reasons = {reason for item in active["results"] for reason in item["reasons"]}

        assert MetadataField.KIND not in project_state["metadata"]
        assert MetadataField.KIND not in entity_profile["metadata"]
        assert active_reasons == {"recent_evidence", "same_session"}
    finally:
        mf.close()


def test_controller_ignored_metadata_stays_out_of_active_recall(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        item_id = mf.long_term.index_raw_item(
            "agent",
            "note",
            "transient-note",
            "Acknowledged.",
            metadata={MetadataField.CONTROLLER_IGNORED: True},
        )
        source = mf.long_term_source("agent", item_id)
        active = mf.active_recall("agent", limit=5)

        assert source is not None
        assert source["metadata"][MetadataField.CONTROLLER_IGNORED] is True
        assert MetadataField.KIND not in source["metadata"]
        assert MetadataField.CONFIDENCE not in source["metadata"]
        assert all(item["item_id"] != item_id for item in active["results"])
    finally:
        mf.close()


def test_task_update_policy_uses_metadata_status_and_merge_key(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        merge_key = "release checklist"
        open_task_id = mf.long_term.index_raw_item(
            "agent",
            "task",
            "task-open",
            "Track the release checklist.",
            metadata={
                MetadataField.KIND: "task",
                "task_status": "open",
                MetadataField.MERGE_KEY: merge_key,
                MetadataField.CONFIDENCE: MemoryConfidence.MEDIUM,
            },
        )
        closed_task_id = mf.long_term.index_raw_item(
            "agent",
            "task",
            "task-closed",
            "The release checklist is complete.",
            metadata={
                MetadataField.KIND: "task",
                "task_status": "closed",
                MetadataField.MERGE_KEY: merge_key,
                MetadataField.CONFIDENCE: MemoryConfidence.HIGH,
            },
        )
        open_source = mf.long_term_source("agent", open_task_id)
        closed_source = mf.long_term_source("agent", closed_task_id)
        active = mf.active_recall("agent", limit=5)

        assert open_source is not None
        assert closed_source is not None
        assert open_source["metadata"][MetadataField.SUPERSEDED_BY] == [closed_task_id]
        assert open_source["metadata"][MetadataField.VALID_TO]
        assert open_source["metadata"]["task_status"] == "open"
        assert closed_source["metadata"][MetadataField.SUPERSEDES] == [open_task_id]
        active_ids = {item["item_id"] for item in active["results"]}
        assert open_task_id not in active_ids
        assert closed_task_id in active_ids
    finally:
        mf.close()


def test_explicit_merge_key_supersedes_previous_item_atomically(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        profile_key = "profile subject"
        merge_key = f"profile:{profile_key}"
        old_profile_id = mf.long_term.index_raw_item(
            "agent",
            "note",
            "profile-old",
            "Profile record version one.",
            metadata={
                MetadataField.KIND: "profile",
                "profile_key": profile_key,
                MetadataField.MERGE_KEY: merge_key,
            },
        )
        statements = []
        mf.long_term.conn.set_trace_callback(statements.append)
        new_profile_id = mf.long_term.index_raw_item(
            "agent",
            "note",
            "profile-new",
            "Profile record version two.",
            metadata={
                MetadataField.KIND: "profile",
                "profile_key": profile_key,
                MetadataField.MERGE_KEY: merge_key,
            },
        )
        old_source = mf.long_term_source("agent", old_profile_id)
        new_source = mf.long_term_source("agent", new_profile_id)

        assert old_source is not None
        assert new_source is not None
        assert old_source["metadata"][MetadataField.SUPERSEDED_BY] == [new_profile_id]
        assert old_source["metadata"].get(MetadataField.CONFIDENCE) is None
        assert new_source["metadata"][MetadataField.SUPERSEDES] == [old_profile_id]
        assert new_source["metadata"][MetadataField.MERGE_KEY] == merge_key
        assert any(statement.upper().startswith("BEGIN IMMEDIATE") for statement in statements)
    finally:
        if mf._long_term is not None:
            mf.long_term.conn.set_trace_callback(None)
        mf.close()


def test_active_recall_surfaces_lifecycle_memory_without_query(tmp_path):
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
        mf.store_conversation(
            "agent",
            [
                {"role": "user", "content": "I prefer SQLite for local durable memory."},
                {"role": "assistant", "content": "We decided to keep schema changes small."},
                {"role": "user", "content": "TODO follow up on context bundle latency."},
            ],
            session_id="session-1",
        )
        correction = mf.record_correction(
            "agent",
            "Project Atlas retention key is amber-17.",
            wrong_item_id=wrong_item_id,
            session_id="session-1",
        )

        recall = mf.active_recall("agent", session_id="session-1", limit=4)
        item_ids = {item["item_id"] for item in recall["results"]}

        assert recall["diagnostics"]["query_required"] is False
        assert recall["diagnostics"]["answer_model_used"] is False
        assert correction["item_id"] in item_ids
        assert wrong_item_id not in item_ids
        assert recall["raw_refs"]
        assert any("correction" in item["reasons"] for item in recall["results"])
    finally:
        mf.close()


def test_correction_does_not_split_task_text_from_wrong_memory(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [
                {
                    "role": "assistant",
                    "content": (
                        "TODO follow up on context bundle latency before release. "
                        "Project Atlas retention key is blue-12."
                    ),
                }
            ],
            session_id="session-1",
        )
        wrong = mf.recall_long_term(
            "agent",
            "Project Atlas retention key blue-12",
            top_k=1,
            include_content=True,
        )[0]

        correction = mf.record_correction(
            "agent",
            "Project Atlas retention key is amber-17.",
            wrong_item_id=wrong["item_id"],
            session_id="session-1",
        )
        active = mf.active_recall(
            "agent",
            session_id="session-1",
            focus="Project Atlas latency",
            limit=5,
            include_content=True,
        )
        active_ids = {item["item_id"] for item in active["results"]}
        assert correction["preserved_item_ids"] == []
        assert correction["item_id"] in active_ids
        assert wrong["item_id"] not in active_ids
    finally:
        mf.close()


def test_contradiction_metadata_links_conflicting_memories_without_schema_change(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        original_item_id = mf.long_term.index_raw_item(
            "agent",
            "note",
            "atlas-owner-mina",
            "Project Atlas owner is Mina.",
            metadata={
                "kind": "fact",
                "confidence": "normal",
                "entities": ["Project Atlas", "Mina"],
            },
        )

        contradiction = mf.record_contradiction(
            "agent",
            "Project Atlas owner is Rina.",
            conflicting_item_ids=[original_item_id],
            session_id="session-1",
        )
        original_source = mf.long_term_source("agent", original_item_id)
        found = mf.find_contradictions(
            "agent",
            query="Project Atlas owner",
            include_content=True,
        )
        found_ids = {item["item_id"] for item in found["results"]}

        assert original_source is not None
        assert original_source["metadata"]["contradicted_by"] == [contradiction["item_id"]]
        assert original_source["metadata"]["contradicts"] == [contradiction["raw_ref"]]
        assert MetadataField.KIND not in contradiction["metadata"]
        assert contradiction["metadata"]["conflicting_item_ids"] == [original_item_id]
        assert found["diagnostics"]["answer_model_used"] is False
        assert contradiction["item_id"] in found_ids
        assert original_item_id in found_ids
        assert any("Project Atlas owner is Rina" in item["content"] for item in found["results"])
    finally:
        mf.close()


def test_record_contradiction_rejects_missing_conflicts(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        try:
            mf.record_contradiction("agent", "Project Atlas owner is Rina.")
        except ValueError as exc:
            assert "requires conflicting_item_ids or conflicting_raw_refs" in str(exc)
        else:
            raise AssertionError("record_contradiction accepted an unlinked statement")

        try:
            mf.record_contradiction(
                "agent",
                "Project Atlas owner is Rina.",
                conflicting_item_ids=["missing-item"],
            )
        except ValueError as exc:
            assert "Unknown conflicting item_id" in str(exc)
        else:
            raise AssertionError("record_contradiction accepted an unknown item")
    finally:
        mf.close()


def test_referenced_files_use_rlm_chunks_and_do_not_enter_lcm(tmp_path, real_data_excerpt):
    db_path = str(tmp_path / "memory.db")
    project = tmp_path / "project"
    project.mkdir()
    marker = "basal cell skin cancer"
    notes_path = project / "notes.md"
    notes_path.write_text(real_data_excerpt.read_text(encoding="utf-8"), encoding="utf-8")

    mf = MemoryForge(db_path)
    try:
        result = mf.ingest_prompt(
            "agent",
            "Please read notes.md",
            session_id="session-files",
            project_root=str(project),
        )
        session_turns = mf.conversations.get_session("agent", "session-files")
        lcm_messages = mf.lcm_store.get_messages("session-files", include_summaries=False)
        hits = mf.recall_long_term("agent", marker, top_k=5, include_content=True)

        assert result["files_ingested"] == 1
        assert result["long_term_file_item_ids"]
        assert len(session_turns) == 1
        assert marker not in session_turns[0]["content"]
        assert len(lcm_messages) == 1
        assert marker not in lcm_messages[0].content
        assert any(hit["source_type"] == "rlm_chunk" and marker in hit["content"] for hit in hits)
        assert all(MetadataField.KIND not in hit["metadata"] for hit in hits)
        assert any(hit["metadata"]["source_origin"] == "file" for hit in hits)
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM rlm_buffers").fetchone()[0] == 1
    finally:
        mf.close()
