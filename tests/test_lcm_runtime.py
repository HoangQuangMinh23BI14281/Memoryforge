from memoryforge import MemoryForge
from memoryforge.cli.main import main
from memoryforge.lcm import (
    CompactionResult,
    ContextBudget,
    ContextBuilder,
    EventBus,
    ImmutableMessageStore,
    LCMCompactionEngine,
    SummaryDAG,
    ToolOutputPruner,
)
from memoryforge.lcm.compaction.file_ids import is_internal_source_ref


class ExpandingCompactor:
    def compact(self, messages):
        return CompactionResult(
            summary="expanded summary " * 2000,
            level_used=1,
            content_ids=[],
            token_count=2000,
            messages_covered=len(messages),
            file_ids_preserved=[],
        )


class FailingCompactor:
    def compact(self, messages):
        raise AssertionError("compactor should not run for deferred soft overflow")


class NoCallCompactor:
    def compact(self, messages):
        raise AssertionError("compactor should not run when cached compaction is reusable")


class ShortCompactor:
    def __init__(self):
        self.calls = 0

    def compact(self, messages):
        self.calls += 1
        return CompactionResult(
            summary="short durable summary",
            level_used=1,
            content_ids=[],
            token_count=4,
            messages_covered=len(messages),
            file_ids_preserved=[],
        )


def test_context_builder_uses_summary_without_deleting_raw(tmp_path):
    db_path = str(tmp_path / "memory.db")
    store = ImmutableMessageStore(db_path)
    dag = SummaryDAG(db_path)
    try:
        old_message_id = store.append_text_message(
            "agent", "session", "user", "old raw detail " * 200
        )
        store.append_text_message("agent", "session", "assistant", "recent answer")
        summary_id = dag.create_leaf(
            "session", "summary preserves old raw detail", old_message_id, old_message_id
        )
        store.swap_context_items("session", [old_message_id], summary_id)

        context = ContextBuilder(store, dag).build(
            "session",
            budget=ContextBudget(
                model_context_limit=900, reserved_output_tokens=100, compaction_buffer=100
            ),
        )

        assert summary_id in context.summary_node_ids
        assert old_message_id not in context.raw_message_ids
        assert store.get_message(old_message_id).content.startswith("old raw detail")
    finally:
        store.close()
        dag.close()


def test_lcm_session_root_is_separate_from_context_items_view(tmp_path):
    db_path = str(tmp_path / "memory.db")
    store = ImmutableMessageStore(db_path)
    try:
        store.ensure_session(
            "agent",
            "session",
            system_prompt="system rules",
            model_id="model-a",
            provider_id="provider-a",
        )
        message_id = store.append_text_message("agent", "session", "user", "hello")

        session_row = store.conn.execute(
            """
            SELECT id, agent, system_prompt, model_id, provider_id
            FROM sessions
            WHERE id = ?
            """,
            ("session",),
        ).fetchone()
        context_items = store.get_context_items("session")

        assert session_row == ("session", "agent", "system rules", "model-a", "provider-a")
        assert context_items == [("message", message_id)]
    finally:
        store.close()


def test_conversation_messages_link_to_content_backed_message_parts(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turn_ids = mf.store_conversation(
            "agent",
            [{"role": "user", "content": "Raw content should be stored once."}],
            session_id="session",
        )

        turn = mf.conversations.get_turns_by_ids("agent", turn_ids)[0]
        message = mf.lcm_store.get_message(turn["message_id"])
        part = message.parts[0]

        assert turn["part_id"] == part.id
        assert turn["content_id"] == part.content_id
        assert (
            mf.conversations.content.retrieve(part.content_id)
            == "Raw content should be stored once."
        )
    finally:
        mf.close()


def test_tool_output_pruner_tombstones_context_only(tmp_path):
    db_path = str(tmp_path / "memory.db")
    store = ImmutableMessageStore(db_path)
    dag = SummaryDAG(db_path)
    try:
        store.append_message(
            "agent",
            "session",
            "assistant",
            [
                {"part_type": "text", "content": "tool call finished"},
                {
                    "part_type": "tool",
                    "tool_name": "search",
                    "tool_call_id": "tool-1",
                    "content": "RAW_TOOL_OUTPUT_SECRET " * 100,
                },
            ],
        )
        store.append_text_message("agent", "session", "user", "recent prompt")

        pruned = ToolOutputPruner(store).prune(
            "session",
            protect_recent_messages=1,
            protect_tokens=0,
            min_prunable_tokens=1,
        )
        context = ContextBuilder(store, dag).build("session")
        old_message = store.get_messages("session", include_summaries=False)[0]

        assert pruned.pruned_count == 1
        assert "RAW_TOOL_OUTPUT_SECRET" not in "\n".join(
            message.content for message in context.messages
        )
        assert "Tool output compacted" in "\n".join(message.content for message in context.messages)
        assert "RAW_TOOL_OUTPUT_SECRET" in old_message.parts[1].content
    finally:
        store.close()
        dag.close()


def test_tool_output_pruner_respects_protected_tools(tmp_path):
    db_path = str(tmp_path / "memory.db")
    store = ImmutableMessageStore(db_path)
    try:
        store.append_message(
            "agent",
            "session",
            "assistant",
            [
                {
                    "part_type": "tool",
                    "tool_name": "secure",
                    "tool_call_id": "tool-1",
                    "tool_state": "completed",
                    "protected": True,
                    "content": "PROTECTED_OUTPUT " * 100,
                },
            ],
        )
        store.append_text_message("agent", "session", "user", "recent prompt")

        pruned = ToolOutputPruner(store).prune(
            "session",
            protect_recent_messages=1,
            protect_tokens=0,
            min_prunable_tokens=1,
        )

        assert pruned.pruned_count == 0
        assert (
            store.get_messages("session", include_summaries=False)[0].parts[0].compacted_at is None
        )
    finally:
        store.close()


def test_lcm_engine_threshold_compacts_and_records_events(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 120)}
            for index in range(8)
        ]
        mf.store_conversation("agent", turns, session_id="session")

        result = mf.lcm_compact_if_needed(
            "agent",
            "session",
            budget=ContextBudget(
                model_context_limit=2_500, reserved_output_tokens=100, compaction_buffer=100
            ),
            runner="mock",
        )
        context = mf.lcm_build_context(
            "session",
            budget=ContextBudget(
                model_context_limit=2_500, reserved_output_tokens=100, compaction_buffer=100
            ),
        )
        events = EventBus(db_path).list_events("session")

        assert result.triggered is True
        assert result.rounds >= 1
        assert result.after_tokens <= result.before_tokens
        assert result.delta_tokens <= 0
        assert result.expanded is False
        assert result.effective is True
        assert result.summary_node_ids
        assert context.has_summary is True
        assert result.summary_node_ids[0] in context.summary_node_ids
        assert result.summary_node_ids[0] not in context.raw_message_ids
        assert mf.lcm_store.get_context_items("session")[0][0] == "summary"
        assert len(mf.lcm_store.get_messages("session", include_summaries=False)) == 8
        assert any(event.event_type == "lcm.compaction.completed" for event in events)
    finally:
        mf.close()


def test_lcm_defer_soft_overflow_does_not_call_compactor(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 120)}
            for index in range(6)
        ]
        mf.store_conversation("agent", turns, session_id="session")
    finally:
        mf.close()

    engine = LCMCompactionEngine(db_path, compactor=FailingCompactor())
    budget = ContextBudget(
        model_context_limit=4_500,
        reserved_output_tokens=300,
        compaction_buffer=200,
    )
    try:
        decision = engine.assess("session", budget=budget)
        result = engine.compact_if_needed(
            "agent",
            "session",
            budget=budget,
            defer_soft=True,
        )
        events = EventBus(db_path).list_events("session")

        assert decision.soft_overflow is True
        assert decision.hard_overflow is False
        assert result.triggered is False
        assert result.deferred is True
        assert result.reason == "soft_overflow_deferred"
        assert result.delta_tokens == 0
        assert any(event.event_type == "lcm.compaction.deferred" for event in events)
        assert engine.builder.build("session", budget=budget).has_summary is False
    finally:
        engine.close()


def test_lcm_deferred_soft_overflow_events_are_coalesced(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 120)}
            for index in range(6)
        ]
        mf.store_conversation("agent", turns, session_id="session")
    finally:
        mf.close()

    engine = LCMCompactionEngine(db_path, compactor=FailingCompactor())
    budget = ContextBudget(
        model_context_limit=4_500,
        reserved_output_tokens=300,
        compaction_buffer=200,
    )
    try:
        for _ in range(5):
            result = engine.compact_if_needed(
                "agent",
                "session",
                budget=budget,
                defer_soft=True,
            )
            assert result.deferred is True

        deferred_events = [
            event
            for event in EventBus(db_path).list_events("session")
            if event.event_type == "lcm.compaction.deferred"
        ]

        assert len(deferred_events) == 1
        assert deferred_events[0].payload["coalesced_count"] == 5
        assert deferred_events[0].payload["reason"] == "soft_overflow_deferred"
    finally:
        engine.close()


def test_lcm_defer_soft_still_compacts_hard_overflow(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 1_000)}
            for index in range(4)
        ]
        mf.store_conversation("agent", turns, session_id="session")
    finally:
        mf.close()

    compactor = ShortCompactor()
    engine = LCMCompactionEngine(db_path, compactor=compactor)
    budget = ContextBudget(
        model_context_limit=1_500,
        reserved_output_tokens=200,
        compaction_buffer=100,
    )
    try:
        decision = engine.assess("session", budget=budget)
        result = engine.compact_if_needed(
            "agent",
            "session",
            budget=budget,
            defer_soft=True,
            max_rounds=1,
        )

        assert decision.hard_overflow is True
        assert result.triggered is True
        assert result.deferred is False
        assert result.summary_node_ids
        assert compactor.calls == 1
    finally:
        engine.close()


def test_lcm_compaction_reuses_input_hash_cache_without_worker_call(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 120)}
            for index in range(8)
        ]
        mf.store_conversation("agent", turns, session_id="session")
    finally:
        mf.close()

    budget = ContextBudget(
        model_context_limit=2_500,
        reserved_output_tokens=100,
        compaction_buffer=100,
    )
    compactor = ShortCompactor()
    engine = LCMCompactionEngine(db_path, compactor=compactor)
    try:
        first = engine.compact_if_needed(
            "agent",
            "session",
            budget=budget,
            force=True,
            max_rounds=1,
            keep_recent_messages=2,
        )
        raw_messages = engine.store.get_messages("session", include_summaries=False)
        with engine.store.conn:
            engine.store.conn.execute("DELETE FROM context_items WHERE session_id = ?", ("session",))
            for position, message in enumerate(raw_messages, start=1):
                engine.store.conn.execute(
                    """
                    INSERT INTO context_items
                    (session_id, item_type, item_id, position, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("session", "message", message.id, position, str(position)),
                )
        node_count = engine.dag.conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]

        assert first.summary_node_ids
        assert first.cached is False
        assert compactor.calls == 1
    finally:
        engine.close()

    cached_engine = LCMCompactionEngine(db_path, compactor=NoCallCompactor())
    try:
        second = cached_engine.compact_if_needed(
            "agent",
            "session",
            budget=budget,
            force=True,
            max_rounds=1,
            keep_recent_messages=2,
        )
        after_node_count = (
            cached_engine.dag.conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
        )
        cached_node = cached_engine.dag.get_node(second.summary_node_ids[0])

        assert second.cached is True
        assert second.reason == "cached_compaction_reused"
        assert second.summary_node_ids == first.summary_node_ids
        assert after_node_count == node_count
        assert cached_node is not None
        assert any(is_internal_source_ref(ref) for ref in cached_node.source_refs)
        assert cached_engine.builder.build("session", budget=budget).has_summary is True
    finally:
        cached_engine.close()

    mf = MemoryForge(db_path)
    try:
        bundle = mf.build_core_context_bundle(
            agent_id="agent",
            session_id="session",
            query="large context",
            budget=budget,
        )

        assert bundle.summary_nodes
        assert not any(is_internal_source_ref(ref) for ref in bundle.raw_refs)
        assert not any(
            is_internal_source_ref(ref)
            for node in bundle.summary_nodes
            for ref in node.get("source_refs", [])
        )
        assert not any(
            is_internal_source_ref(ref)
            for item in bundle.provenance
            for ref in item.get("raw_refs", [])
        )
    finally:
        mf.close()


def test_lcm_compact_due_skips_soft_overflow_when_hard_only(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    budget = ContextBudget(
        model_context_limit=4_500,
        reserved_output_tokens=300,
        compaction_buffer=200,
    )
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 120)}
            for index in range(6)
        ]
        mf.store_conversation("agent", turns, session_id="session")

        result = mf.lcm_compact_due(
            "agent",
            budget=budget,
            hard_only=True,
        )

        assert result["checked"] == 1
        assert result["compacted"] == 0
        assert result["results"][0]["status"] == "not_due"
        assert result["results"][0]["decision"]["soft_overflow"] is True
        assert result["results"][0]["decision"]["hard_overflow"] is False
        assert mf.lcm_build_context("session", budget=budget).has_summary is False
    finally:
        mf.close()


def test_lcm_compact_due_runs_out_of_band_maintenance(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    budget = ContextBudget(
        model_context_limit=2_500,
        reserved_output_tokens=100,
        compaction_buffer=100,
    )
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 120)}
            for index in range(8)
        ]
        mf.store_conversation("agent", turns, session_id="session")

        result = mf.lcm_compact_due(
            "agent",
            budget=budget,
            runner="mock",
            max_rounds=1,
        )

        assert result["checked"] == 1
        assert result["compacted"] == 1
        assert result["results"][0]["status"] == "compacted"
        assert result["results"][0]["compaction"]["triggered"] is True
        assert mf.lcm_build_context("session", budget=budget).has_summary is True
    finally:
        mf.close()


def test_lcm_maintain_cli_runs_compaction(tmp_path, capsys):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        turns = [
            {"role": "user", "content": f"message {index} " + ("large context " * 120)}
            for index in range(8)
        ]
        mf.store_conversation("agent", turns, session_id="session")
    finally:
        mf.close()

    exit_code = main(
        [
            "--db",
            db_path,
            "lcm-maintain",
            "--agent-id",
            "agent",
            "--context-limit",
            "2500",
            "--reserved-output",
            "100",
            "--compaction-buffer",
            "100",
            "--runner",
            "mock",
            "--max-rounds",
            "1",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"compacted": 1' in output


def test_lcm_compaction_skips_context_expanding_summary(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    try:
        mf.store_conversation(
            "agent",
            [
                {"role": "user", "content": f"short message {index}"}
                for index in range(4)
            ],
            session_id="session",
        )
    finally:
        mf.close()

    engine = LCMCompactionEngine(db_path, compactor=ExpandingCompactor())
    try:
        result = engine.compact_if_needed(
            "agent",
            "session",
            force=True,
            max_rounds=1,
            keep_recent_messages=2,
            budget=ContextBudget(
                model_context_limit=300, reserved_output_tokens=50, compaction_buffer=50
            ),
        )
        completed = [
            event
            for event in EventBus(db_path).list_events("session")
            if event.event_type == "lcm.compaction.completed"
        ][0]
        skipped = [
            event
            for event in EventBus(db_path).list_events("session")
            if event.event_type == "lcm.compaction.skipped"
        ][0]

        assert result.triggered is True
        assert result.delta_tokens == result.after_tokens - result.before_tokens
        assert result.delta_tokens == 0
        assert result.expanded is False
        assert result.effective is False
        assert result.summary_node_ids == []
        assert completed.payload["expanded"] is False
        assert completed.payload["effective"] is False
        assert skipped.payload["reason"] == "no_context_convergence"
    finally:
        engine.close()
