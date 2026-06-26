from memoryforge.agents import BaseSubAgentRunner, SubAgentResponse
from memoryforge.lcm import (
    ConversationStore,
    LCMCompactor,
    Message,
    SubAgentLCMProvider,
    SummaryDAG,
)


class SyncProvider:
    name = "test-sync"
    model = "test-model"

    def complete(self, **kwargs):
        return "Short provider summary file_abc123."


class L2Provider:
    name = "test-l2"
    model = "test-model"

    def complete(self, **kwargs):
        if kwargs["max_tokens"] == 2000:
            return "x" * 20_000
        return "Aggressive provider summary file_abc123."


class AsyncProvider:
    name = "test-async"
    model = "test-model"

    async def complete(self, **kwargs):
        return "Async provider summary file_abc123."


class FakeSubAgentRunner(BaseSubAgentRunner):
    provider = "fake"

    def __init__(self, outputs):
        super().__init__(model="fake-model")
        self.outputs = list(outputs)
        self.prompts = []

    def complete(self, prompt: str) -> SubAgentResponse:
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        return SubAgentResponse(self.provider, self.model, output, 0.01)


def test_lcm_compactor_l3_fallback():
    compactor = LCMCompactor()
    result = compactor.compact(
        [
            Message(role="user", content="We decided to use SQLite for local storage."),
            Message(role="assistant", content="The implementation plan is Python plus SQLite."),
        ]
    )
    assert result.level_used == 3
    assert "SQLite" in result.summary


def test_lcm_compactor_l1_runs_with_provider():
    compactor = LCMCompactor(llm_provider=SyncProvider())

    result = compactor.compact(
        [Message(role="user", content="Please inspect file_abc123 " + "x" * 2000)]
    )

    assert result.level_used == 1
    assert "Short provider summary" in result.summary
    assert result.file_ids_preserved == ["file_abc123"]
    assert compactor.provider_info()["model"] == "test-model"


def test_lcm_compactor_l2_runs_when_l1_does_not_converge():
    compactor = LCMCompactor(llm_provider=L2Provider())

    result = compactor.compact(
        [Message(role="user", content="Please inspect file_abc123 " + "x" * 2000)]
    )

    assert result.level_used == 2
    assert "Aggressive provider summary" in result.summary
    assert result.file_ids_preserved == ["file_abc123"]


def test_lcm_compactor_supports_async_provider_from_sync_context():
    compactor = LCMCompactor(llm_provider=AsyncProvider())

    result = compactor.compact(
        [Message(role="user", content="Please inspect file_abc123 " + "x" * 2000)]
    )

    assert result.level_used == 1
    assert "Async provider summary" in result.summary


def test_lcm_l1_uses_subagent_provider_first():
    runner = FakeSubAgentRunner(["Sub-agent L1 summary."])
    compactor = LCMCompactor(llm_provider=SubAgentLCMProvider(subagent=runner))

    result = compactor.compact(
        [Message(role="user", content="Please inspect file_abc123 " + "x" * 2000)]
    )

    assert result.level_used == 1
    assert "Sub-agent L1 summary" in result.summary
    assert "file_abc123" in result.summary
    assert "MemoryForge LCM compaction sub-agent" in runner.prompts[0]


def test_lcm_l2_uses_subagent_when_l1_does_not_converge():
    runner = FakeSubAgentRunner(["x" * 20_000, "Sub-agent L2 compressed summary."])
    compactor = LCMCompactor(llm_provider=SubAgentLCMProvider(subagent=runner))

    result = compactor.compact(
        [Message(role="user", content="Please inspect file_abc123 " + "x" * 2000)]
    )

    assert result.level_used == 2
    assert "Sub-agent L2 compressed summary" in result.summary
    assert len(runner.prompts) == 2


def test_summary_dag_condense(tmp_path):
    dag = SummaryDAG(str(tmp_path / "memory.db"))
    first = dag.create_leaf("s1", "first summary file_aaa111", "m1", "m2")
    second = dag.create_leaf("s1", "second summary file_bbb222", "m3", "m4")
    parent = dag.condense([first, second], "condensed")

    active = dag.get_active_summaries("s1")
    assert [node.id for node in active] == [parent]
    assert active[0].parent_node_ids == [first, second]
    assert active[0].file_ids == ["file_aaa111", "file_bbb222"]
    assert "file_aaa111" in active[0].content
    assert "file_bbb222" in active[0].content


def test_conversation_store_search(tmp_path):
    store = ConversationStore(str(tmp_path / "memory.db"))
    store.store_session(
        "agent",
        [
            {"role": "user", "content": "I adopted a corgi named Mochi."},
            {"role": "assistant", "content": "Mochi sounds great."},
        ],
        session_id="session",
    )

    results = store.search("agent", "corgi Mochi", top_k=5)
    assert results
    assert any("Mochi" in result["content"] for result in results)
