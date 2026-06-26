from memoryforge import MemoryForge
from memoryforge.lcm.refs import HashRefResolver


def test_memoryforge_end_to_end_conversation_ltm_and_refs(tmp_path):
    db_path = str(tmp_path / "memory.db")
    mf = MemoryForge(db_path)
    turn_ids = mf.store_conversation(
        "agent",
        [{"role": "user", "content": "We decided to use SQLite."}],
        session_id="s1",
    )
    assert turn_ids
    assert mf.search("agent", "SQLite")
    assert mf.recall_long_term("agent", "SQLite", top_k=5)

    content_id, _ = mf.conversations.content.store("expanded content")
    assert HashRefResolver(db_path).expand(f"[ref:{content_id}]") == "expanded content"
