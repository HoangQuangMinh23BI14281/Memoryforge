from memoryforge import ContentHashTable


def test_content_store_deduplicates_and_rehydrates_raw_text(tmp_path):
    store = ContentHashTable(str(tmp_path / "memory.db"))
    try:
        first_id, first_new = store.store("Alice uses SQLite for local memory.")
        second_id, second_new = store.store("Alice uses SQLite for local memory.")

        assert first_id == second_id
        assert first_new is True
        assert second_new is False
        assert store.retrieve(first_id) == "Alice uses SQLite for local memory."
    finally:
        store.conn.close()
