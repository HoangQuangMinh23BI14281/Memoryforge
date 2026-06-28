def test_public_imports():
    import memoryforge

    assert memoryforge.__version__ == "6.1.1"
    assert memoryforge.ContentHashTable
    assert memoryforge.BM25Index
    assert memoryforge.MemoryForge
    assert memoryforge.RLMEngine
