def test_public_imports():
    import memoryforge

    assert memoryforge.__version__ == "6.0.0"
    assert memoryforge.ContentHashTable
    assert memoryforge.BM25Index
    assert memoryforge.MemoryForge
    assert memoryforge.RLMEngine
