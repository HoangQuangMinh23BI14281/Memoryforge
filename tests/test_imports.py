def test_public_imports():
    import memoryforge

    assert memoryforge.__version__ == "6.1.5"
    assert memoryforge.ContentHashTable
    assert memoryforge.BM25Index
    assert memoryforge.MemoryForge
    assert not hasattr(memoryforge, "MemoryForgeSession")
    assert memoryforge.RLMEngine
