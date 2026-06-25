from memoryforge.search import UnifiedQueryRouter


class FakeSearch:
    def __init__(self, key):
        self.key = key

    def search(self, query, top_k=10):
        return [{self.key: f"{self.key}-1", "score": 1.0, "query": query}]


def test_router_uses_memory_search():
    router = UnifiedQueryRouter(lcm_search=FakeSearch("content_id"))
    result = router.query("authentication", top_k=3)
    assert result["query_type"] == "memory"
    assert len(result["results"]) == 1
