"""MemoryForge v6.0 - Lossless Context Management for Recursive Language Model.

Recursive Language Model substrate:
- LCM Pipeline: active conversation context with recursive compaction
- LTM Pipeline: durable file/conversation knowledge with BM25 and vector recall
- Search Pipeline: BM25/vector fusion for pass-by-reference context recall
"""

__version__ = "6.1.1"

from memoryforge._core import BM25Index, ContentHashTable, rrf_fusion
from memoryforge.agents import create_subagent_runner
from memoryforge.api import CoreContextBundle, MemoryForge
from memoryforge.lcm import LCMCompactor, SummaryDAG
from memoryforge.rlm import RLMEngine
from memoryforge.search.router import UnifiedQueryRouter
from memoryforge.search.vector import VectorIndex

__all__ = [
    "ContentHashTable",
    "BM25Index",
    "rrf_fusion",
    "LCMCompactor",
    "SummaryDAG",
    "UnifiedQueryRouter",
    "VectorIndex",
    "MemoryForge",
    "CoreContextBundle",
    "RLMEngine",
    "create_subagent_runner",
]
