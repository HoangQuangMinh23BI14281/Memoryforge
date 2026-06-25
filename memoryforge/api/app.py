"""High-level MemoryForge API."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from memoryforge.api.active_recall import ActiveRecallMixin
from memoryforge.api.context_builder import ContextBuilderMixin
from memoryforge.api.lcm_facade import LCMFacadeMixin
from memoryforge.api.lifecycle import MemoryLifecycleMixin
from memoryforge.api.memory_io import MemoryIOMixin
from memoryforge.api.rlm_facade import RLMFacadeMixin
from memoryforge.lcm import ImmutableMessageStore, LCMCompactionEngine
from memoryforge.lcm.conversation import ConversationStore
from memoryforge.memory.longterm.store import LongTermMemoryIndex
from memoryforge.rlm.chunking import ChunkingStrategy
from memoryforge.rlm.engine import RLMEngine


class MemoryForge(
    MemoryIOMixin,
    ContextBuilderMixin,
    LCMFacadeMixin,
    ActiveRecallMixin,
    MemoryLifecycleMixin,
    RLMFacadeMixin,
):
    def __init__(
        self,
        db_path: str = "~/.memoryforge/memory.db",
    ):
        self.db_path = str(Path(db_path).expanduser())
        self._conversations: ConversationStore | None = None
        self._long_term: LongTermMemoryIndex | None = None
        self._chunking: ChunkingStrategy | None = None
        self._lcm_store: ImmutableMessageStore | None = None
        self._lcm_engine: LCMCompactionEngine | None = None
        self._rlm: RLMEngine | None = None

    @property
    def conversations(self) -> ConversationStore:
        if self._conversations is None:
            self._conversations = ConversationStore(self.db_path)
        return self._conversations

    @property
    def long_term(self) -> LongTermMemoryIndex:
        if self._long_term is None:
            self._long_term = LongTermMemoryIndex(self.db_path)
        return self._long_term

    @property
    def chunking(self) -> ChunkingStrategy:
        if self._chunking is None:
            self._chunking = ChunkingStrategy()
        return self._chunking

    @property
    def lcm_store(self) -> ImmutableMessageStore:
        if self._lcm_store is None:
            self._lcm_store = ImmutableMessageStore(self.db_path)
        return self._lcm_store

    @property
    def lcm_engine(self) -> LCMCompactionEngine:
        if self._lcm_engine is None:
            self._lcm_engine = LCMCompactionEngine(self.db_path)
        return self._lcm_engine

    @property
    def rlm(self) -> RLMEngine:
        if self._rlm is None:
            self._rlm = RLMEngine(self.db_path)
        return self._rlm

    def close(self) -> None:
        for attr_name in (
            "_conversations",
            "_long_term",
            "_chunking",
            "_lcm_store",
            "_lcm_engine",
            "_rlm",
        ):
            component = getattr(self, attr_name)
            if component is None:
                continue
            close = getattr(component, "close", None)
            if close:
                close()
            setattr(self, attr_name, None)

    def __enter__(self) -> MemoryForge:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

