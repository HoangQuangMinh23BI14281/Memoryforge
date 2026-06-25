"""Multi-model token estimation with caching and graceful fallback."""

from __future__ import annotations

import hashlib
from typing import Any


class TokenEstimator:
    """
    Multi-model token counting with caching and graceful fallback.

    Priority order:
    1. tiktoken for OpenAI model families (gpt-4, gpt-3.5, o-series)
    2. Character-based heuristic (``len // 4``) for all other models

    Caching:
    - Encoder objects are cached by encoding name (one load per process).
    - Token counts are cached by SHA-256 of content for immutable content
      (file references, summary nodes). Use ``estimate_cached()`` for this.
    """

    def __init__(self, *, heuristic_only: bool = False) -> None:
        """
        Args:
            heuristic_only: When ``True``, always use the character-based
                heuristic and skip tiktoken entirely. Useful for testing,
                benchmarking, or environments where tiktoken is not installed.
        """
        self._encoder_cache: dict[str, Any] = {}
        self._count_cache: dict[str, int] = {}
        self._force_heuristic: bool = heuristic_only

    def estimate(self, text: str, model: Any | None = None) -> int:
        """
        Estimate the token count for a string.

        Args:
            text: The text to estimate.
            model: Optional model info for accurate tokenisation. Uses heuristic
                when None or model encoding is unknown.

        Returns:
            Estimated token count, always >= 1 for non-empty text.
        """
        if not text:
            return 0
        if self._force_heuristic or model is None:
            return self._heuristic(text)

        encoding = model.encoding
        if encoding in ("cl100k_base", "o200k_base") and not self._force_heuristic:
            try:
                return self._tiktoken_estimate(text, encoding)
            except Exception:
                pass
        return self._heuristic(text)

    def estimate_cached(self, text: str, cache_key: str, model: Any | None = None) -> int:
        """
        Estimate with caching, keyed by ``cache_key``.

        Use for immutable content (file references, summary nodes) where the
        same text will be estimated multiple times across context builds.

        Args:
            text: The text to estimate.
            cache_key: A stable identifier for this content (e.g. SHA-256 hash).
            model: Optional model info for accurate tokenisation.

        Returns:
            Estimated token count from cache or fresh computation.
        """
        if cache_key in self._count_cache:
            return self._count_cache[cache_key]
        count = self.estimate(text, model)
        self._count_cache[cache_key] = count
        return count

    def _heuristic(self, text: str) -> int:
        """Conservative heuristic: 4 characters per token, minimum 1."""
        return max(1, len(text) // 4)

    def _tiktoken_estimate(self, text: str, encoding_name: str) -> int:
        """Encode with tiktoken, caching the encoder object."""
        if encoding_name not in self._encoder_cache:
            import tiktoken

            self._encoder_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        encoder = self._encoder_cache[encoding_name]
        return len(encoder.encode(text))

    @staticmethod
    def content_hash(text: str) -> str:
        """Return a stable SHA-256 hex digest for use as a cache key."""
        return hashlib.sha256(text.encode()).hexdigest()
