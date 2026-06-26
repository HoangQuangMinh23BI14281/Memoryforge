from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

INTERSHIP_ROOT = Path("/mnt/c/users/admin/onedrive/desktop/intership")
DEFAULT_REAL_DATA = INTERSHIP_ROOT / "data.md"


@pytest.fixture
def real_data_path() -> Path:
    path = Path(os.environ.get("MEMORYFORGE_REAL_DATA_PATH", str(DEFAULT_REAL_DATA))).expanduser()
    if not path.exists():
        pytest.skip(f"Real data file not found: {path}")
    return path


@pytest.fixture
def real_data_excerpt(tmp_path: Path, real_data_path: Path) -> Path:
    max_chars = int(os.environ.get("MEMORYFORGE_REAL_DATA_CHARS", "12000"))
    text = real_data_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    if not text.strip():
        pytest.skip(f"Real data file is empty: {real_data_path}")
    path = tmp_path / "data.md"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def real_data_text(real_data_excerpt: Path) -> str:
    return real_data_excerpt.read_text(encoding="utf-8")


@pytest.fixture
def real_runner_args() -> dict[str, object]:
    runner = os.environ.get("MEMORYFORGE_SUBAGENT_RUNNER", "codex").lower()
    if runner != "codex":
        pytest.fail("Real sub-agent tests must use MEMORYFORGE_SUBAGENT_RUNNER=codex")
    if shutil.which("codex") is None:
        pytest.fail("Real Codex sub-agent tests require Codex CLI on PATH")
    model = os.environ.get("MEMORYFORGE_MODEL") or os.environ.get("OPENAI_MODEL")
    if not model:
        pytest.fail("Set MEMORYFORGE_MODEL or OPENAI_MODEL for real Codex sub-agent tests")
    base_url = os.environ.get("MEMORYFORGE_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    return {
        "runner": runner,
        "model": model,
        "base_url": base_url,
        "project_root": str(INTERSHIP_ROOT),
        "timeout_s": float(os.environ.get("MEMORYFORGE_REAL_TIMEOUT", "180")),
    }
