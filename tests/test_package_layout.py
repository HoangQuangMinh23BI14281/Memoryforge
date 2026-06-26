from pathlib import Path

from memoryforge.api import MemoryForge
from memoryforge.cli import build_parser
from memoryforge.init import init_project
from memoryforge.memory.longterm import LongTermMemoryIndex, LongTermRecallResult
from memoryforge.rlm import ChunkingStrategy, RLMEngine


def test_public_entrypoints_survive_folder_refactor():
    assert MemoryForge
    assert RLMEngine
    assert ChunkingStrategy
    assert LongTermMemoryIndex
    assert LongTermRecallResult
    assert init_project
    assert build_parser


def test_removed_temporary_modules_stay_removed():
    package_root = Path(__file__).resolve().parents[1] / "memoryforge"

    assert not (package_root / "ui.py").exists()
    assert not (package_root / "core" / "__init__.py").exists()
    assert not (package_root / "chunking" / "__init__.py").exists()


def test_py_typed_marker_is_kept_for_pep_561():
    package_root = Path(__file__).resolve().parents[1] / "memoryforge"

    assert (package_root / "py.typed").exists()


def test_cli_no_longer_exposes_temporary_ui_command():
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if action.dest == "command")

    assert "ui" not in subparsers_action.choices
    assert "rlm-run" in subparsers_action.choices
