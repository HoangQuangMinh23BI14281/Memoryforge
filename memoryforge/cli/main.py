"""MemoryForge CLI entrypoint."""

from __future__ import annotations

from memoryforge.cli.commands import run_command
from memoryforge.cli.parser import build_parser


def main(argv: list[str] | None = None) -> int:
    return run_command(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
