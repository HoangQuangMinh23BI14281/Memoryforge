.PHONY: help fmt check lint typecheck test test-python test-real-subagents build develop benchmark clean

PYTHON ?= python3
PYTHONPATH ?= .
COVERAGE_MIN ?= 77
REAL_SUBAGENT_MODEL ?= gpt-5.4

help:
	@echo "MemoryForge development commands"
	@echo "  make fmt          Format Python code"
	@echo "  make check        Run lint + typecheck + tests + coverage gate"
	@echo "  make test         Run all required tests"
	@echo "  make test-real-subagents  Run real Codex CLI sub-agent tests"
	@echo "  make build        Build pure-Python wheel"
	@echo "  make develop      Install editable package"
	@echo "  make benchmark    Run benchmark smoke suite"
	@echo "  make clean        Remove generated artifacts"

fmt:
	uv run --with ruff ruff format memoryforge tests

check:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test-python

test:
	$(MAKE) check

test-python:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) uv run --with pytest --with pytest-cov pytest \
		--cov=memoryforge --cov-report=term-missing --cov-fail-under=$(COVERAGE_MIN)

test-real-subagents:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) MEMORYFORGE_REAL_SUBAGENT=1 MEMORYFORGE_REAL_PROJECT_ROOT=$$(pwd) MEMORYFORGE_SUBAGENT_RUNNER=codex MEMORYFORGE_MODEL=$(REAL_SUBAGENT_MODEL) uv run --with pytest pytest tests/test_real_subagents.py -vv

build:
	uv build

develop:
	uv pip install -e .

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/synthetic_test.py

clean:
	rm -rf dist build .pytest_cache **/__pycache__ *.egg-info memoryforge.egg-info