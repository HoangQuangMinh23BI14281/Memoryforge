.PHONY: help fmt check lint typecheck test test-python build develop benchmark clean

PYTHON ?= python3
PYTHONPATH ?= .
COVERAGE_MIN ?= 77

help:
	@echo "MemoryForge development commands"
	@echo "  make fmt          Format Python code"
	@echo "  make check        Run lint + typecheck + tests + coverage gate"
	@echo "  make test         Run all tests"
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

lint:
	uv run --with ruff ruff check memoryforge tests benchmarks

typecheck:
	env PYTHONDONTWRITEBYTECODE=1 uv run --with mypy mypy memoryforge

test: check

test-python:
	env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$(PYTHONPATH) uv run --with pytest --with pytest-cov pytest \
		--cov=memoryforge --cov-report=term-missing --cov-fail-under=$(COVERAGE_MIN)

build:
	uv build

develop:
	uv pip install -e .

benchmark:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) benchmarks/synthetic_test.py

clean:
	rm -rf dist build .pytest_cache **/__pycache__ *.egg-info memoryforge.egg-info
