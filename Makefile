.PHONY: install lint format test build clean

# Install all dev dependencies
install:
	uv sync --extra dev

# Lint with ruff
lint:
	uv run --extra dev ruff check deepagents_chromafs tests

# Format with ruff
format:
	uv run --extra dev ruff format deepagents_chromafs tests

# Type-check with ty
typecheck:
	uv run --extra dev ty check deepagents_chromafs

# Run unit tests
test:
	uv run --extra dev pytest tests/unit_tests -v

# Build distribution packages
build:
	uv run python -m build

# Remove build artifacts
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache __pycache__ .ruff_cache
