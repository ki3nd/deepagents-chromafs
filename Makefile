.PHONY: install lint format test build clean

# Install all dev dependencies
install:
	uv sync --extra dev --project python

# Lint with ruff
lint:
	uv run --extra dev --project python ruff check python/deepagents_chromafs python/tests

# Format with ruff
format:
	uv run --extra dev --project python ruff format python/deepagents_chromafs python/tests

# Type-check with ty
typecheck:
	uv run --extra dev --project python ty check python/deepagents_chromafs

# Run unit tests
test:
	uv run --extra dev --project python pytest python/tests/unit_tests -v

# Build distribution packages
build:
	cd python && uv run python -m build

# Remove build artifacts
clean:
	rm -rf python/dist/ python/build/ python/*.egg-info python/.pytest_cache python/__pycache__ python/.ruff_cache
