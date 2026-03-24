.PHONY: help install lint format fix test pre-commit clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv sync

lint: ## Run ruff linter
	uv run ruff check .

format: ## Run ruff formatter
	uv run ruff format .

fix: ## Run ruff linter with auto-fix
	uv run ruff check --fix .

test: ## Run tests with pytest
	uv run pytest

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ **/__pycache__ .pytest_cache .mypy_cache .ruff_cache dist *.egg-info
