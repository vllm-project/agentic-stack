# AGENTS.md

Instructions for AI coding agents working on this repository.

## Project Overview

This is a Python project under the `vllm-project` GitHub organization. It uses
[uv](https://docs.astral.sh/uv/) for dependency management and virtual environment
handling.

## Setup

Install all dependencies (including dev dependencies):

```bash
uv sync
```

## Testing

Run the full test suite with:

```bash
uv run pytest
```

## Linting and Formatting

Linting and formatting are enforced via pre-commit using ruff.

```bash
ruff check .          # lint
ruff check --fix .    # lint with auto-fix
ruff format .         # format
```

To run all pre-commit hooks manually:

```bash
pre-commit run --all-files
```

## Code Style

- Maximum line length: 120 characters.
- Target Python version: 3.12+.
- Type hints are encouraged on all public functions and methods.
- Style is enforced automatically by ruff; do not override or disable rules without
  discussion.

## Commits

- Always sign off commits with the `-s` flag (`git commit -s`).
- Use conventional commit prefixes:
  - `feat:` -- new feature
  - `fix:` -- bug fix
  - `ci:` -- CI/CD changes
  - `chore:` -- maintenance tasks (deps, config)
  - `docs:` -- documentation only

## Pull Requests

- Target the `main` branch.
- Include two sections in the PR description:
  - **Summary** -- what the PR does and why.
  - **Test Plan** -- how the changes were verified.
- Ensure all pre-commit hooks pass before opening the PR.
