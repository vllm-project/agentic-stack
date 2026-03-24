# Contributing to vLLM Agentic Stack

Thank you for your interest in contributing to the vLLM Agentic Stack. This guide covers
everything you need to get started.

## Getting Started

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/vllm-project/agentic-stack.git
   cd agentic-stack
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (the project's
   package and dependency manager).

3. Install project dependencies:
   ```bash
   uv sync
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development

### Running Tests

```bash
uv run pytest
```

### Linting

```bash
ruff check .
```

To auto-fix issues:

```bash
ruff check --fix .
```

### Formatting

```bash
ruff format .
```

All linting and formatting checks are also run automatically via pre-commit hooks on
each commit.

## Pull Requests

- Branch from `main`.
- Write tests for new functionality.
- Ensure all pre-commit hooks pass before pushing.
- Sign off your commits (`git commit -s`).
- Use the PR template, which includes two required sections:
  - **Summary** -- a concise description of what the PR does and why.
  - **Test Plan** -- how the changes were tested.

## Code Style

Code style is enforced by [ruff](https://docs.astral.sh/ruff/) via pre-commit. Key
settings:

- Maximum line length: 120 characters.
- Target Python version: 3.12+.

Do not worry about manually formatting code -- the pre-commit hooks will handle it.

## Reporting Issues

Use the issue templates provided on the
[GitHub Issues](https://github.com/vllm-project/agentic-stack/issues) page. Choose the
template that best matches your report (bug report, feature request, etc.).

## Code of Conduct

This project follows a Code of Conduct. Please review
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on expected behavior.
