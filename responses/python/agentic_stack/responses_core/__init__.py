"""Internal implementation core for Responses API parity.

This package is intentionally separate from `agentic_stack.types`:
- `agentic_stack.types` defines wire-contract Pydantic models and should remain importable without
  pulling in orchestration/state machines.
- `agentic_stack.responses_core` owns internal normalization and contract-composition logic.
"""
