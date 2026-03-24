from __future__ import annotations

from pathlib import Path

import pytest

from agentic_stack.configs.builders import build_runtime_config_for_standalone
from agentic_stack.configs.sources import EnvSource
from agentic_stack.responses_core.store import DBResponseStore


def _install_store_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: int,
) -> None:
    import agentic_stack.responses_core.store as store_mod

    runtime_config = build_runtime_config_for_standalone(
        env=EnvSource(
            environ={
                "VR_LLM_API_BASE": "http://mock/v1",
                "VR_WORKERS": str(workers),
                "VR_DB_PATH": "sqlite+aiosqlite:///ignored.db",
            }
        )
    )
    monkeypatch.setattr(store_mod, "_STORE_RUNTIME_CONFIG", runtime_config)


@pytest.mark.anyio
async def test_sqlite_multi_worker_schema_init_requires_supervisor(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    _install_store_runtime_config(monkeypatch, workers=2)
    monkeypatch.delenv("VR_DB_SCHEMA_READY", raising=False)

    with pytest.raises(
        RuntimeError, match="SQLite schema initialization is not multi-worker safe"
    ):
        await store.ensure_schema()


@pytest.mark.anyio
async def test_schema_ready_env_skips_init(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "state.db"
    store = DBResponseStore.from_db_url(db_url=f"sqlite+aiosqlite:///{db_path}")

    monkeypatch.setenv("VR_DB_SCHEMA_READY", "1")

    # Should be a no-op and not create any files/tables.
    await store.ensure_schema()
    assert not db_path.exists()

    await store.aclose()
