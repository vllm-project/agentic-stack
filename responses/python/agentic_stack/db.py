"""
DB engine plumbing.

This module intentionally owns only:
- engine creation (dialect-specific connect args, pooling)
- instrumentation hooks
- connection-level tuning (e.g. SQLite PRAGMAs)
- session helpers

It must not depend on ORM model definitions. The Responses gateway persistence
layer (ResponseStore) owns its own schema and semantics under `agentic_stack.responses_core`.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import AsyncGenerator, Callable, Generator

from loguru import logger
from sqlalchemy import Engine, NullPool, TextClause, event, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlmodel import Session, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from agentic_stack.configs.runtime import RuntimeConfig
from agentic_stack.utils import uuid7_str

SQLITE_BUSY_TIMEOUT_MS = 5_000


def configure_db(runtime_config: RuntimeConfig) -> None:
    create_db_engine(runtime_config)
    create_db_engine_async(runtime_config)


def _apply_sqlite_pragmas(dbapi_connection) -> None:  # type: ignore[no-untyped-def]
    """
    Apply SQLite performance PRAGMAs.

    These are set via the SQLAlchemy engine connect hook so they are consistently applied for
    all SQLite connections (sync and async engines).
    """
    # IMPORTANT:
    # - For `sqlite+aiosqlite`, SQLAlchemy passes an adapted connection object to events.
    #   The underlying sqlite3 connection is available as `driver_connection`.
    # - Executing PRAGMAs through the adapted execute/cursor path can run inside an implicit
    #   transaction and fail for `journal_mode=WAL` ("cannot change into wal mode from within a transaction").
    #   Execute directly on the raw sqlite3 connection instead.
    raw = getattr(dbapi_connection, "driver_connection", None) or dbapi_connection
    # For aiosqlite, the "driver connection" is typically an `aiosqlite.Connection` wrapper;
    # its underlying `sqlite3.Connection` is exposed as `_conn`.
    raw = getattr(raw, "_conn", raw)
    try:
        # Concurrency: allow readers during writes (still single-writer).
        raw.execute("PRAGMA journal_mode=WAL;")
        # In WAL mode, NORMAL is a common durability/perf tradeoff.
        raw.execute("PRAGMA synchronous=NORMAL;")
        # Avoid transient "database is locked" failures under concurrent access.
        raw.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS};")
        # Correctness: enable FK enforcement (off by default in SQLite).
        raw.execute("PRAGMA foreign_keys=ON;")
    except Exception as e:
        # Best-effort: if a PRAGMA fails, keep the gateway running and log once per connection.
        logger.warning(f"Failed to apply SQLite PRAGMAs: {e!r}")


def _create_db_engine(
    *,
    runtime_config: RuntimeConfig,
    engine_create_fn: Callable[..., Engine | AsyncEngine] | None = None,
    echo: bool = False,
) -> Engine:
    if engine_create_fn is None:
        engine_create_fn = create_engine
    kwargs = {"echo": echo}

    db_dialect = runtime_config.db_dialect
    if db_dialect == "sqlite":
        logger.debug("Using SQLite DB.")
        connect_args = {
            "check_same_thread": False,
        }
    elif db_dialect == "postgresql":
        logger.debug("Using PostgreSQL DB.")
        connect_args = {}
        if "asyncpg" in runtime_config.db_path:
            connect_args["prepared_statement_name_func"] = lambda: f"__asyncpg_{uuid7_str()}__"
        kwargs["poolclass"] = NullPool
    else:
        raise ValueError(f'DB type "{db_dialect}" is not supported.')

    engine: AsyncEngine | Engine = engine_create_fn(
        runtime_config.db_path,
        connect_args=connect_args,
        **kwargs,
    )

    if db_dialect == "sqlite":
        base_engine = engine if isinstance(engine, Engine) else engine.sync_engine

        @event.listens_for(base_engine, "connect")
        def _on_connect(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
            _apply_sqlite_pragmas(dbapi_connection)

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    except ImportError:
        logger.warning("Skip sqlalchemy instrumentation.")
    else:
        SQLAlchemyInstrumentor().instrument(
            engine=engine if isinstance(engine, Engine) else engine.sync_engine,
            enable_commenter=True,
            commenter_options={},
        )
    return engine


@lru_cache(maxsize=1)
def create_db_engine(runtime_config: RuntimeConfig) -> Engine:
    engine = _create_db_engine(runtime_config=runtime_config)
    return engine


@lru_cache(maxsize=1)
def create_db_engine_async(runtime_config: RuntimeConfig) -> AsyncEngine:
    engine = _create_db_engine(
        runtime_config=runtime_config,
        engine_create_fn=create_async_engine,
    )
    return engine


def yield_session(runtime_config: RuntimeConfig) -> Generator[Session, None, None]:
    with Session(create_db_engine(runtime_config)) as session:
        yield session


async def yield_async_session(runtime_config: RuntimeConfig) -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(
        create_db_engine_async(runtime_config),
        expire_on_commit=False,
    ) as session:
        yield session


# Sync Session context manager
sync_session = contextmanager(yield_session)
# Async Session context manager
async_session = asynccontextmanager(yield_async_session)


@lru_cache(maxsize=10000)
def cached_text(query: str) -> TextClause:
    return text(query)


def postgres_advisory_lock_key(name: str) -> int:
    """
    Convert a stable string name into an int64 Postgres advisory lock key.

    - Uses sha256 and takes the first 8 bytes as a signed big-endian int64.
    - Keep stable: changing this breaks cross-version lock compatibility.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


@asynccontextmanager
async def postgres_advisory_lock(
    conn: AsyncConnection,
    *,
    name: str,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.1,
) -> AsyncGenerator[None, None]:
    """
    Acquire a Postgres advisory lock for the current session and release it on exit.

    This is intended for one-time init tasks (schema creation, migrations, etc.) where
    multiple gateway instances may race at startup.
    """
    key = postgres_advisory_lock_key(name)
    deadline = time.perf_counter() + timeout_s
    locked = False
    try:
        while True:
            result = await conn.execute(cached_text("SELECT pg_try_advisory_lock(:k)"), {"k": key})
            locked = bool(result.scalar_one())
            if locked:
                break
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for Postgres advisory lock. "
                    "Another instance may be stuck performing one-time initialization."
                )
            await asyncio.sleep(poll_interval_s)
        yield
    finally:
        if locked:
            try:
                await conn.execute(cached_text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            except Exception:
                return
