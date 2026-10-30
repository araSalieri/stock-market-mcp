"""The server's one SQLite file (ADR-0003). Each store keeps its own table in it."""

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager

DEFAULT_DB_PATH = "stock-market-mcp.db"


@contextmanager
def transaction(path: str | os.PathLike, schema: str) -> Iterator[sqlite3.Connection]:
    """Open the file (created if missing), ensure `schema`, then commit on exit or roll back on error.

    Any database error is raised as a ValueError naming the path, so a broken file fails loudly.
    """
    try:
        # timeout: wait for another writer's lock instead of failing at once.
        with closing(sqlite3.connect(path, timeout=10)) as conn, conn:
            conn.execute(schema)
            yield conn
    except sqlite3.Error as exc:
        raise ValueError(f"Database {path}: {exc}") from exc
