"""The judge cache: one SQLite file instead of one JSON file per judge call.

Tens of thousands of tiny files are slow on Windows and in CI caches; SQLite in WAL mode stays one file,
and several processes (parallel shards sharing a workdir) can read and write it at the same time.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

FILENAME = "judge-cache.sqlite3"


class JudgeCache:
    def __init__(self, directory: Path) -> None:
        self.path = directory / FILENAME
        self.db: Any = None
        self._open()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS judgements (key TEXT PRIMARY KEY, value TEXT NOT NULL, at REAL)"
            )
            db.execute("SELECT 1 FROM judgements LIMIT 1").fetchall()
        except sqlite3.Error:
            # Release the file: on Windows an open handle would block moving the corrupt file aside
            db.close()
            raise
        return db

    def _open(self) -> None:
        try:
            self.db = self._connect()
        except sqlite3.DatabaseError:
            # A corrupt file: move it aside and start an empty cache rather than failing the run
            try:
                self.path.rename(self.path.with_suffix(f".corrupt-{int(time.time())}"))
                self.db = self._connect()
            except (OSError, sqlite3.Error):
                self.db = None  # caching off for this run; grading still works

    def get(self, key: str) -> dict[str, Any] | None:
        if self.db is None:
            return None
        try:
            row = self.db.execute("SELECT value FROM judgements WHERE key = ?", (key,)).fetchone()
            data = json.loads(row[0]) if row else None
        except (sqlite3.Error, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def put(self, key: str, data: dict[str, Any]) -> None:
        if self.db is None:
            return
        # Caching is an optimisation: a locked or read-only cache must not fail the run
        with contextlib.suppress(sqlite3.Error):
            self.db.execute(
                "INSERT OR REPLACE INTO judgements (key, value, at) VALUES (?, ?, ?)",
                (key, json.dumps(data), time.time()),
            )

    def __len__(self) -> int:
        if self.db is None:
            return 0
        return int(self.db.execute("SELECT COUNT(*) FROM judgements").fetchone()[0])

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
