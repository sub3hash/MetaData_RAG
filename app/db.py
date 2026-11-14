"""SQLite registry of contract documents and their indexing status."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

Status = Literal["not_indexed", "processing", "ready", "failed"]


@dataclass
class DocumentRecord:
    """One row of the ``documents`` table."""

    doc_id: str
    file_name: str
    status: Status
    num_chunks: int | None
    indexed_at: str | None
    error: str | None = None


class DocumentRegistry:
    """Thin, thread-safe wrapper around the ``documents`` SQLite table."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._tx() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id     TEXT PRIMARY KEY,
                    file_name  TEXT NOT NULL,
                    status     TEXT NOT NULL DEFAULT 'not_indexed',
                    num_chunks INTEGER,
                    indexed_at TEXT,
                    error      TEXT
                )
                """
            )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(**{k: row[k] for k in row.keys()})

    def upsert_file(self, doc_id: str, file_name: str) -> DocumentRecord:
        """Register a file (status ``not_indexed`` if new) and refresh its name."""
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO documents (doc_id, file_name) VALUES (?, ?) "
                "ON CONFLICT(doc_id) DO UPDATE SET file_name = excluded.file_name",
                (doc_id, file_name),
            )
        rec = self.get(doc_id)
        assert rec is not None
        return rec

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._tx() as cur:
            row = cur.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row(row) if row else None

    def all(self) -> list[DocumentRecord]:
        with self._tx() as cur:
            rows = cur.execute("SELECT * FROM documents ORDER BY file_name").fetchall()
        return [self._row(r) for r in rows]

    def try_mark_processing(self, doc_id: str) -> bool:
        """Atomically move a document to ``processing``.

        Returns False if it is already ``processing`` or ``ready`` (nothing to do).
        """
        with self._tx() as cur:
            cur.execute(
                "UPDATE documents SET status = 'processing', error = NULL "
                "WHERE doc_id = ? AND status NOT IN ('processing', 'ready')",
                (doc_id,),
            )
            return cur.rowcount == 1

    def mark_ready(self, doc_id: str, num_chunks: int) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE documents SET status = 'ready', num_chunks = ?, indexed_at = ?, error = NULL "
                "WHERE doc_id = ?",
                (num_chunks, datetime.now(timezone.utc).isoformat(timespec="seconds"), doc_id),
            )

    def mark_failed(self, doc_id: str, error: str) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed', num_chunks = NULL, error = ? WHERE doc_id = ?",
                (error[:1000], doc_id),
            )

    def reset_status(self, doc_id: str) -> None:
        """Force a document back to ``not_indexed`` (used for explicit re-index)."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE documents SET status = 'not_indexed', num_chunks = NULL, indexed_at = NULL, "
                "error = NULL WHERE doc_id = ?",
                (doc_id,),
            )

    def reset_stale_processing(self) -> int:
        """On startup, jobs left in ``processing`` by a crash are reset."""
        with self._tx() as cur:
            cur.execute("UPDATE documents SET status = 'not_indexed' WHERE status = 'processing'")
            return cur.rowcount

    def delete(self, doc_id: str) -> None:
        with self._tx() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
