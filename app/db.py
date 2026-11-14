"""Registry of contract documents and their indexing status.

Two interchangeable backends with the same interface (:class:`Registry`):
:class:`FirestoreRegistry` (default, Google Cloud) and :class:`DocumentRegistry`
(SQLite ``documents`` table, for local development and tests).
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

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


class Registry(Protocol):
    """Interface shared by the SQLite and Firestore registries."""

    def upsert_file(self, doc_id: str, file_name: str) -> DocumentRecord: ...
    def get(self, doc_id: str) -> DocumentRecord | None: ...
    def all(self) -> list[DocumentRecord]: ...
    def try_mark_processing(self, doc_id: str) -> bool: ...
    def mark_ready(self, doc_id: str, num_chunks: int) -> None: ...
    def mark_failed(self, doc_id: str, error: str) -> None: ...
    def reset_status(self, doc_id: str) -> None: ...
    def reset_stale_processing(self) -> int: ...
    def delete(self, doc_id: str) -> None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
                (num_chunks, _now(), doc_id),
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


class FirestoreRegistry:
    """Registry stored in a Firestore collection (one document per doc_id)."""

    def __init__(self, project: str | None, database: str, collection: str) -> None:
        from google.cloud import firestore

        self._firestore = firestore
        self._client = firestore.Client(project=project, database=database)
        self._col = self._client.collection(collection)

    @staticmethod
    def _record(doc_id: str, data: dict[str, Any]) -> DocumentRecord:
        return DocumentRecord(
            doc_id=doc_id,
            file_name=data.get("file_name", ""),
            status=data.get("status", "not_indexed"),
            num_chunks=data.get("num_chunks"),
            indexed_at=data.get("indexed_at"),
            error=data.get("error"),
        )

    def upsert_file(self, doc_id: str, file_name: str) -> DocumentRecord:
        """Register a file (status ``not_indexed`` if new) and refresh its name."""
        ref = self._col.document(doc_id)
        snap = ref.get()
        if not snap.exists:
            from google.api_core.exceptions import AlreadyExists

            data = {"file_name": file_name, "status": "not_indexed", "num_chunks": None,
                    "indexed_at": None, "error": None}
            try:
                ref.create(data)
                return self._record(doc_id, data)
            except AlreadyExists:  # created concurrently
                snap = ref.get()
        data = snap.to_dict() or {}
        if data.get("file_name") != file_name:
            ref.update({"file_name": file_name})
            data["file_name"] = file_name
        return self._record(doc_id, data)

    def get(self, doc_id: str) -> DocumentRecord | None:
        snap = self._col.document(doc_id).get()
        return self._record(doc_id, snap.to_dict() or {}) if snap.exists else None

    def all(self) -> list[DocumentRecord]:
        recs = [self._record(s.id, s.to_dict() or {}) for s in self._col.stream()]
        return sorted(recs, key=lambda r: r.file_name)

    def try_mark_processing(self, doc_id: str) -> bool:
        """Atomically claim the document for indexing (Firestore transaction)."""
        ref = self._col.document(doc_id)

        @self._firestore.transactional
        def claim(tx: Any) -> bool:
            snap = ref.get(transaction=tx)
            if not snap.exists or (snap.to_dict() or {}).get("status") in ("processing", "ready"):
                return False
            tx.update(ref, {"status": "processing", "error": None})
            return True

        return claim(self._client.transaction())

    def mark_ready(self, doc_id: str, num_chunks: int) -> None:
        self._col.document(doc_id).update(
            {"status": "ready", "num_chunks": num_chunks, "indexed_at": _now(), "error": None}
        )

    def mark_failed(self, doc_id: str, error: str) -> None:
        self._col.document(doc_id).update({"status": "failed", "num_chunks": None, "error": error[:1000]})

    def reset_status(self, doc_id: str) -> None:
        self._col.document(doc_id).update(
            {"status": "not_indexed", "num_chunks": None, "indexed_at": None, "error": None}
        )

    def reset_stale_processing(self) -> int:
        from google.cloud.firestore_v1.base_query import FieldFilter

        n = 0
        for snap in self._col.where(filter=FieldFilter("status", "==", "processing")).stream():
            snap.reference.update({"status": "not_indexed"})
            n += 1
        return n

    def delete(self, doc_id: str) -> None:
        self._col.document(doc_id).delete()


def build_registry(settings: Any) -> Registry:
    """Create the registry selected by ``REGISTRY_BACKEND``."""
    kind = settings.registry_backend.lower()
    if kind == "sqlite":
        return DocumentRegistry(settings.sqlite_path)
    if kind == "firestore":
        return FirestoreRegistry(
            settings.gcp_project, settings.firestore_database, settings.firestore_documents_collection
        )
    raise ValueError(f"Unknown REGISTRY_BACKEND: {settings.registry_backend}")
