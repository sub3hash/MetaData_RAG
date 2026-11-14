"""Business logic: document discovery, ingestion and document-scoped chat.

Kept separate from the FastAPI routes so it can be unit-tested with fake
embedders/LLMs and temporary storage.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from app.chunking import chunk_document
from app.config import Settings
from app.db import DocumentRecord, DocumentRegistry
from app.extract import ExtractionError, extract_pages
from app.llm import (
    ChatModel,
    Embedder,
    LLMError,
    Message,
    build_chat_model,
    build_embedder,
    generate_answer,
    rewrite_question,
)
from app.prompts import CONTEXT_TEMPLATE, NOT_FOUND
from app.vectorstore import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class DocumentNotFound(Exception):
    """Unknown doc_id (file missing from the contracts folder)."""


class DocumentNotReady(Exception):
    """Chat requested for a document that is not indexed yet."""


@dataclass
class DocumentInfo:
    doc_id: str
    file_name: str
    size: int
    status: str
    num_chunks: int | None
    indexed_at: str | None
    error: str | None


@dataclass
class Citation:
    clause_no: str
    clause_title: str
    snippet: str
    page: int | None = None


@dataclass
class ChatResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    standalone_question: str | None = None


def file_sha256(path: Path) -> str:
    """SHA-256 of the file content (the document id)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ContractService:
    """Facade over registry + vector store + providers."""

    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        store: VectorStore,
        embedder: Embedder | None = None,
        llm: ChatModel | None = None,
        embedder_factory: Callable[[Settings], Embedder] = build_embedder,
        llm_factory: Callable[[Settings], ChatModel] = build_chat_model,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store
        self._embedder = embedder
        self._llm = llm
        self._embedder_factory = embedder_factory
        self._llm_factory = llm_factory
        self._hash_cache: dict[Path, tuple[float, int, str]] = {}
        self._lock = threading.Lock()

    # Providers are created lazily so a missing API key fails a request, not startup.
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._embedder_factory(self.settings)
        return self._embedder

    @property
    def llm(self) -> ChatModel:
        if self._llm is None:
            self._llm = self._llm_factory(self.settings)
        return self._llm

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def _scan(self) -> dict[str, Path]:
        """Map doc_id -> path for every supported file in the contracts folder."""
        folder = self.settings.contracts_dir
        folder.mkdir(parents=True, exist_ok=True)
        found: dict[str, Path] = {}
        with self._lock:
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix.lower() not in self.settings.supported_extensions:
                    continue
                st = path.stat()
                cached = self._hash_cache.get(path)
                if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
                    doc_id = cached[2]
                else:
                    doc_id = file_sha256(path)
                    self._hash_cache[path] = (st.st_mtime, st.st_size, doc_id)
                found[doc_id] = path
        return found

    def list_documents(self) -> list[DocumentInfo]:
        """List contracts with their status; registers new files and prunes stale ones.

        A file whose content changed gets a new doc_id (→ ``not_indexed``); the
        rows and vectors of doc_ids no longer present on disk are removed.
        """
        files = self._scan()
        for doc_id, path in files.items():
            self.registry.upsert_file(doc_id, path.name)
        for rec in self.registry.all():
            if rec.doc_id not in files:
                logger.info("Pruning stale document %s (%s)", rec.doc_id[:12], rec.file_name)
                try:
                    self.store.delete_document(rec.doc_id)
                except Exception:  # pragma: no cover - best effort
                    logger.exception("Could not delete vectors for %s", rec.doc_id[:12])
                self.registry.delete(rec.doc_id)
        out = []
        for doc_id, path in files.items():
            rec = self.registry.get(doc_id)
            assert rec is not None
            out.append(self._info(rec, path))
        return sorted(out, key=lambda d: d.file_name.lower())

    def _info(self, rec: DocumentRecord, path: Path) -> DocumentInfo:
        return DocumentInfo(
            doc_id=rec.doc_id,
            file_name=rec.file_name,
            size=path.stat().st_size if path.exists() else 0,
            status=rec.status,
            num_chunks=rec.num_chunks,
            indexed_at=rec.indexed_at,
            error=rec.error,
        )

    def get_document(self, doc_id: str) -> DocumentInfo:
        files = self._scan()
        if doc_id not in files:
            raise DocumentNotFound(doc_id)
        rec = self.registry.get(doc_id) or self.registry.upsert_file(doc_id, files[doc_id].name)
        return self._info(rec, files[doc_id])

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def prepare_index(self, doc_id: str, force: bool = False) -> bool:
        """Claim a document for indexing.

        Returns True if the caller should run :meth:`run_index`, False if the
        document is already ``ready`` (and not forced) or already ``processing``.
        """
        doc = self.get_document(doc_id)
        if force and doc.status in ("ready", "failed"):
            self.registry.reset_status(doc_id)
        claimed = self.registry.try_mark_processing(doc_id)
        if not claimed:
            logger.info("Doc %s is %s; skipping indexing", doc_id[:12], doc.status)
        return claimed

    def run_index(self, doc_id: str) -> None:
        """Extract → chunk → embed → upsert. Status ends as ``ready`` or ``failed``."""
        files = self._scan()
        path = files.get(doc_id)
        if path is None:
            self.registry.mark_failed(doc_id, "File no longer exists")
            return
        s = self.settings
        try:
            logger.info("[%s] Step 1/5: extracting text from %s", doc_id[:12], path.name)
            pages = extract_pages(path)

            logger.info("[%s] Step 2/5: clause-based chunking", doc_id[:12])
            chunks = chunk_document(
                pages,
                max_tokens=s.max_chunk_tokens,
                min_tokens=s.min_chunk_tokens,
                min_clauses=s.min_clauses,
                fallback_chunk_size=s.fallback_chunk_size,
                fallback_overlap=s.fallback_chunk_overlap,
            )
            if not chunks:
                raise ExtractionError("No chunks produced (empty text?)")

            logger.info("[%s] Step 3/5: embedding %d chunk(s)", doc_id[:12], len(chunks))
            embeddings = self.embedder.embed([c.text for c in chunks])

            logger.info("[%s] Step 4/5: replacing chunks in vector store", doc_id[:12])
            self.store.delete_document(doc_id)
            self.store.add_chunks(doc_id, path.name, chunks, embeddings)

            self.registry.mark_ready(doc_id, len(chunks))
            logger.info("[%s] Step 5/5: ready (%d chunks)", doc_id[:12], len(chunks))
        except (ExtractionError, LLMError, ValueError) as exc:
            logger.error("[%s] Indexing failed: %s", doc_id[:12], exc)
            self.registry.mark_failed(doc_id, str(exc))
        except Exception as exc:  # unexpected: keep the traceback in the log
            logger.exception("[%s] Indexing failed unexpectedly", doc_id[:12])
            self.registry.mark_failed(doc_id, f"{type(exc).__name__}: {exc}")

    def index_document(self, doc_id: str, force: bool = False) -> DocumentInfo:
        """Synchronous convenience wrapper: index unless already ready."""
        if self.prepare_index(doc_id, force=force):
            self.run_index(doc_id)
        return self.get_document(doc_id)

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #

    def retrieve(self, doc_id: str, question: str) -> list[RetrievedChunk]:
        """Top-k chunks of ``doc_id`` above the similarity threshold."""
        [vector] = self.embedder.embed([question])
        results = self.store.query(doc_id, vector, top_k=self.settings.top_k)
        threshold = self.settings.similarity_threshold
        kept = [r for r in results if r.similarity >= threshold]
        logger.info(
            "Retrieved %d chunk(s) for doc %s, %d above threshold %.2f: %s",
            len(results), doc_id[:12], len(kept), threshold,
            [(r.metadata.get("clause_no"), round(r.similarity, 3)) for r in results],
        )
        return kept

    def chat(self, doc_id: str, question: str, history: Sequence[Message] = ()) -> ChatResult:
        """Answer ``question`` using only chunks of ``doc_id``."""
        doc = self.get_document(doc_id)
        if doc.status != "ready":
            raise DocumentNotReady(f"Document status is '{doc.status}'")

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ][-2 * self.settings.max_history_turns :]

        standalone = rewrite_question(self.llm, question, history) if history else question
        hits = self.retrieve(doc_id, standalone)
        if not hits:
            return ChatResult(answer=NOT_FOUND, citations=[], standalone_question=standalone)

        excerpts = "\n\n---\n\n".join(h.text for h in hits)
        context = CONTEXT_TEMPLATE.format(file_name=doc.file_name, excerpts=excerpts)
        answer = generate_answer(self.llm, context, standalone, history, self.settings.llm_temperature)

        if answer.strip().strip('"').strip() == NOT_FOUND:
            return ChatResult(answer=NOT_FOUND, citations=[], standalone_question=standalone)
        return ChatResult(answer=answer, citations=_citations(answer, hits), standalone_question=standalone)


def _mentions(answer: str, clause_no: str) -> bool:
    for part in clause_no.split(", "):
        if re.search(rf"(?<![\w.]){re.escape(part)}(?![\w]|\.\d)", answer):
            return True
    return False


def _citations(answer: str, hits: Sequence[RetrievedChunk]) -> list[Citation]:
    """Citations for the chunks the answer refers to (all retrieved ones if none are named)."""
    def cite(h: RetrievedChunk) -> Citation:
        md: dict[str, Any] = h.metadata
        body = str(md.get("body") or h.text)
        snippet = body if len(body) <= 400 else body[:400].rsplit(" ", 1)[0] + " …"
        return Citation(
            clause_no=str(md.get("clause_no", "")),
            clause_title=str(md.get("clause_title", "")),
            snippet=snippet,
            page=md.get("page"),
        )

    named = [h for h in hits if _mentions(answer, str(h.metadata.get("clause_no", "")))]
    chosen = named or list(hits)
    seen: set[str] = set()
    out: list[Citation] = []
    for h in chosen:
        if h.id not in seen:
            seen.add(h.id)
            out.append(cite(h))
    return out
