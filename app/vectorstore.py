"""ChromaDB wrapper. One collection for all contracts; every query is filtered by doc_id."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity query."""

    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float  # cosine similarity, 1 = identical


class VectorStore:
    """Persistent Chroma collection using cosine distance and externally computed embeddings."""

    def __init__(self, persist_dir: Path | str | None, collection: str = "contracts") -> None:
        settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def add_chunks(
        self, doc_id: str, file_name: str, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Upsert ``chunks`` (with their embeddings) for ``doc_id``."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        ids = [f"{doc_id}:{c.chunk_index}" for c in chunks]
        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            md: dict[str, Any] = {
                "doc_id": doc_id,
                "file_name": file_name,
                "clause_no": c.clause_no,
                "clause_title": c.clause_title,
                "chunk_index": c.chunk_index,
                "body": c.body,
            }
            if c.page is not None:  # Chroma metadata cannot hold None
                md["page"] = c.page
            metadatas.append(md)
        self._collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=[list(e) for e in embeddings],
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunk(s) for doc %s", len(chunks), doc_id[:12])

    def delete_document(self, doc_id: str) -> None:
        """Delete every chunk belonging to ``doc_id``."""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted existing chunks for doc %s", doc_id[:12])

    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"doc_id": doc_id}, include=[])["ids"])

    def query(self, doc_id: str, embedding: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Nearest chunks to ``embedding`` restricted to ``doc_id``."""
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for cid, text, md, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(RetrievedChunk(id=cid, text=text, metadata=dict(md), similarity=1.0 - float(dist)))
        return out
