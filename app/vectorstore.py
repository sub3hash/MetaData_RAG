"""Vector stores. Every query is restricted to a single ``doc_id``.

* :class:`VertexVectorStore` (default): vectors in Vertex AI Vector Search with a
  ``doc_id`` *restrict* (namespace filter); chunk text + metadata in Firestore,
  because Vector Search only stores vectors.
* :class:`ChromaVectorStore`: local ChromaDB collection (development / tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by a similarity query."""

    id: str
    text: str
    metadata: dict[str, Any]
    similarity: float  # cosine similarity, 1 = identical


class VectorStore(Protocol):
    """Interface shared by all vector stores."""

    def add_chunks(
        self, doc_id: str, file_name: str, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None: ...
    def delete_document(self, doc_id: str) -> None: ...
    def count(self, doc_id: str) -> int: ...
    def query(self, doc_id: str, embedding: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]: ...


def chunk_id(doc_id: str, chunk: Chunk) -> str:
    return f"{doc_id}_{chunk.chunk_index}"


def chunk_metadata(doc_id: str, file_name: str, c: Chunk) -> dict[str, Any]:
    """Metadata stored for every chunk (``page`` only when known)."""
    md: dict[str, Any] = {
        "doc_id": doc_id,
        "file_name": file_name,
        "clause_no": c.clause_no,
        "clause_title": c.clause_title,
        "chunk_index": c.chunk_index,
        "body": c.body,
    }
    if c.page is not None:
        md["page"] = c.page
    return md


def _check_lengths(chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")


# --------------------------------------------------------------------------- #
# ChromaDB (local)
# --------------------------------------------------------------------------- #


class ChromaVectorStore:
    """Persistent (or in-memory) Chroma collection using cosine distance."""

    def __init__(self, persist_dir: Path | str | None, collection: str = "contracts") -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

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
        _check_lengths(chunks, embeddings)
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk_id(doc_id, c) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[list(e) for e in embeddings],
            metadatas=[chunk_metadata(doc_id, file_name, c) for c in chunks],
        )
        logger.info("Upserted %d chunk(s) for doc %s", len(chunks), doc_id[:12])

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted existing chunks for doc %s", doc_id[:12])

    def count(self, doc_id: str) -> int:
        return len(self._collection.get(where={"doc_id": doc_id}, include=[])["ids"])

    def query(self, doc_id: str, embedding: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(id=cid, text=text, metadata=dict(md), similarity=1.0 - float(dist))
            for cid, text, md, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]


# --------------------------------------------------------------------------- #
# Chunk text/metadata store used alongside Vertex AI Vector Search
# --------------------------------------------------------------------------- #


@dataclass
class StoredChunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


class ChunkStore(Protocol):
    def put_many(self, chunks: Sequence[StoredChunk]) -> None: ...
    def get_many(self, ids: Sequence[str]) -> dict[str, StoredChunk]: ...
    def ids_for(self, doc_id: str) -> list[str]: ...
    def delete_many(self, ids: Sequence[str]) -> None: ...


class InMemoryChunkStore:
    """Dict-backed chunk store (tests)."""

    def __init__(self) -> None:
        self.data: dict[str, StoredChunk] = {}

    def put_many(self, chunks: Sequence[StoredChunk]) -> None:
        self.data.update({c.id: c for c in chunks})

    def get_many(self, ids: Sequence[str]) -> dict[str, StoredChunk]:
        return {i: self.data[i] for i in ids if i in self.data}

    def ids_for(self, doc_id: str) -> list[str]:
        return [i for i, c in self.data.items() if c.doc_id == doc_id]

    def delete_many(self, ids: Sequence[str]) -> None:
        for i in ids:
            self.data.pop(i, None)


class FirestoreChunkStore:
    """Chunk text + metadata in a Firestore collection (document id = datapoint id)."""

    _BATCH = 400  # Firestore allows up to 500 writes per batch

    def __init__(self, project: str | None, database: str, collection: str) -> None:
        from google.cloud import firestore

        self._client = firestore.Client(project=project, database=database)
        self._col = self._client.collection(collection)

    def put_many(self, chunks: Sequence[StoredChunk]) -> None:
        for start in range(0, len(chunks), self._BATCH):
            batch = self._client.batch()
            for c in chunks[start : start + self._BATCH]:
                batch.set(self._col.document(c.id), {"doc_id": c.doc_id, "text": c.text, "metadata": c.metadata})
            batch.commit()

    def get_many(self, ids: Sequence[str]) -> dict[str, StoredChunk]:
        if not ids:
            return {}
        out: dict[str, StoredChunk] = {}
        for snap in self._client.get_all([self._col.document(i) for i in ids]):
            if snap.exists:
                d = snap.to_dict() or {}
                out[snap.id] = StoredChunk(snap.id, d.get("doc_id", ""), d.get("text", ""), d.get("metadata", {}))
        return out

    def ids_for(self, doc_id: str) -> list[str]:
        from google.cloud.firestore_v1.field_path import FieldPath
        from google.cloud.firestore_v1.base_query import FieldFilter

        # Project to the document name only, so chunk text is not downloaded.
        query = self._col.where(filter=FieldFilter("doc_id", "==", doc_id)).select([FieldPath.document_id()])
        return [snap.id for snap in query.stream()]

    def delete_many(self, ids: Sequence[str]) -> None:
        ids = list(ids)
        for start in range(0, len(ids), self._BATCH):
            batch = self._client.batch()
            for i in ids[start : start + self._BATCH]:
                batch.delete(self._col.document(i))
            batch.commit()


# --------------------------------------------------------------------------- #
# Vertex AI Vector Search
# --------------------------------------------------------------------------- #


class VertexVectorStore:
    """Vertex AI Vector Search (streaming-update index) + a :class:`ChunkStore`.

    The index must use ``DOT_PRODUCT_DISTANCE`` on unit-normalised vectors (see
    ``scripts/setup_gcp.py``), so the returned "distance" is the cosine similarity.
    Each datapoint carries a ``doc_id`` restrict and every query filters on it.
    """

    NAMESPACE = "doc_id"
    _UPSERT_BATCH = 100
    _REMOVE_BATCH = 500

    def __init__(self, index: Any, endpoint: Any, deployed_index_id: str, chunk_store: ChunkStore) -> None:
        self.index = index
        self.endpoint = endpoint
        self.deployed_index_id = deployed_index_id
        self.chunks = chunk_store

    @classmethod
    def from_ids(
        cls,
        project: str | None,
        location: str,
        index_id: str,
        endpoint_id: str,
        deployed_index_id: str,
        chunk_store: ChunkStore,
    ) -> "VertexVectorStore":
        from google.cloud import aiplatform

        aiplatform.init(project=project, location=location)
        index = aiplatform.MatchingEngineIndex(index_name=index_id)
        endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_id)
        return cls(index, endpoint, deployed_index_id, chunk_store)

    def add_chunks(
        self, doc_id: str, file_name: str, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        from google.cloud.aiplatform_v1.types import IndexDatapoint

        _check_lengths(chunks, embeddings)
        if not chunks:
            return
        # Text first: every id that can exist in the index is then known to the chunk store.
        self.chunks.put_many(
            [StoredChunk(chunk_id(doc_id, c), doc_id, c.text, chunk_metadata(doc_id, file_name, c)) for c in chunks]
        )
        datapoints = [
            IndexDatapoint(
                datapoint_id=chunk_id(doc_id, c),
                feature_vector=list(e),
                restricts=[IndexDatapoint.Restriction(namespace=self.NAMESPACE, allow_list=[doc_id])],
            )
            for c, e in zip(chunks, embeddings)
        ]
        for start in range(0, len(datapoints), self._UPSERT_BATCH):
            self.index.upsert_datapoints(datapoints=datapoints[start : start + self._UPSERT_BATCH])
        logger.info("Upserted %d datapoint(s) to Vertex AI Vector Search for doc %s", len(datapoints), doc_id[:12])

    def delete_document(self, doc_id: str) -> None:
        ids = self.chunks.ids_for(doc_id)
        for start in range(0, len(ids), self._REMOVE_BATCH):
            self.index.remove_datapoints(datapoint_ids=ids[start : start + self._REMOVE_BATCH])
        self.chunks.delete_many(ids)
        logger.info("Deleted %d existing chunk(s) for doc %s", len(ids), doc_id[:12])

    def count(self, doc_id: str) -> int:
        return len(self.chunks.ids_for(doc_id))

    def query(self, doc_id: str, embedding: Sequence[float], top_k: int = 5) -> list[RetrievedChunk]:
        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

        neighbors = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[list(embedding)],
            num_neighbors=top_k,
            filter=[Namespace(self.NAMESPACE, [doc_id], [])],
        )
        matches = neighbors[0] if neighbors else []
        stored = self.chunks.get_many([m.id for m in matches])
        out: list[RetrievedChunk] = []
        for m in matches:
            sc = stored.get(m.id)
            if sc is None or sc.doc_id != doc_id:  # stale datapoint or (defensively) another document
                continue
            out.append(RetrievedChunk(id=m.id, text=sc.text, metadata=dict(sc.metadata), similarity=float(m.distance)))
        return out


def build_vector_store(settings: Any) -> VectorStore:
    """Create the vector store selected by ``VECTOR_STORE``."""
    kind = settings.vector_store.lower()
    if kind == "chroma":
        return ChromaVectorStore(settings.chroma_dir, settings.chroma_collection)
    if kind == "vertex":
        missing = [
            n for n in ("vertex_index_id", "vertex_index_endpoint_id", "vertex_deployed_index_id")
            if not getattr(settings, n)
        ]
        if missing:
            raise ValueError(f"VECTOR_STORE=vertex requires {', '.join(m.upper() for m in missing)}")
        chunk_store = FirestoreChunkStore(
            settings.gcp_project, settings.firestore_database, settings.firestore_chunks_collection
        )
        return VertexVectorStore.from_ids(
            settings.gcp_project,
            settings.gcp_location,
            settings.vertex_index_id,
            settings.vertex_index_endpoint_id,
            settings.vertex_deployed_index_id,
            chunk_store,
        )
    raise ValueError(f"Unknown VECTOR_STORE: {settings.vector_store}")
