"""FastAPI routes for the contract chat backend.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import AsyncIterator, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.config import get_settings, setup_logging
from app.db import build_registry
from app.llm import LLMError
from app.services import ContractService, DocumentNotFound, DocumentNotReady
from app.sources import build_source
from app.vectorstore import build_vector_store

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class DocumentOut(BaseModel):
    doc_id: str
    file_name: str
    size: int
    status: Literal["not_indexed", "processing", "ready", "failed"]
    num_chunks: int | None = None
    indexed_at: str | None = None
    error: str | None = None


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list)


class CitationOut(BaseModel):
    clause_no: str
    clause_title: str
    snippet: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    standalone_question: str | None = None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def build_service() -> ContractService:
    """Wire the service from settings (GCS/Firestore/Vertex AI by default)."""
    s = get_settings()
    logger.info(
        "Backends: source=%s registry=%s vectors=%s embeddings=%s/%s llm=%s/%s (vertexai=%s)",
        s.contracts_source, s.registry_backend, s.vector_store, s.embedding_provider, s.embedding_model,
        s.llm_provider, s.llm_model, s.use_vertexai,
    )
    registry = build_registry(s)
    reset = registry.reset_stale_processing()
    if reset:
        logger.warning("Reset %d document(s) left in 'processing' by a previous run", reset)
    return ContractService(s, build_source(s), registry, build_vector_store(s))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(get_settings().log_level)
    if not hasattr(app.state, "service"):
        app.state.service = build_service()
    logger.info("Backend ready")
    yield


app = FastAPI(title="Contract Document Chat", version="1.0.0", lifespan=lifespan)


def get_service(request: Request) -> ContractService:
    return request.app.state.service


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(svc: ContractService = Depends(get_service)) -> list[DocumentOut]:
    """List contract files with their indexing status."""
    return [DocumentOut(**asdict(d)) for d in svc.list_documents()]


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str, svc: ContractService = Depends(get_service)) -> DocumentOut:
    try:
        return DocumentOut(**asdict(svc.get_document(doc_id)))
    except DocumentNotFound:
        raise HTTPException(404, "Document not found")


@app.post("/documents/{doc_id}/index", response_model=DocumentOut, status_code=202)
def index_document(
    doc_id: str,
    background: BackgroundTasks,
    force: bool = Query(False, description="Re-index even if already ready"),
    wait: bool = Query(False, description="Index synchronously and return the final status"),
    svc: ContractService = Depends(get_service),
) -> DocumentOut:
    """Embed a document unless it is already ``ready``.

    By default indexing runs in the background and the response has status
    ``processing``; poll ``GET /documents/{doc_id}`` until ``ready``/``failed``.
    """
    try:
        if svc.prepare_index(doc_id, force=force):
            if wait:
                svc.run_index(doc_id)
            else:
                background.add_task(svc.run_index, doc_id)
        return DocumentOut(**asdict(svc.get_document(doc_id)))
    except DocumentNotFound:
        raise HTTPException(404, "Document not found")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, svc: ContractService = Depends(get_service)) -> ChatResponse:
    """Answer a question using only the selected contract."""
    try:
        result = svc.chat(req.doc_id, req.question, [m.model_dump() for m in req.history])
    except DocumentNotFound:
        raise HTTPException(404, "Document not found")
    except DocumentNotReady as exc:
        raise HTTPException(409, str(exc))
    except LLMError as exc:
        logger.error("Chat failed: %s", exc)
        raise HTTPException(502, str(exc))
    return ChatResponse(
        answer=result.answer,
        citations=[CitationOut(**asdict(c)) for c in result.citations],
        standalone_question=result.standalone_question,
    )
