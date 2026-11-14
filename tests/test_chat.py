"""Tests for document isolation (doc_id filter) and "not found" behaviour.

Runs fully offline: local folder source, SQLite registry, the HashEmbedder, a
fake LLM, and two vector stores - an in-memory Chroma collection and
``VertexVectorStore`` backed by a fake Vertex AI index/endpoint that applies
namespace restricts the way Vector Search does.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import DocumentRegistry
from app.llm import HashEmbedder, LLMError, Message
from app.prompts import NOT_FOUND
from app.services import ContractService
from app.sources import LocalFolderSource
from app.vectorstore import ChromaVectorStore, InMemoryChunkStore, VertexVectorStore

CONTRACTS = Path(__file__).resolve().parent.parent / "contracts"
MSA = "master_services_agreement.txt"
NDA = "mutual_nda.txt"


class FakeLLM:
    """Records calls; answers by citing a clause."""

    def __init__(self, reply: str | None = None) -> None:
        self.calls: list[list[Message]] = []
        self.reply = reply

    def complete(self, messages: list[Message], temperature: float = 0.1) -> str:
        self.calls.append(messages)
        if "standalone" in messages[0]["content"].lower():  # rewrite prompt
            return "What is the governing law of the agreement?"
        if self.reply is not None:
            return self.reply
        return "The answer is in the excerpts (Clause 11.1)."


class CountingEmbedder(HashEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def embed(self, texts, task="document"):  # type: ignore[override]
        self.calls.append(task)
        return super().embed(texts, task)


# --------------------------------------------------------------------------- #
# Fake Vertex AI Vector Search (index + endpoint)
# --------------------------------------------------------------------------- #


@dataclass
class _Neighbor:
    id: str
    distance: float


class FakeVertexIndex:
    def __init__(self) -> None:
        self.points: dict[str, tuple[list[float], dict[str, list[str]]]] = {}

    def upsert_datapoints(self, datapoints: list[Any]) -> None:
        for dp in datapoints:
            restricts = {r.namespace: list(r.allow_list) for r in dp.restricts}
            self.points[dp.datapoint_id] = (list(dp.feature_vector), restricts)

    def remove_datapoints(self, datapoint_ids: list[str]) -> None:
        for i in datapoint_ids:
            self.points.pop(i, None)


class FakeVertexEndpoint:
    def __init__(self, index: FakeVertexIndex) -> None:
        self.index = index
        self.filters: list[Any] = []

    def find_neighbors(self, *, deployed_index_id, queries, num_neighbors, filter):  # noqa: A002
        self.filters.append(filter)
        q = queries[0]
        out = []
        for pid, (vec, restricts) in self.index.points.items():
            if all(set(ns.allow_tokens) & set(restricts.get(ns.name, [])) for ns in filter):
                out.append(_Neighbor(pid, sum(a * b for a, b in zip(q, vec))))  # DOT_PRODUCT_DISTANCE
        return [sorted(out, key=lambda n: -n.distance)[:num_neighbors]]


def _make_store(kind: str):
    if kind == "chroma":
        return ChromaVectorStore(None, collection=f"test_{uuid.uuid4().hex}")
    index = FakeVertexIndex()
    return VertexVectorStore(index, FakeVertexEndpoint(index), "deployed", InMemoryChunkStore())


@pytest.fixture(params=["chroma", "vertex"])
def service(request: pytest.FixtureRequest, tmp_path: Path) -> ContractService:
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    for name in (MSA, NDA):
        shutil.copy(CONTRACTS / name, contracts / name)
    settings = replace(
        Settings(),
        contracts_source="local",
        contracts_dir=contracts,
        registry_backend="sqlite",
        sqlite_path=tmp_path / "docs.db",
        vector_store=request.param,
        similarity_threshold=0.05,
        top_k=5,
    )
    return ContractService(
        settings,
        LocalFolderSource(contracts, settings.supported_extensions),
        DocumentRegistry(settings.sqlite_path),
        _make_store(request.param),
        CountingEmbedder(),
        FakeLLM(),
    )


def _ids(service: ContractService) -> dict[str, str]:
    return {d.file_name: d.doc_id for d in service.list_documents()}


def test_listing_and_indexing_statuses(service: ContractService) -> None:
    docs = service.list_documents()
    assert {d.file_name for d in docs} == {MSA, NDA}
    assert all(d.status == "not_indexed" and len(d.doc_id) == 64 for d in docs)

    doc_id = _ids(service)[MSA]
    info = service.index_document(doc_id)
    assert info.status == "ready" and info.num_chunks and info.num_chunks > 5
    assert service.store.count(doc_id) == info.num_chunks

    # Already ready -> no re-embedding
    embedder = service.embedder
    assert isinstance(embedder, CountingEmbedder)
    calls = len(embedder.calls)
    service.index_document(doc_id)
    assert len(embedder.calls) == calls
    assert embedder.calls == ["document"]  # chunks use the document task type

    # Forced re-index replaces (does not duplicate) chunks
    service.index_document(doc_id, force=True)
    assert service.store.count(doc_id) == info.num_chunks


def test_chat_with_doc_a_never_returns_chunks_from_doc_b(service: ContractService) -> None:
    ids = _ids(service)
    msa_id, nda_id = ids[MSA], ids[NDA]
    service.index_document(msa_id)
    service.index_document(nda_id)
    # Disable the threshold so every query returns a full top_k (an empty result
    # would make the isolation check trivially true).
    service.settings = replace(service.settings, similarity_threshold=-1.0)

    # Questions that lexically match the *other* document best.
    nda_flavoured = [
        "How long do the confidentiality obligations survive for trade secrets?",
        "Which court in New York County has jurisdiction over the Receiving Party?",
        "What does Section 3.2 Duration of Obligations say about Orion and Helix?",
    ]
    msa_flavoured = [
        "What is the late payment interest rate above the Bank of England base rate?",
        "Within how many days must the Customer pay each invoice to the Supplier?",
    ]
    for question in nda_flavoured:
        hits = service.retrieve(msa_id, question)
        assert len(hits) == service.settings.top_k
        assert all(h.metadata["doc_id"] == msa_id for h in hits)
        assert all(h.metadata["file_name"] == MSA for h in hits)
        assert all(h.id.startswith(msa_id) for h in hits)
    for question in msa_flavoured:
        hits = service.retrieve(nda_id, question)
        assert len(hits) == service.settings.top_k
        assert all(h.metadata["doc_id"] == nda_id for h in hits)

    if isinstance(service.store, VertexVectorStore):  # every Vertex query carried the doc_id restrict
        endpoint = service.store.endpoint
        assert endpoint.filters and all(
            [(ns.name, ns.allow_tokens) for ns in f] in ([("doc_id", [msa_id])], [("doc_id", [nda_id])])
            for f in endpoint.filters
        )

    # The context the LLM sees for doc A contains nothing from doc B.
    llm = service.llm
    assert isinstance(llm, FakeLLM)
    service.chat(msa_id, nda_flavoured[0])
    context = "\n".join(m["content"] for m in llm.calls[-1])
    assert "Orion" not in context and "Helix" not in context
    assert "Northwind" in context or "Supplier" in context


def test_not_found_when_nothing_relevant_skips_llm(service: ContractService) -> None:
    doc_id = _ids(service)[NDA]
    service.index_document(doc_id)
    service.settings = replace(service.settings, similarity_threshold=0.99)
    llm = service.llm
    assert isinstance(llm, FakeLLM)
    result = service.chat(doc_id, "What is the recipe for chocolate cake?")
    assert result.answer == NOT_FOUND
    assert result.citations == []
    assert llm.calls == []  # LLM never called


def test_llm_not_found_answer_has_no_citations(service: ContractService) -> None:
    doc_id = _ids(service)[NDA]
    service.index_document(doc_id)
    service._llm = FakeLLM(reply=NOT_FOUND)
    result = service.chat(doc_id, "What are the confidentiality obligations?")
    assert result.answer == NOT_FOUND and result.citations == []


def test_citations_and_followup_rewrite(service: ContractService) -> None:
    doc_id = _ids(service)[MSA]
    service.index_document(doc_id)
    history = [
        {"role": "user", "content": "Which law applies?"},
        {"role": "assistant", "content": "English law (Clause 11.1)."},
    ]
    result = service.chat(doc_id, "And which courts?", history)
    assert result.standalone_question == "What is the governing law of the agreement?"
    assert result.citations
    assert all("11.1" in c.clause_no for c in result.citations)


def test_failed_status_on_bad_file_and_api_error(service: ContractService) -> None:
    folder = service.settings.contracts_dir
    (folder / "empty.txt").write_text("   \n\n  ")
    ids = _ids(service)
    info = service.index_document(ids["empty.txt"])
    assert info.status == "failed" and "No extractable text" in (info.error or "")

    class BrokenEmbedder:
        def embed(self, texts, task="document"):
            raise LLMError("API down")

    service._embedder = BrokenEmbedder()
    info = service.index_document(ids[NDA])
    assert info.status == "failed" and "API down" in (info.error or "")


def test_api_endpoints(service: ContractService) -> None:
    from app.main import app

    app.state.service = service
    with TestClient(app) as client:
        docs = client.get("/documents").json()
        nda = next(d for d in docs if d["file_name"] == NDA)
        assert nda["status"] == "not_indexed"

        r = client.post("/chat", json={"doc_id": nda["doc_id"], "question": "Term?"})
        assert r.status_code == 409  # not indexed yet

        r = client.post(f"/documents/{nda['doc_id']}/index")
        assert r.status_code == 202
        assert client.get(f"/documents/{nda['doc_id']}").json()["status"] == "ready"

        r = client.post("/chat", json={"doc_id": nda["doc_id"], "question": "How long does the Agreement remain in effect?"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"answer", "citations"}
        assert client.post("/documents/unknown/index").status_code == 404
    del app.state.service
