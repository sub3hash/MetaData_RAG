"""Offline tests for the Google Cloud adapters (Gemini client wiring, embeddings, chat)."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.llm import GeminiChatModel, GeminiEmbedder, LLMError, build_genai_client


class FakeModels:
    def __init__(self, fail_batches: bool = False, text: str = "Answer (Clause 1).") -> None:
        self.embed_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.fail_batches = fail_batches
        self.text = text

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        self.embed_calls.append({"model": model, "contents": contents, "config": config})
        if self.fail_batches and len(contents) > 1:
            raise RuntimeError("only one input per request")
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0] + [0.0] * 6) for _ in contents])

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text=self.text, candidates=[SimpleNamespace(finish_reason="STOP")])


def _client(models: FakeModels) -> Any:
    return SimpleNamespace(models=models)


def test_gemini_embedder_task_types_batching_and_normalisation() -> None:
    models = FakeModels()
    emb = GeminiEmbedder(_client(models), "gemini-embedding-001", dim=8, batch_size=2)
    vecs = emb.embed(["a", "b", "c"], task="document")
    assert len(vecs) == 3
    assert all(math.isclose(sum(v * v for v in vec), 1.0) for vec in vecs)
    assert [len(c["contents"]) for c in models.embed_calls] == [2, 1]
    assert models.embed_calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert models.embed_calls[0]["config"].output_dimensionality == 8

    emb.embed(["question"], task="query")
    assert models.embed_calls[-1]["config"].task_type == "RETRIEVAL_QUERY"


def test_gemini_embedder_falls_back_to_single_requests() -> None:
    models = FakeModels(fail_batches=True)
    vecs = GeminiEmbedder(_client(models), "m", dim=8, batch_size=10).embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert [len(c["contents"]) for c in models.embed_calls] == [3, 1, 1, 1]


def test_gemini_chat_maps_system_and_roles() -> None:
    models = FakeModels()
    llm = GeminiChatModel(_client(models), "gemini-2.5-flash")
    answer = llm.complete(
        [
            {"role": "system", "content": "RULES"},
            {"role": "system", "content": "CONTEXT"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        temperature=0.1,
    )
    assert answer == "Answer (Clause 1)."
    call = models.generate_calls[0]
    assert call["config"].system_instruction == "RULES\n\nCONTEXT"
    assert call["config"].temperature == 0.1
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][-1].parts[0].text == "q2"


def test_gemini_chat_empty_response_raises() -> None:
    llm = GeminiChatModel(_client(FakeModels(text="")), "m")
    with pytest.raises(LLMError):
        llm.complete([{"role": "user", "content": "q"}])


def test_build_genai_client_modes() -> None:
    base = replace(Settings(), google_api_key=None, use_vertexai=False, gcp_project=None)
    with pytest.raises(LLMError):
        build_genai_client(base)
    with pytest.raises(LLMError):
        build_genai_client(replace(base, use_vertexai=True))

    api = build_genai_client(replace(base, google_api_key="test-key"))
    assert api.vertexai is False
    express = build_genai_client(replace(base, use_vertexai=True, google_api_key="test-key"))
    assert express.vertexai is True
