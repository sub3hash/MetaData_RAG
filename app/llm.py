"""Gemini embeddings and chat (Gemini API key or Vertex AI), query rewriting and answering."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Any, Literal, Protocol, Sequence

from app.config import Settings
from app.prompts import ANSWER_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT, REWRITE_USER_TEMPLATE

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an embedding / LLM provider call fails."""


Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": ...}
Task = Literal["document", "query"]


def _normalise(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# --------------------------------------------------------------------------- #
# Gemini client (Gemini API key or Vertex AI)
# --------------------------------------------------------------------------- #


def build_genai_client(settings: Settings) -> Any:
    """Create a ``google.genai`` client.

    * ``GOOGLE_GENAI_USE_VERTEXAI=false`` -> Gemini API with ``GOOGLE_API_KEY``.
    * ``GOOGLE_GENAI_USE_VERTEXAI=true`` + ``GCP_PROJECT`` -> Vertex AI with
      Application Default Credentials (service account / ``gcloud auth``).
    * ``GOOGLE_GENAI_USE_VERTEXAI=true`` without project + ``GOOGLE_API_KEY`` ->
      Vertex AI express mode.
    """
    from google import genai
    from google.genai import types

    http_options = types.HttpOptions(
        timeout=120_000, retry_options=types.HttpRetryOptions(attempts=4, initial_delay=1.0, max_delay=20.0)
    )
    if settings.use_vertexai:
        if settings.gcp_project:
            return genai.Client(
                vertexai=True, project=settings.gcp_project, location=settings.gcp_location, http_options=http_options
            )
        if settings.google_api_key:
            return genai.Client(vertexai=True, api_key=settings.google_api_key, http_options=http_options)
        raise LLMError("Vertex AI needs GCP_PROJECT (with ADC credentials) or GOOGLE_API_KEY (express mode)")
    if not settings.google_api_key:
        raise LLMError("GOOGLE_API_KEY is not set (or set GOOGLE_GENAI_USE_VERTEXAI=true)")
    return genai.Client(api_key=settings.google_api_key, http_options=http_options)


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    """Turns texts into unit-length vectors. ``task`` separates documents from queries."""

    def embed(self, texts: Sequence[str], task: Task = "document") -> list[list[float]]: ...


class GeminiEmbedder:
    """Gemini embeddings (``gemini-embedding-001`` by default), batched and L2-normalised.

    Uses task types ``RETRIEVAL_DOCUMENT`` for chunks and ``RETRIEVAL_QUERY`` for
    questions. Vectors are normalised because a reduced ``output_dimensionality``
    is not unit-length by default.
    """

    def __init__(self, client: Any, model: str, dim: int, batch_size: int = 50) -> None:
        self._client = client
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)

    def _call(self, batch: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        resp = self._client.models.embed_content(
            model=self.model,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
        )
        vectors = [_normalise(e.values or []) for e in (resp.embeddings or [])]
        if len(vectors) != len(batch):
            raise LLMError(f"Expected {len(batch)} embeddings, got {len(vectors)}")
        return vectors

    def embed(self, texts: Sequence[str], task: Task = "document") -> list[list[float]]:
        task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY"
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                out.extend(self._call(batch, task_type))
            except Exception as exc:
                if len(batch) == 1:
                    raise LLMError(f"Embedding request failed: {exc}") from exc
                # Some endpoints accept only one input per request: retry one by one.
                logger.warning("Batch embedding failed (%s); retrying one text per request", exc)
                try:
                    for text in batch:
                        out.extend(self._call([text], task_type))
                except Exception as exc2:
                    raise LLMError(f"Embedding request failed: {exc2}") from exc2
            logger.info("Embedded %s batch %d-%d", task, start, start + len(batch) - 1)
        return out


class HashEmbedder:
    """Offline, deterministic bag-of-words hashing embedder.

    Captures lexical overlap only – meant for tests and offline demos, not production.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")
    _STOPWORDS = frozenset(
        "a an and are as at be by do does for from has have how in is it its of on or shall that the "
        "this to was what when which who will with within any all such under".split()
    )

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._TOKEN_RE.findall(text.lower()):
            if tok in self._STOPWORDS:
                continue
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalise(vec)

    def embed(self, texts: Sequence[str], task: Task = "document") -> list[list[float]]:
        return [self._vector(t) for t in texts]


def build_embedder(settings: Settings) -> Embedder:
    """Create the embedder selected by ``EMBEDDING_PROVIDER``."""
    provider = settings.embedding_provider.lower()
    if provider == "gemini":
        return GeminiEmbedder(
            build_genai_client(settings), settings.embedding_model, settings.embedding_dim, settings.embed_batch_size
        )
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")


# --------------------------------------------------------------------------- #
# Chat LLM
# --------------------------------------------------------------------------- #


class ChatModel(Protocol):
    """Completes a list of chat messages (system / user / assistant)."""

    def complete(self, messages: list[Message], temperature: float = 0.1) -> str: ...


class GeminiChatModel:
    """Gemini chat (``gemini-2.5-flash`` by default).

    System messages become the ``system_instruction``; ``assistant`` turns map to
    Gemini's ``model`` role.
    """

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    def complete(self, messages: list[Message], temperature: float = 0.1) -> str:
        from google.genai import types

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
            if m["role"] != "system"
        ]
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system or None, temperature=temperature),
            )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        text = (resp.text or "").strip()
        if not text:
            reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
            raise LLMError(f"LLM returned an empty response (finish_reason={reason})")
        return text


def build_chat_model(settings: Settings) -> ChatModel:
    """Create the chat model selected by ``LLM_PROVIDER``."""
    if settings.llm_provider.lower() == "gemini":
        return GeminiChatModel(build_genai_client(settings), settings.llm_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


# --------------------------------------------------------------------------- #
# RAG helpers
# --------------------------------------------------------------------------- #


def _format_history(history: Sequence[Message]) -> str:
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)


def rewrite_question(llm: ChatModel, question: str, history: Sequence[Message]) -> str:
    """Rewrite a follow-up into a standalone question (no-op without history)."""
    if not history:
        return question
    messages: list[Message] = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": REWRITE_USER_TEMPLATE.format(history=_format_history(history), question=question)},
    ]
    rewritten = llm.complete(messages, temperature=0.0).strip().strip('"')
    logger.info("Rewrote question: %r -> %r", question, rewritten)
    return rewritten or question


def generate_answer(
    llm: ChatModel, context: str, question: str, history: Sequence[Message], temperature: float = 0.1
) -> str:
    """Answer ``question`` grounded in ``context`` (retrieved clauses)."""
    messages: list[Message] = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": context})
    messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    messages.append({"role": "user", "content": question})
    return llm.complete(messages, temperature=temperature)
