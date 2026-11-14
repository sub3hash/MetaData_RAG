"""Embedding and chat-completion providers, query rewriting and answer generation."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol, Sequence

from app.config import Settings
from app.prompts import ANSWER_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT, REWRITE_USER_TEMPLATE

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an embedding / LLM provider call fails."""


Message = dict[str, str]  # {"role": ..., "content": ...}


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    """Anything that turns texts into vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI (or OpenAI-compatible) embeddings, batched."""

    def __init__(self, model: str, api_key: str | None, base_url: str | None, batch_size: int = 64) -> None:
        from openai import OpenAI

        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3, timeout=60)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                resp = self._client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:
                raise LLMError(f"Embedding request failed: {exc}") from exc
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
            logger.info("Embedded batch %d-%d", start, start + len(batch) - 1)
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
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def build_embedder(settings: Settings) -> Embedder:
    """Create the embedder selected by ``EMBEDDING_PROVIDER``."""
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(
            settings.embedding_model, settings.openai_api_key, settings.openai_base_url, settings.embed_batch_size
        )
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")


# --------------------------------------------------------------------------- #
# Chat LLM
# --------------------------------------------------------------------------- #


class ChatModel(Protocol):
    """Anything that completes a list of chat messages."""

    def complete(self, messages: list[Message], temperature: float = 0.1) -> str: ...


class OpenAIChatModel:
    """OpenAI (or OpenAI-compatible) chat completions."""

    def __init__(self, model: str, api_key: str | None, base_url: str | None) -> None:
        from openai import OpenAI

        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3, timeout=120)

    def complete(self, messages: list[Message], temperature: float = 0.1) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()


def build_chat_model(settings: Settings) -> ChatModel:
    """Create the chat model selected by ``LLM_PROVIDER``."""
    if settings.llm_provider.lower() == "openai":
        return OpenAIChatModel(settings.llm_model, settings.openai_api_key, settings.openai_base_url)
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
