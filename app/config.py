"""Application settings loaded from environment variables / ``.env``."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """All tunables for the backend. Every field can be overridden via env vars."""

    # Providers
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "openai"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-3-small"))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None)

    # Paths
    contracts_dir: Path = field(default_factory=lambda: Path(_env("CONTRACTS_DIR", "./contracts")))
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", "./chroma_db")))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "contracts"))
    sqlite_path: Path = field(default_factory=lambda: Path(_env("SQLITE_PATH", "./data/documents.db")))

    # Chunking
    max_chunk_tokens: int = field(default_factory=lambda: _env_int("MAX_CHUNK_TOKENS", 800))
    min_chunk_tokens: int = field(default_factory=lambda: _env_int("MIN_CHUNK_TOKENS", 50))
    min_clauses: int = field(default_factory=lambda: _env_int("MIN_CLAUSES_FOR_CLAUSE_CHUNKING", 3))
    fallback_chunk_size: int = field(default_factory=lambda: _env_int("FALLBACK_CHUNK_SIZE", 1000))
    fallback_chunk_overlap: int = field(default_factory=lambda: _env_int("FALLBACK_CHUNK_OVERLAP", 150))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 64))

    # Retrieval / chat
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.2))
    max_history_turns: int = field(default_factory=lambda: _env_int("MAX_HISTORY_TURNS", 6))

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    supported_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings()


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
