"""Application settings loaded from environment variables / ``.env``.

Default deployment is fully Google Cloud:
  * Gemini (via Gemini API key or Vertex AI) for embeddings and answers
  * Cloud Storage bucket for the contract files
  * Firestore for the document registry and chunk text/metadata
  * Vertex AI Vector Search for the vectors

Every backend has a local alternative (``local`` / ``sqlite`` / ``chroma`` /
``hash``) used by the tests and for offline development.
"""

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


def _env_opt(name: str) -> str | None:
    return os.getenv(name) or None


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """All tunables for the backend. Every field can be overridden via env vars."""

    # ---- Google Cloud / Gemini auth --------------------------------------
    # Gemini API (AI Studio) key, or a Vertex AI express-mode key.
    google_api_key: str | None = field(default_factory=lambda: _env_opt("GOOGLE_API_KEY"))
    # True -> call Gemini through Vertex AI (project/location + ADC, or express key).
    use_vertexai: bool = field(default_factory=lambda: _env_bool("GOOGLE_GENAI_USE_VERTEXAI", False))
    gcp_project: str | None = field(default_factory=lambda: _env_opt("GCP_PROJECT"))
    gcp_location: str = field(default_factory=lambda: _env("GCP_LOCATION", "us-central1"))

    # ---- Models ------------------------------------------------------------
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "gemini"))  # gemini|hash
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "gemini-embedding-001"))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-2.5-flash"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.1))

    # ---- Contract files ----------------------------------------------------
    contracts_source: str = field(default_factory=lambda: _env("CONTRACTS_SOURCE", "gcs"))  # gcs|local
    gcs_bucket: str | None = field(default_factory=lambda: _env_opt("GCS_BUCKET"))
    gcs_prefix: str = field(default_factory=lambda: _env("GCS_PREFIX", "contracts/"))
    contracts_dir: Path = field(default_factory=lambda: Path(_env("CONTRACTS_DIR", "./contracts")))

    # ---- Document registry ---------------------------------------------------
    registry_backend: str = field(default_factory=lambda: _env("REGISTRY_BACKEND", "firestore"))  # firestore|sqlite
    firestore_database: str = field(default_factory=lambda: _env("FIRESTORE_DATABASE", "(default)"))
    firestore_documents_collection: str = field(
        default_factory=lambda: _env("FIRESTORE_DOCUMENTS_COLLECTION", "contract_documents")
    )
    firestore_chunks_collection: str = field(
        default_factory=lambda: _env("FIRESTORE_CHUNKS_COLLECTION", "contract_chunks")
    )
    sqlite_path: Path = field(default_factory=lambda: Path(_env("SQLITE_PATH", "./data/documents.db")))

    # ---- Vector store --------------------------------------------------------
    vector_store: str = field(default_factory=lambda: _env("VECTOR_STORE", "vertex"))  # vertex|chroma
    vertex_index_id: str | None = field(default_factory=lambda: _env_opt("VERTEX_INDEX_ID"))
    vertex_index_endpoint_id: str | None = field(default_factory=lambda: _env_opt("VERTEX_INDEX_ENDPOINT_ID"))
    vertex_deployed_index_id: str | None = field(default_factory=lambda: _env_opt("VERTEX_DEPLOYED_INDEX_ID"))
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", "./chroma_db")))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "contracts"))

    # ---- Chunking ------------------------------------------------------------
    max_chunk_tokens: int = field(default_factory=lambda: _env_int("MAX_CHUNK_TOKENS", 800))
    min_chunk_tokens: int = field(default_factory=lambda: _env_int("MIN_CHUNK_TOKENS", 50))
    min_clauses: int = field(default_factory=lambda: _env_int("MIN_CLAUSES_FOR_CLAUSE_CHUNKING", 3))
    fallback_chunk_size: int = field(default_factory=lambda: _env_int("FALLBACK_CHUNK_SIZE", 1000))
    fallback_chunk_overlap: int = field(default_factory=lambda: _env_int("FALLBACK_CHUNK_OVERLAP", 150))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 50))

    # ---- Retrieval / chat ------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.5))
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
