"""Where contract files come from: a Cloud Storage bucket or a local folder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFile:
    """A contract file in the source.

    ``key`` locates the file in the source (object name / path), ``version``
    changes whenever the content may have changed (GCS generation / mtime).
    """

    key: str
    file_name: str
    size: int
    version: str


class ContractSource(Protocol):
    """Lists and reads contract files."""

    def list_files(self) -> list[SourceFile]: ...

    def read_bytes(self, key: str) -> bytes: ...


def _supported(name: str, extensions: Sequence[str]) -> bool:
    return Path(name).suffix.lower() in extensions


class LocalFolderSource:
    """Contracts stored in a local directory (development / tests)."""

    def __init__(self, folder: Path, extensions: Sequence[str]) -> None:
        self.folder = folder
        self.extensions = tuple(extensions)

    def list_files(self) -> list[SourceFile]:
        self.folder.mkdir(parents=True, exist_ok=True)
        out = []
        for path in sorted(self.folder.iterdir()):
            if path.is_file() and _supported(path.name, self.extensions):
                st = path.stat()
                out.append(SourceFile(str(path), path.name, st.st_size, f"{st.st_mtime_ns}"))
        return out

    def read_bytes(self, key: str) -> bytes:
        return Path(key).read_bytes()


class GCSSource:
    """Contracts stored as objects under ``gs://<bucket>/<prefix>``."""

    def __init__(self, bucket: str, prefix: str, extensions: Sequence[str], project: str | None = None) -> None:
        from google.cloud import storage

        self._client = storage.Client(project=project)
        self._bucket = self._client.bucket(bucket)
        self.bucket_name = bucket
        self.prefix = prefix
        self.extensions = tuple(extensions)

    def list_files(self) -> list[SourceFile]:
        out = []
        for blob in self._client.list_blobs(self.bucket_name, prefix=self.prefix or None):
            if blob.name.endswith("/") or not _supported(blob.name, self.extensions):
                continue
            rel = blob.name[len(self.prefix):] if self.prefix and blob.name.startswith(self.prefix) else blob.name
            out.append(SourceFile(blob.name, rel, int(blob.size or 0), str(blob.generation)))
        return sorted(out, key=lambda f: f.file_name.lower())

    def read_bytes(self, key: str) -> bytes:
        logger.info("Downloading gs://%s/%s", self.bucket_name, key)
        return self._bucket.blob(key).download_as_bytes()


def build_source(settings: Any) -> ContractSource:
    """Create the contract source selected by ``CONTRACTS_SOURCE``."""
    kind = settings.contracts_source.lower()
    if kind == "local":
        return LocalFolderSource(settings.contracts_dir, settings.supported_extensions)
    if kind == "gcs":
        if not settings.gcs_bucket:
            raise ValueError("CONTRACTS_SOURCE=gcs requires GCS_BUCKET")
        return GCSSource(settings.gcs_bucket, settings.gcs_prefix, settings.supported_extensions, settings.gcp_project)
    raise ValueError(f"Unknown CONTRACTS_SOURCE: {settings.contracts_source}")
