"""Text extraction (PDF / DOCX / TXT) and cleaning.

Output is a list of pages (``Page``). Formats without real pages (DOCX, TXT
without form-feeds) produce a single page whose ``number`` is ``None``.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when a file cannot be turned into usable text."""


@dataclass
class Page:
    """Cleaned text of one page. ``number`` is 1-based, or None if unknown."""

    number: int | None
    text: str


# "12", "- 12 -", "Page 3", "Page 3 of 10", "3/10", "[3]"
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:[-–—\[(]\s*)?(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?(?:\s*[-–—\])])?\s*$",
    re.IGNORECASE,
)
_EDGE_LINES = 2  # how many lines at the top/bottom of a page are header/footer candidates
_MAX_HEADER_LEN = 120


def _edge_indices(lines: list[str]) -> set[int]:
    """Indices of the short, non-empty lines at the top/bottom of a page.

    Short pages only contribute their very first and last line, so body text
    is never mistaken for a running header/footer.
    """
    non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
    n = _EDGE_LINES if len(non_empty) >= 4 * _EDGE_LINES else 1
    return {i for i in non_empty[:n] + non_empty[-n:] if len(lines[i].strip()) <= _MAX_HEADER_LEN}


def _normalise_for_repeat(line: str) -> str:
    """Normalise a line so that 'Page 3 of 9' and 'Page 4 of 9' compare equal."""
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(raw_pages: list[str]) -> list[str]:
    """Remove page numbers and headers/footers repeated across pages.

    A line counts as a header/footer if (after digit-normalisation) it appears in
    the first/last ``_EDGE_LINES`` lines of at least half the pages (min. 2 pages).
    """
    split = [p.splitlines() for p in raw_pages]
    repeated: set[str] = set()
    if len(split) >= 2:
        counts: Counter[str] = Counter()
        for lines in split:
            counts.update({_normalise_for_repeat(lines[i]) for i in _edge_indices(lines)})
        threshold = max(2, (len(split) + 1) // 2)
        repeated = {line for line, n in counts.items() if n >= threshold and line}
        if repeated:
            logger.info("Removing %d repeated header/footer line(s)", len(repeated))

    cleaned: list[str] = []
    for lines in split:
        edge_idx = _edge_indices(lines)
        kept = []
        for i, ln in enumerate(lines):
            if _PAGE_NUMBER_RE.match(ln):
                continue
            if i in edge_idx and _normalise_for_repeat(ln) in repeated:
                continue
            kept.append(ln.rstrip())
        text = "\n".join(kept)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        cleaned.append(text)
    return cleaned


def _extract_pdf(path: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


def _extract_docx(path: Path) -> list[str]:
    import docx  # python-docx

    document = docx.Document(str(path))
    # Body paragraphs only: Word headers/footers live in separate parts and are skipped.
    return ["\n".join(p.text for p in document.paragraphs)]


def _extract_txt(path: Path) -> list[str]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\f")  # form-feed = page break


def extract_pages(path: Path) -> list[Page]:
    """Extract and clean the text of ``path``.

    Raises:
        ExtractionError: unsupported extension, unreadable file, or no text.
    """
    suffix = path.suffix.lower()
    extractors = {".pdf": _extract_pdf, ".docx": _extract_docx, ".txt": _extract_txt}
    if suffix not in extractors:
        raise ExtractionError(f"Unsupported file type: {suffix or '(none)'}")
    try:
        raw_pages = extractors[suffix](path)
    except ExtractionError:
        raise
    except Exception as exc:  # corrupt PDF, bad zip, ...
        raise ExtractionError(f"Could not read {path.name}: {exc}") from exc

    logger.info("Extracted %d raw page(s) from %s", len(raw_pages), path.name)
    cleaned = clean_pages(raw_pages)
    paginated = suffix == ".pdf" or len(raw_pages) > 1
    pages = [
        Page(number=(i + 1) if paginated else None, text=t)
        for i, t in enumerate(cleaned)
    ]
    if not any(p.text.strip() for p in pages):
        raise ExtractionError(f"No extractable text in {path.name}")
    return pages
