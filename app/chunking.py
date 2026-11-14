"""Clause-based chunking of contract text.

Pipeline (see README for the rationale):

1. Flatten pages into lines, remembering the page of every line.
2. Find *candidate* clause headings with regexes (``1.``, ``1.1``, ``2.3.1``,
   ``Clause 5``, ``Article IV``, ``Section 3`` ... optionally followed by a title).
3. Validate candidates: per numbering family we keep the best chain of headings
   whose numbers increase in a plausible way (``7`` -> ``7.1`` -> ``7.2`` -> ``8``).
   This drops false matches such as list items or numbers inside sentences.
4. Text between two accepted headings is one clause; text before the first one
   is the ``Preamble``.
5. Tiny clauses (< ``min_tokens``) are merged into the next clause of the same
   parent; long clauses (> ``max_tokens``) are split on sub-clause/paragraph
   boundaries, repeating the ``[Clause X – Title]`` header on every piece.
6. If fewer than ``min_clauses`` headings survive, fall back to a
   LangChain-style recursive character splitter (1000 chars, 150 overlap).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

from app.extract import Page

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _encoder():  # pragma: no cover - depends on environment
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # offline, not installed, ...
        logger.warning("tiktoken unavailable (%s); approximating tokens as chars/4", exc)
        return None


def count_tokens(text: str) -> int:
    """Number of tokens in ``text`` (cl100k_base, or chars/4 if tiktoken is unavailable)."""
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, len(text) // 4) if text else 0


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """A piece of contract text ready to be embedded."""

    text: str  # header + body; this is what gets embedded
    body: str  # body without the header (used for snippets)
    clause_no: str  # "7.2", "Article IV", "Preamble", "7, 7.1" (merged), "Part 3" (fallback)
    clause_title: str
    chunk_index: int
    page: int | None = None


@dataclass
class _Line:
    text: str
    page: int | None


@dataclass
class _Heading:
    line_idx: int
    family: str  # "article" | "section" | "clause"
    number: tuple[int, ...]
    clause_no: str
    title: str
    has_title: bool


@dataclass
class _Clause:
    clause_nos: list[str]
    title: str
    lines: list[_Line]
    family: str | None = None  # None for the preamble
    number: tuple[int, ...] = ()

    @property
    def body(self) -> str:
        return "\n".join(l.text for l in self.lines).strip()


# --------------------------------------------------------------------------- #
# Heading detection
# --------------------------------------------------------------------------- #

_KEYWORD_RE = re.compile(
    r"^\s*(?P<kw>article|section|clause)\s+"
    r"(?P<num>\d{1,3}(?:\.\d{1,3})*|(?-i:[IVXLCDM]+))\b\.?"
    r"\s*(?:[:.\-–—)]\s*)?(?P<rest>.*)$",
    re.IGNORECASE,
)
# "1." / "1.1" / "2.3.1" / "1.1." ; a bare "5" (no dot, single level) is NOT a heading.
_NUMERIC_RE = re.compile(r"^\s*(?P<num>\d{1,3}(?:\.\d{1,3})*)(?P<dot>\.)?(?=\s|$)\s*(?P<rest>.*)$")
# Sub-clause markers used when splitting long clauses: (a) a) (iv) 7.2.1
_SUBCLAUSE_RE = re.compile(r"^\s*(?:\(?[a-z]{1,2}\)|\(?[ivx]{1,5}\)|\d{1,3}(?:\.\d{1,3}){2,})\s")

_SMALL_WORDS = {"of", "and", "the", "to", "for", "in", "on", "a", "an", "or", "by", "with", "at", "&"}
_FAMILY_RANK = {"article": 0, "section": 1, "clause": 2}
_MAX_GAP = 3  # tolerate up to two missing headings in a sequence


def _roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = values[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def _looks_like_title(text: str) -> bool:
    """Short, Title Case or UPPER CASE phrase (e.g. 'Limitation of Liability')."""
    text = text.strip().rstrip(".:")
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)
    if not words or len(words) > 10 or len(text) > 90:
        return False
    if text.endswith((";", ",")):
        return False
    significant = [w for w in words if w.lower() not in _SMALL_WORDS] or words
    capitalised = sum(1 for w in significant if w[0].isupper())
    return capitalised / len(significant) >= 0.8


def _split_title(rest: str) -> tuple[str, bool]:
    """Extract a clause title from the text following the clause number."""
    rest = rest.strip()
    if not rest:
        return "", False
    # "Notice Period. Either party may ..." / "Term: This Agreement ..."
    m = re.match(r"^(?P<title>[^.:;]{1,90}?)\s*[.:]\s+\S", rest)
    if m and _looks_like_title(m.group("title")):
        return m.group("title").strip(), True
    if _looks_like_title(rest):
        return rest.rstrip(".:").strip(), True
    return "", False


def _parse_heading(line: str, idx: int) -> _Heading | None:
    """Return a heading candidate for ``line`` or None."""
    m = _KEYWORD_RE.match(line)
    if m:
        kw = m.group("kw").lower()
        raw = m.group("num")
        rest = m.group("rest")
        if rest and rest[0].islower():  # "Section 3 of this Agreement ..." (a cross-reference)
            return None
        number = (_roman_to_int(raw),) if raw.isalpha() else tuple(int(x) for x in raw.split("."))
        family = "clause" if kw == "clause" else kw
        clause_no = raw if kw == "clause" else f"{kw.title()} {raw}"
        title, has_title = _split_title(rest)
        return _Heading(idx, family, number, clause_no, title, has_title)

    m = _NUMERIC_RE.match(line)
    if m:
        raw, dot, rest = m.group("num"), m.group("dot"), m.group("rest")
        if "." not in raw and not dot:  # "5 days" -> not a heading
            return None
        if rest and (rest[0].islower() or rest[0] in "%$€£"):  # "5. days", "1.5 %"
            return None
        if not rest.strip() and "." not in raw and not dot:
            return None
        number = tuple(int(x) for x in raw.split("."))
        title, has_title = _split_title(rest)
        return _Heading(idx, "clause", number, raw, title, has_title)
    return None


def _is_successor(prev: tuple[int, ...], cand: tuple[int, ...]) -> bool:
    """Is ``cand`` a plausible next heading after ``prev``?

    Accepts first children (7 -> 7.1), siblings / next ancestors (7.2 -> 7.3,
    7.2 -> 8) and new parents without their own heading line (3.2 -> 4.1).
    """
    n = len(cand)
    if n > len(prev) and cand[: len(prev)] == prev and all(x == 1 for x in cand[len(prev):]):
        return True
    if n <= len(prev) and cand[: n - 1] == prev[: n - 1] and prev[n - 1] < cand[-1] <= prev[n - 1] + _MAX_GAP:
        return True
    if n >= 2 and cand[-1] == 1 and _is_successor(prev, cand[:-1]):
        return True
    return False


def _can_start(cand: tuple[int, ...], first_segment: bool) -> bool:
    """Can ``cand`` be the first heading of a chain?"""
    if any(x > 1 for x in cand[1:]):
        return False
    return cand[0] <= 1 or not first_segment


def _best_chain(cands: list[_Heading], first_segment: bool) -> list[_Heading]:
    """Highest-scoring chain of candidates with increasing numbering (DP, O(n^2))."""
    if not cands:
        return []
    weights = [1.0 + (0.5 if c.has_title else 0.0) for c in cands]
    best = [float("-inf")] * len(cands)
    back: list[int | None] = [None] * len(cands)
    for i, c in enumerate(cands):
        if _can_start(c.number, first_segment):
            best[i] = weights[i]
        for j in range(i):
            if best[j] != float("-inf") and _is_successor(cands[j].number, c.number):
                if best[j] + weights[i] > best[i]:
                    best[i], back[i] = best[j] + weights[i], j
    end = max(range(len(cands)), key=lambda k: best[k])
    if best[end] == float("-inf"):
        return []
    chain: list[_Heading] = []
    k: int | None = end
    while k is not None:
        chain.append(cands[k])
        k = back[k]
    return chain[::-1]


def detect_headings(lines: Sequence[str]) -> list[_Heading]:
    """Detect and validate clause headings in ``lines``.

    Families are processed from the highest (Article) to the lowest (numeric
    clauses). Lower families are validated within the segments delimited by
    already-accepted higher-level headings, so numbering that restarts inside
    each Article is handled.
    """
    candidates = [h for i, ln in enumerate(lines) if (h := _parse_heading(ln, i))]
    accepted: list[_Heading] = []
    for family in ("article", "section", "clause"):
        fam = [c for c in candidates if c.family == family]
        if not fam:
            continue
        bounds = sorted(h.line_idx for h in accepted)
        segments: list[list[_Heading]] = [[] for _ in range(len(bounds) + 1)]
        for c in fam:
            seg = sum(1 for b in bounds if b < c.line_idx)
            segments[seg].append(c)
        for seg_idx, seg in enumerate(segments):
            accepted.extend(_best_chain(seg, first_segment=(seg_idx == 0 or not bounds)))
    accepted.sort(key=lambda h: h.line_idx)

    # Title on the following line: "ARTICLE IV" / "NOTICES"
    for h in accepted:
        if not h.title and _KEYWORD_RE.match(lines[h.line_idx]) and not _KEYWORD_RE.match(lines[h.line_idx]).group("rest").strip():
            nxt = next((l for l in lines[h.line_idx + 1 : h.line_idx + 3] if l.strip()), "")
            if nxt and _looks_like_title(nxt) and not _parse_heading(nxt, -1):
                h.title, h.has_title = nxt.strip().rstrip(".:"), True
    return accepted


# --------------------------------------------------------------------------- #
# Generic recursive splitter (LangChain RecursiveCharacterTextSplitter style)
# --------------------------------------------------------------------------- #


def recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    length_fn: Callable[[str], int] = len,
    separators: Sequence[str] = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Split ``text`` into pieces of at most ``chunk_size`` (measured by ``length_fn``)."""

    def merge(splits: list[str]) -> list[str]:
        docs: list[str] = []
        current: list[str] = []
        total = 0
        for s in splits:
            ln = length_fn(s)
            if current and total + ln > chunk_size:
                docs.append("".join(current))
                while current and (total > chunk_overlap or total + ln > chunk_size):
                    total -= length_fn(current.pop(0))
            current.append(s)
            total += ln
        if current:
            docs.append("".join(current))
        return docs

    def split(txt: str, seps: Sequence[str]) -> list[str]:
        sep, rest = seps[-1], []
        for i, s in enumerate(seps):
            if s == "" or s in txt:
                sep, rest = s, list(seps[i + 1 :])
                break
        if sep:
            parts = txt.split(sep)
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
        else:
            parts = list(txt)
        parts = [p for p in parts if p]
        out: list[str] = []
        good: list[str] = []
        for p in parts:
            if length_fn(p) <= chunk_size:
                good.append(p)
            else:
                if good:
                    out.extend(merge(good))
                    good = []
                out.extend(split(p, rest) if rest else [p])
        if good:
            out.extend(merge(good))
        return out

    return [c.strip() for c in split(text, separators) if c.strip()]


# --------------------------------------------------------------------------- #
# Clause assembly, merge and split
# --------------------------------------------------------------------------- #


def _label(clause_nos: list[str]) -> str:
    """'Clause 7.2', 'Clause 7, 7.1', 'Article IV', 'Preamble'."""
    if all(n[:1].isdigit() for n in clause_nos):
        return "Clause " + ", ".join(clause_nos)
    return ", ".join(f"Clause {n}" if n[:1].isdigit() else n for n in clause_nos)


def make_header(clause_nos: list[str], title: str) -> str:
    """Build the ``[Clause 7.2 – Notice Period]`` header."""
    label = _label(clause_nos)
    return f"[{label} – {title}]" if title else f"[{label}]"


def _same_parent(cur: _Clause, nxt: _Clause) -> bool:
    if cur.family is None or nxt.family is None:  # preamble
        return False
    if cur.family == nxt.family:
        return cur.number[:1] == nxt.number[:1]
    # e.g. "Article IV" heading followed by its first "Section 4.01"
    return _FAMILY_RANK[cur.family] < _FAMILY_RANK[nxt.family]


def _merge_tiny(clauses: list[_Clause], min_tokens: int) -> list[_Clause]:
    out: list[_Clause] = []
    carry: _Clause | None = None
    for i, cl in enumerate(clauses):
        if carry is not None:
            cl = _Clause(
                clause_nos=carry.clause_nos + cl.clause_nos,
                title=carry.title or cl.title,
                lines=carry.lines + cl.lines,
                family=cl.family,
                number=cl.number,
            )
            carry = None
        nxt = clauses[i + 1] if i + 1 < len(clauses) else None
        if nxt is not None and count_tokens(cl.body) < min_tokens and _same_parent(cl, nxt):
            carry = cl
            continue
        out.append(cl)
    return out


def _blocks(lines: list[_Line]) -> list[list[_Line]]:
    """Split clause lines into paragraph / sub-clause blocks."""
    blocks: list[list[_Line]] = []
    cur: list[_Line] = []
    for ln in lines:
        if not ln.text.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if cur and _SUBCLAUSE_RE.match(ln.text):
            blocks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    return blocks


def _split_long(cl: _Clause, max_tokens: int) -> list[tuple[str, int | None]]:
    """Split a clause body into pieces of <= ``max_tokens``. Returns (text, page)."""
    body = cl.body
    budget = max(50, max_tokens - count_tokens(make_header(cl.clause_nos, cl.title)) - 8)
    if count_tokens(body) <= budget:
        return [(body, cl.lines[0].page if cl.lines else None)]

    units: list[tuple[str, int | None]] = []
    for block in _blocks(cl.lines):
        text = "\n".join(l.text for l in block).strip()
        if count_tokens(text) <= budget:
            units.append((text, block[0].page))
        else:  # one huge paragraph: sentence/word level split
            for piece in recursive_split(text, budget, 0, count_tokens, ("\n", ". ", "; ", " ", "")):
                units.append((piece, block[0].page))

    pieces: list[tuple[str, int | None]] = []
    cur_text, cur_page, cur_tokens = "", None, 0
    for text, page in units:
        t = count_tokens(text)
        if cur_text and cur_tokens + t + 1 > budget:
            pieces.append((cur_text, cur_page))
            cur_text, cur_page, cur_tokens = "", None, 0
        cur_text = f"{cur_text}\n\n{text}" if cur_text else text
        cur_page = cur_page if cur_page is not None else page
        cur_tokens += t + 1
    if cur_text:
        pieces.append((cur_text, cur_page))
    return pieces


def _flatten(pages: Sequence[Page]) -> list[_Line]:
    lines: list[_Line] = []
    for p in pages:
        for t in p.text.split("\n"):
            lines.append(_Line(t.rstrip(), p.number))
        lines.append(_Line("", p.number))
    return lines


def _fallback_chunks(lines: list[_Line], chunk_size: int, overlap: int) -> list[Chunk]:
    """Recursive character splitting when no clause structure is detected."""
    full = ""
    offsets: list[tuple[int, int | None]] = []  # (start offset, page)
    for ln in lines:
        offsets.append((len(full), ln.page))
        full += ln.text + "\n"
    pieces = recursive_split(full, chunk_size, overlap)
    chunks: list[Chunk] = []
    pos = 0
    for i, piece in enumerate(pieces):
        start = full.find(piece[:80], pos)
        start = pos if start < 0 else start
        pos = start + 1
        page = None
        for off, pg in offsets:
            if off > start:
                break
            page = pg
        clause_no = f"Part {i + 1}"
        header = make_header([clause_no], "")
        chunks.append(Chunk(f"{header}\n{piece}", piece, clause_no, "", i, page))
    return chunks


def chunk_document(
    pages: Sequence[Page],
    max_tokens: int = 800,
    min_tokens: int = 50,
    min_clauses: int = 3,
    fallback_chunk_size: int = 1000,
    fallback_overlap: int = 150,
) -> list[Chunk]:
    """Turn extracted pages into header-prefixed chunks (see module docstring)."""
    lines = _flatten(pages)
    headings = detect_headings([l.text for l in lines])
    logger.info("Detected %d clause heading(s)", len(headings))

    if len(headings) < min_clauses:
        logger.info(
            "Fewer than %d clauses detected -> recursive character fallback (%d chars, %d overlap)",
            min_clauses, fallback_chunk_size, fallback_overlap,
        )
        return _fallback_chunks(lines, fallback_chunk_size, fallback_overlap)

    clauses: list[_Clause] = []
    preamble = lines[: headings[0].line_idx]
    if any(l.text.strip() for l in preamble):
        clauses.append(_Clause(["Preamble"], "", preamble))
    for i, h in enumerate(headings):
        end = headings[i + 1].line_idx if i + 1 < len(headings) else len(lines)
        clauses.append(_Clause([h.clause_no], h.title, lines[h.line_idx : end], h.family, h.number))

    before = len(clauses)
    clauses = _merge_tiny(clauses, min_tokens)
    if len(clauses) != before:
        logger.info("Merged tiny clauses: %d -> %d", before, len(clauses))

    chunks: list[Chunk] = []
    for cl in clauses:
        if not cl.body:
            continue
        pieces = _split_long(cl, max_tokens)
        header = make_header(cl.clause_nos, cl.title)
        for k, (text, page) in enumerate(pieces):
            hdr = header if len(pieces) == 1 else f"{header} (part {k + 1} of {len(pieces)})"
            chunks.append(
                Chunk(
                    text=f"{hdr}\n{text}",
                    body=text,
                    clause_no=", ".join(cl.clause_nos),
                    clause_title=cl.title,
                    chunk_index=len(chunks),
                    page=page,
                )
            )
    logger.info("Produced %d chunk(s) from %d clause(s)", len(chunks), len(clauses))
    return chunks
