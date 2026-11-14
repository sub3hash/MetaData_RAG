"""Tests for clause detection, long-clause splitting, tiny-clause merging and fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.chunking import chunk_document, count_tokens, detect_headings, recursive_split
from app.extract import Page, clean_pages, extract_pages

CONTRACTS = Path(__file__).resolve().parent.parent / "contracts"


def _pages(text: str) -> list[Page]:
    return [Page(number=None, text=text)]


def _filler(n_words: int, word: str = "obligation") -> str:
    return " ".join([word] * n_words) + "."


# --------------------------------------------------------------------------- #
# Heading regex + validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "line, expected_no, expected_title",
    [
        ("1. DEFINITIONS", "1", "DEFINITIONS"),
        ("1.1 Notice Period. Either party may terminate.", "1.1", "Notice Period"),
        ("2.3.1 Payment Terms", "2.3.1", "Payment Terms"),
        ("Clause 5 - Confidentiality", "5", "Confidentiality"),
        ("Article IV: Termination", "Article IV", "Termination"),
        ("Section 3 Governing Law", "Section 3", "Governing Law"),
    ],
)
def test_heading_formats(line: str, expected_no: str, expected_title: str) -> None:
    # Build a valid sequence that ends with the heading under test.
    prefix = {
        "1": [],
        "1.1": ["1. Intro"],
        "2.3.1": ["1. A", "2. B", "2.1 C", "2.2 D", "2.3 E"],
        "5": ["Clause 1 A", "Clause 2 B", "Clause 3 C", "Clause 4 D"],
        "Article IV": ["Article I A", "Article II B", "Article III C"],
        "Section 3": ["Section 1 A", "Section 2 B"],
    }[expected_no]
    lines = prefix + [line]
    headings = detect_headings(lines)
    last = headings[-1]
    assert last.line_idx == len(lines) - 1
    assert last.clause_no == expected_no
    assert last.title == expected_title


def test_false_matches_rejected() -> None:
    lines = [
        "1. Term",
        "This agreement lasts until terminated. Payment is due within",
        "5 days of the invoice date.",  # bare number, no dot -> not a heading
        "2. Payment",
        "The Supplier shall:",
        "1. invoice monthly;",  # lowercase list item -> not a heading
        "1. Invoice monthly",  # out of sequence (2 -> 1) -> rejected
        "3. Notices",
        "Section 2 of this Agreement applies.",  # cross-reference (lowercase rest)
        "4. Law",
    ]
    nos = [h.clause_no for h in detect_headings(lines)]
    assert nos == ["1", "2", "3", "4"]


def test_increasing_order_and_restart_per_article() -> None:
    lines = [
        "Article I Definitions",
        "1. Meaning",
        "2. Interpretation",
        "Article II Obligations",
        "1. Use",  # numbering restarts inside the article
        "2. Care",
    ]
    nos = [h.clause_no for h in detect_headings(lines)]
    assert nos == ["Article I", "1", "2", "Article II", "1", "2"]


def test_preamble_and_header_prefix() -> None:
    text = "\n".join(
        [
            "SERVICES AGREEMENT between Alpha Ltd and Beta plc.",
            "",
            "1. Term. " + _filler(60),
            "2. Payment. " + _filler(60),
            "3. Notice Period. Either party may terminate on 30 days notice. " + _filler(60),
        ]
    )
    chunks = chunk_document(_pages(text))
    assert chunks[0].clause_no == "Preamble"
    assert chunks[0].text.startswith("[Preamble]")
    by_no = {c.clause_no: c for c in chunks}
    assert by_no["3"].clause_title == "Notice Period"
    assert by_no["3"].text.startswith("[Clause 3 – Notice Period]\n")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# --------------------------------------------------------------------------- #
# Long-clause split
# --------------------------------------------------------------------------- #


def test_long_clause_is_split_with_repeated_header() -> None:
    sub = "\n".join(f"({chr(97 + i)}) " + _filler(120, "indemnify") for i in range(8))
    text = "\n".join(
        [
            "1. Definitions. " + _filler(60),
            "2. Indemnities. The Supplier shall indemnify the Customer as follows:",
            sub,
            "3. Notices. " + _filler(60),
        ]
    )
    max_tokens = 300
    chunks = chunk_document(_pages(text), max_tokens=max_tokens)
    pieces = [c for c in chunks if c.clause_no == "2"]
    assert len(pieces) > 1
    for i, c in enumerate(pieces, start=1):
        assert c.text.startswith(f"[Clause 2 – Indemnities] (part {i} of {len(pieces)})")
        assert count_tokens(c.text) <= max_tokens + 10
    # Sub-clauses are kept intact (split happens on (a), (b), ... boundaries)
    joined = "\n".join(c.body for c in pieces)
    for letter in "abcdefgh":
        assert f"({letter})" in joined


def test_single_huge_paragraph_is_still_split() -> None:
    text = "\n".join(
        ["1. A. " + _filler(60), "2. Huge. " + ". ".join(["The party shall comply"] * 400), "3. C. " + _filler(60)]
    )
    pieces = [c for c in chunk_document(_pages(text), max_tokens=200) if c.clause_no == "2"]
    assert len(pieces) > 1
    assert all(count_tokens(c.text) <= 210 for c in pieces)


# --------------------------------------------------------------------------- #
# Tiny-clause merge
# --------------------------------------------------------------------------- #


def test_tiny_clauses_merge_into_next_under_same_parent() -> None:
    text = "\n".join(
        [
            "1. TERM",  # tiny parent heading -> merged with 1.1
            "1.1 Initial Term. " + _filler(80),
            "1.2 Renewal. " + _filler(80),
            "2. PAYMENT",
            "2.1 Fees. " + _filler(80),
            "2.2 Short.",  # tiny, but next clause has another parent -> NOT merged
            "3. LAW",
            "3.1 Governing Law. " + _filler(80),
        ]
    )
    chunks = chunk_document(_pages(text), min_tokens=50)
    nos = [c.clause_no for c in chunks]
    assert nos == ["1, 1.1", "1.2", "2, 2.1", "2.2", "3, 3.1"]
    first = chunks[0]
    assert first.text.startswith("[Clause 1, 1.1 – TERM]")
    assert "1.1 Initial Term" in first.body


# --------------------------------------------------------------------------- #
# Fallback
# --------------------------------------------------------------------------- #


def test_fallback_when_fewer_than_three_clauses() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + _filler(40, "confidential") for i in range(12))
    chunks = chunk_document(_pages(text), fallback_chunk_size=1000, fallback_overlap=150)
    assert len(chunks) > 1
    assert all(c.clause_no.startswith("Part ") for c in chunks)
    assert all(len(c.body) <= 1000 for c in chunks)


def test_recursive_split_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(600))
    parts = recursive_split(text, 1000, 150)
    assert len(parts) >= 2
    assert all(len(p) <= 1000 for p in parts)
    # consecutive chunks overlap
    tail = parts[0].split()[-5:]
    assert all(t in parts[1].split() for t in tail)


# --------------------------------------------------------------------------- #
# Cleaning + sample contracts
# --------------------------------------------------------------------------- #


def test_clean_pages_removes_headers_footers_and_page_numbers() -> None:
    pages = [
        f"ACME CONFIDENTIAL\n{i}. Clause text number {i}.\nMore text.\nPage {i} of 3" for i in range(1, 4)
    ]
    cleaned = clean_pages(pages)
    for i, page in enumerate(cleaned, start=1):
        assert "ACME CONFIDENTIAL" not in page
        assert "Page" not in page
        assert f"{i}. Clause text number {i}." in page


def test_sample_contracts_chunk_by_clause() -> None:
    msa = chunk_document(extract_pages(CONTRACTS / "master_services_agreement.txt"))
    nos = {n for c in msa for n in c.clause_no.split(", ")}
    assert {"Preamble", "3.3", "8.3", "9.2", "11.2"} <= nos
    nda = chunk_document(extract_pages(CONTRACTS / "mutual_nda.txt"))
    nos = {n for c in nda for n in c.clause_no.split(", ")}
    assert {"Article III", "Section 3.2", "Section 6.1"} <= nos
    assert all("Page 1 of 2" not in c.text for c in nda)
    assert next(c for c in nda if "Section 3.2" in c.clause_no).page == 2
