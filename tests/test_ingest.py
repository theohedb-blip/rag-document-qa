from itertools import pairwise

import pytest

from app.ingest import (
    EmptyFile,
    NoExtractableText,
    Page,
    UnsupportedFileType,
    chunk_pages,
    extract_pages,
)


def _paragraphs(n: int, words: int = 30) -> str:
    return "\n\n".join(" ".join(f"p{i}w{j}" for j in range(words)) for i in range(n))


def test_chunks_respect_size_and_overlap() -> None:
    chunks = chunk_pages([Page(1, _paragraphs(20))], size=800, overlap=150)

    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)
    for prev, nxt in pairwise(chunks):
        # The next chunk starts with text taken from the end of the previous one.
        assert nxt.text[:40] in prev.text[-160:]


def test_chunks_keep_page_numbers_and_global_index() -> None:
    pages = [Page(1, _paragraphs(8)), Page(2, "Short page two."), Page(5, _paragraphs(8))]
    chunks = chunk_pages(pages, size=800, overlap=150)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2, 5}
    assert [c.text for c in chunks if c.page == 2] == ["Short page two."]


def test_oversized_paragraph_is_split_on_finer_boundaries() -> None:
    sentence = "This sentence is about expenses and receipts. "
    chunks = chunk_pages([Page(1, sentence * 60)], size=800, overlap=150)

    assert all(len(c.text) <= 800 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)  # split at sentence ends, not mid-word


def test_extracts_pdf_pages_with_numbers(sample_pdf: bytes) -> None:
    pages = extract_pages("acme_handbook.pdf", sample_pdf)

    assert [p.number for p in pages] == [1, 2, 3, 4, 5, 6]
    assert "25 paid vacation days" in pages[2].text


def test_txt_form_feed_starts_a_new_page() -> None:
    pages = extract_pages("notes.txt", b"first page\fsecond page")
    assert [(p.number, p.text) for p in pages] == [(1, "first page"), (2, "second page")]


@pytest.mark.parametrize(
    ("filename", "data", "error"),
    [
        ("report.docx", b"data", UnsupportedFileType),
        ("empty.txt", b"", EmptyFile),
        ("blank.txt", b"  \n\n ", EmptyFile),
    ],
)
def test_invalid_files_are_rejected(filename: str, data: bytes, error: type[Exception]) -> None:
    with pytest.raises(error):
        extract_pages(filename, data)


def test_pdf_without_text_asks_for_ocr(blank_pdf: bytes) -> None:
    with pytest.raises(NoExtractableText, match="OCR"):
        extract_pages("scan.pdf", blank_pdf)
