"""Text extraction (PDF / TXT) and paragraph-aware chunking."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
# Split on the coarsest boundary first, then fall back to finer ones.
SEPARATORS = ["\n\n", "\n", ". ", " "]


class IngestError(ValueError):
    """A problem with the uploaded file that the client can fix."""


class UnsupportedFileType(IngestError):
    pass


class EmptyFile(IngestError):
    pass


class NoExtractableText(IngestError):
    pass


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int  # position of the chunk within the document


def extract_pages(filename: str, data: bytes) -> list[Page]:
    """Return the non-empty pages of a PDF or text file, keeping page numbers."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(f"Unsupported file type '{suffix or filename}'. Upload a .pdf or .txt file.")
    if not data.strip():
        raise EmptyFile("The uploaded file is empty.")

    raw_pages = _read_pdf(data) if suffix == ".pdf" else _read_txt(data)
    pages = [Page(i, clean) for i, text in enumerate(raw_pages, start=1) if (clean := _normalize(text))]
    if not pages:
        if suffix == ".pdf":
            raise NoExtractableText(
                "No extractable text found in this PDF. It is probably a scanned image; OCR would be needed."
            )
        raise EmptyFile("The uploaded file contains no text.")
    return pages


def _read_pdf(data: bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        return [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise IngestError(f"Could not read the PDF: {exc}") from exc


def _read_txt(data: bytes) -> list[str]:
    # Form feeds (\f) are treated as page breaks; otherwise the whole file is page 1.
    return data.decode("utf-8", errors="replace").split("\f")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[Page], size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split every page into chunks of at most `size` chars with ~`overlap` chars of shared context.

    Chunks never span two pages, so each chunk has exactly one page number to cite.
    """
    chunks: list[Chunk] = []
    for page in pages:
        for text in _merge(_split(page.text, size, SEPARATORS), size, overlap):
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks)))
    return chunks


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively split text into pieces no longer than `size`, keeping separators attached."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, finer = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + sep if i < len(parts) - 1 else part
        pieces.extend(_split(piece, size, finer))
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, starting each new chunk with the tail of the previous one."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current.strip())
            tail = _tail(current, overlap)
            current = tail if len(tail) + len(piece) <= size else ""
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _tail(text: str, n: int) -> str:
    """The last ~n characters of text, starting at a word boundary."""
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
