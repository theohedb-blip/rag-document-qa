"""Shared fixtures: a RagService wired to deterministic fakes, so tests need no API key."""

import hashlib
import math
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_service
from app.rag import RagService
from app.store import VectorStore

SAMPLE_PDF = Path(__file__).parents[1] / "sample" / "acme_handbook.pdf"


class FakeEmbedder:
    """Bag-of-words hashed into a fixed-size unit vector: texts sharing words get similar vectors."""

    dims = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FakeLLM:
    """Returns a canned reply and records every call."""

    def __init__(self, reply: str = "Full-time employees get 25 vacation days (p. 3).") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def answer(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.reply


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def service(tmp_path: Path, llm: FakeLLM) -> RagService:
    settings = Settings(chroma_dir=str(tmp_path / "chroma"), min_score=0.2, _env_file=None)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    return RagService(store=store, embedder=FakeEmbedder(), llm=llm, settings=settings)


@pytest.fixture
def client(service: RagService):
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf() -> bytes:
    return SAMPLE_PDF.read_bytes()


@pytest.fixture
def blank_pdf() -> bytes:
    """A valid PDF page with no text layer, like a scanned image."""
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
