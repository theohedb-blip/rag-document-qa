"""FastAPI routes and the static single-page UI."""

import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingest import IngestError, NoExtractableText, UnsupportedFileType
from app.llm import MissingAPIKey, make_answer_model, make_embedder
from app.rag import DocumentNotFound, RagService
from app.store import VectorStore

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="RAG Document Q&A", version="1.0.0")


@app.exception_handler(MissingAPIKey)
def missing_key(_: Request, exc: MissingAPIKey) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@lru_cache
def get_service() -> RagService:
    settings = get_settings()
    return RagService(
        store=VectorStore(settings.chroma_dir, settings.collection_name),
        embedder=make_embedder(settings),
        llm=make_answer_model(settings),
        settings=settings,
    )


Service = Annotated[RagService, Depends(get_service)]


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class AskIn(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    page: int
    snippet: str
    score: float


class AskOut(BaseModel):
    answered: bool
    answer: str
    sources: list[SourceOut]
    latency_ms: int


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm_provider": get_settings().llm_provider}


@app.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(file: UploadFile, service: Service) -> DocumentOut:
    max_bytes = service.settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File is larger than {service.settings.max_upload_mb} MB.")
    try:
        doc = service.ingest(file.filename or "", data)
    except UnsupportedFileType as exc:
        raise HTTPException(415, str(exc)) from exc
    except NoExtractableText as exc:
        raise HTTPException(422, str(exc)) from exc
    except IngestError as exc:  # empty or unreadable file
        raise HTTPException(400, str(exc)) from exc
    return DocumentOut(**doc.__dict__)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(service: Service) -> list[DocumentOut]:
    return [DocumentOut(**d.__dict__) for d in service.store.list_documents()]


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, service: Service) -> AskOut:
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Question must not be blank.")
    started = time.perf_counter()
    try:
        result = service.ask(body.doc_id, question, body.top_k)
    except DocumentNotFound as exc:
        raise HTTPException(404, f"Unknown doc_id '{body.doc_id}'. Upload the document first.") from exc
    return AskOut(
        answered=result.answered,
        answer=result.answer,
        sources=[SourceOut(page=h.page, snippet=h.text, score=h.score) for h in result.sources],
        latency_ms=round((time.perf_counter() - started) * 1000),
    )
