"""Thin wrapper around a persistent Chroma collection (cosine distance)."""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingest import Chunk


@dataclass(frozen=True)
class Hit:
    text: str
    page: int
    chunk_index: int
    score: float  # cosine similarity, 1.0 = identical


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    filename: str
    pages: int
    chunks: int


class VectorStore:
    def __init__(self, path: str, collection: str = "documents") -> None:
        client = chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
        self._collection = client.get_or_create_collection(name=collection, configuration={"hnsw": {"space": "cosine"}})

    def add(self, doc: DocumentInfo, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[f"{doc.doc_id}:{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": doc.doc_id,
                    "filename": doc.filename,
                    "page": c.page,
                    "chunk_index": c.index,
                    "doc_pages": doc.pages,
                }
                for c in chunks
            ],
        )

    def search(self, doc_id: str, embedding: list[float], top_k: int) -> list[Hit]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(text=text, page=int(meta["page"]), chunk_index=int(meta["chunk_index"]), score=round(1 - dist, 4))
            for text, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        return next((d for d in self.list_documents() if d.doc_id == doc_id), None)

    def list_documents(self) -> list[DocumentInfo]:
        """Aggregate chunk metadata into one entry per document (fine at prototype scale)."""
        docs: dict[str, DocumentInfo] = {}
        for meta in self._collection.get(include=["metadatas"])["metadatas"]:
            doc_id = str(meta["doc_id"])
            seen = docs.get(doc_id)
            docs[doc_id] = DocumentInfo(
                doc_id=doc_id,
                filename=str(meta["filename"]),
                pages=int(meta["doc_pages"]),
                chunks=(seen.chunks if seen else 0) + 1,
            )
        return sorted(docs.values(), key=lambda d: d.filename)
