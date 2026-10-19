"""The RAG pipeline: ingest a document, then retrieve → threshold → grounded prompt → answer."""

import hashlib
from dataclasses import dataclass

from app.config import Settings
from app.ingest import chunk_pages, extract_pages
from app.llm import AnswerModel, Embedder
from app.store import DocumentInfo, Hit, VectorStore

NOT_FOUND_SENTINEL = "NOT_IN_DOCUMENT"
NOT_FOUND_MESSAGE = "Not found in the document."

SYSTEM_PROMPT = f"""You answer questions about a document using ONLY the context passages provided.
Rules:
- Use only facts stated in the context. Do not use outside knowledge, and do not guess.
- If the context does not contain the answer, reply with exactly: {NOT_FOUND_SENTINEL}
- Be concise (1-3 sentences) and cite the page(s) you used, like (p. 3)."""


class DocumentNotFound(LookupError):
    pass


@dataclass(frozen=True)
class Answer:
    answered: bool
    answer: str
    sources: list[Hit]


class RagService:
    def __init__(self, store: VectorStore, embedder: Embedder, llm: AnswerModel, settings: Settings) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.settings = settings

    def ingest(self, filename: str, data: bytes) -> DocumentInfo:
        """Extract, chunk, embed and store a file. Re-uploading the same bytes is a no-op."""
        doc_id = hashlib.sha256(data).hexdigest()[:12]
        if existing := self.store.get_document(doc_id):
            return existing

        pages = extract_pages(filename, data)
        chunks = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        doc = DocumentInfo(doc_id=doc_id, filename=filename, pages=pages[-1].number, chunks=len(chunks))
        self.store.add(doc, chunks, self.embedder.embed([c.text for c in chunks]))
        return doc

    def retrieve(self, doc_id: str, question: str, top_k: int | None = None) -> list[Hit]:
        if self.store.get_document(doc_id) is None:
            raise DocumentNotFound(doc_id)
        [query_vector] = self.embedder.embed([question])
        return self.store.search(doc_id, query_vector, top_k or self.settings.top_k)

    def ask(self, doc_id: str, question: str, top_k: int | None = None) -> Answer:
        hits = self.retrieve(doc_id, question, top_k)
        relevant = [h for h in hits if h.score >= self.settings.min_score]

        # Layer 1: nothing similar enough was retrieved, so don't spend an LLM call.
        if not relevant:
            return Answer(answered=False, answer=NOT_FOUND_MESSAGE, sources=hits)

        # Layer 2: the model itself says the context doesn't contain the answer.
        reply = self.llm.answer(SYSTEM_PROMPT, build_prompt(question, relevant)).strip()
        if not reply or NOT_FOUND_SENTINEL in reply:
            return Answer(answered=False, answer=NOT_FOUND_MESSAGE, sources=relevant)
        return Answer(answered=True, answer=reply, sources=relevant)


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[page {h.page}]\n{h.text}" for h in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"
