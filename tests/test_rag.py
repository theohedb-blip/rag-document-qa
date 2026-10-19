from app.rag import NOT_FOUND_MESSAGE, NOT_FOUND_SENTINEL, RagService
from tests.conftest import FakeLLM


def test_low_similarity_refuses_without_calling_llm(service: RagService, llm: FakeLLM, sample_pdf: bytes) -> None:
    doc = service.ingest("acme_handbook.pdf", sample_pdf)

    result = service.ask(doc.doc_id, "zebra quantum saxophone?")

    assert result.answered is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert llm.calls == []
    assert all(s.score < service.settings.min_score for s in result.sources)


def test_sentinel_maps_to_not_answered(service: RagService, llm: FakeLLM, sample_pdf: bytes) -> None:
    llm.reply = NOT_FOUND_SENTINEL
    doc = service.ingest("acme_handbook.pdf", sample_pdf)

    result = service.ask(doc.doc_id, "How many paid vacation days do full-time employees receive?")

    assert len(llm.calls) == 1
    assert result.answered is False
    assert result.answer == NOT_FOUND_MESSAGE


def test_grounded_prompt_contains_context_and_rules(service: RagService, llm: FakeLLM, sample_pdf: bytes) -> None:
    doc = service.ingest("acme_handbook.pdf", sample_pdf)

    result = service.ask(doc.doc_id, "How many paid vacation days do full-time employees receive?")

    system, prompt = llm.calls[0]
    assert NOT_FOUND_SENTINEL in system
    assert "[page 3]" in prompt and "25 paid vacation days" in prompt
    assert result.answered is True
    assert result.sources[0].page == 3


def test_reingesting_same_file_is_idempotent(service: RagService, sample_pdf: bytes) -> None:
    first = service.ingest("acme_handbook.pdf", sample_pdf)
    second = service.ingest("acme_handbook.pdf", sample_pdf)

    assert first == second
    assert len(service.store.list_documents()) == 1
