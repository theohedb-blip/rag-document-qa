from fastapi.testclient import TestClient

from tests.conftest import FakeLLM


def _upload(client: TestClient, filename: str, data: bytes):
    return client.post("/documents", files={"file": (filename, data)})


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Q&amp;A" in response.text


def test_upload_and_list(client: TestClient, sample_pdf: bytes) -> None:
    response = _upload(client, "acme_handbook.pdf", sample_pdf)

    assert response.status_code == 201
    doc = response.json()
    assert doc["filename"] == "acme_handbook.pdf"
    assert doc["pages"] == 6
    assert doc["chunks"] > 6
    assert client.get("/documents").json() == [doc]


def test_upload_rejects_wrong_type(client: TestClient) -> None:
    response = _upload(client, "slides.pptx", b"not a pdf")
    assert response.status_code == 415
    assert ".pdf or .txt" in response.json()["detail"]


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = _upload(client, "empty.txt", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_upload_rejects_corrupt_pdf(client: TestClient) -> None:
    assert _upload(client, "broken.pdf", b"%PDF-1.4 garbage").status_code == 400


def test_upload_scanned_pdf_mentions_ocr(client: TestClient, blank_pdf: bytes) -> None:
    response = _upload(client, "scan.pdf", blank_pdf)
    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_ask_end_to_end(client: TestClient, llm: FakeLLM, sample_pdf: bytes) -> None:
    doc_id = _upload(client, "acme_handbook.pdf", sample_pdf).json()["doc_id"]

    response = client.post(
        "/ask",
        json={"doc_id": doc_id, "question": "How many paid vacation days do full-time employees receive?", "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is True
    assert body["answer"] == llm.reply
    assert 1 <= len(body["sources"]) <= 3
    assert body["sources"][0]["page"] == 3
    assert set(body["sources"][0]) == {"page", "snippet", "score"}
    assert isinstance(body["latency_ms"], int)


def test_ask_unanswerable_end_to_end(client: TestClient, llm: FakeLLM, sample_pdf: bytes) -> None:
    doc_id = _upload(client, "acme_handbook.pdf", sample_pdf).json()["doc_id"]

    body = client.post("/ask", json={"doc_id": doc_id, "question": "zebra quantum saxophone?"}).json()

    assert body["answered"] is False
    assert body["answer"] == "Not found in the document."
    assert llm.calls == []


def test_ask_unknown_document(client: TestClient) -> None:
    response = client.post("/ask", json={"doc_id": "nope", "question": "Anything?"})
    assert response.status_code == 404


def test_ask_rejects_blank_question(client: TestClient) -> None:
    assert client.post("/ask", json={"doc_id": "x", "question": "   "}).status_code == 422


def test_missing_api_key_returns_503(monkeypatch) -> None:
    from app import main
    from app.config import Settings

    monkeypatch.setattr(main, "get_settings", lambda: Settings(openai_api_key="", _env_file=None))
    main.get_service.cache_clear()
    response = TestClient(main.app).get("/documents")
    main.get_service.cache_clear()

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
