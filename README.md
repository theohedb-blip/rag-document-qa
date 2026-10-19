# RAG Document Q&A

**Upload a PDF or text file and ask questions about it. Every answer comes from the document, cites its pages, and says "not found" instead of guessing.**

FastAPI · Chroma · OpenAI embeddings · OpenAI or Claude for answers · about 500 lines of plain Python, no LangChain.

| Answer with sources | Question the document can't answer |
|---|---|
| ![Answer with sources](docs/screenshot.png) | ![Not found state](docs/screenshot-not-found.png) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then set OPENAI_API_KEY in .env
uvicorn app.main:app --reload   # open http://localhost:8000
```

Upload `sample/acme_handbook.pdf`, a fictional six-page employee handbook, and ask *"How many vacation days do I get?"*.

**Docker** works too:

```bash
docker build -t rag-document-qa .
docker run --rm -p 8000:8000 --env-file .env -v rag-data:/data rag-document-qa
```

To answer with Claude instead of OpenAI, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env`. Embeddings always use OpenAI.

## API

Interactive docs are at `http://localhost:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload a `.pdf` or `.txt` (multipart) and index it |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/ask` | Ask a question about one document |
| `GET` | `/health` | Liveness check |

<!-- CURL_EXAMPLES -->

Upload errors return clear 4xx responses: `415` for a wrong file type, `400` for an empty or corrupt file, `413` for a file over `MAX_UPLOAD_MB`, and `422` for a PDF without a text layer (*"…It is probably a scanned image; OCR would be needed."*). Asking about an unknown `doc_id` returns `404`.

## How indexing and retrieval work

```mermaid
flowchart LR
    subgraph Indexing
        A[Upload PDF / TXT] --> B[Extract text<br/>per page]
        B --> C[Chunk<br/>~800 chars, 150 overlap]
        C --> D[Embed<br/>text-embedding-3-small]
        D --> E[(Chroma<br/>cosine)]
    end
    subgraph Answering
        Q[Question] --> QE[Embed]
        QE --> K[Top-k search<br/>in this document]
        E --> K
        K --> T{Best score<br/>≥ MIN_SCORE?}
        T -- no --> NF[Not found<br/>no LLM call]
        T -- yes --> P[Grounded prompt<br/>context + rules]
        P --> L[LLM]
        L --> S{NOT_IN_DOCUMENT?}
        S -- yes --> NF
        S -- no --> R[Answer + sources<br/>page · snippet · score]
    end
```

**Indexing.** `pypdf` extracts the text of each page separately, so every chunk knows its page number. The chunker splits each page on the coarsest boundary that works: paragraphs first, then lines, then sentences, then words. It packs the pieces into chunks of at most 800 characters, and each new chunk starts with the last ~150 characters of the previous one, so a fact that falls on a chunk boundary still appears whole in at least one chunk. Chunks are embedded with OpenAI `text-embedding-3-small` and stored in a persistent Chroma collection that uses cosine distance. Each chunk is stored with `doc_id`, `filename`, `page` and `chunk_index` metadata. The `doc_id` is a hash of the file's contents, so uploading the same file twice costs nothing.

**Answering.** The question is embedded the same way, and Chroma returns the `top_k` (default 4) most similar chunks *from that document*, each with a similarity score from 0 to 1. Chunks below the threshold are dropped. The rest go into a prompt that tells the model to answer only from those passages and to cite pages. The API returns the answer together with every passage it used.

## Handling unanswerable questions

A RAG system that invents answers is worse than none, so refusals happen in two layers:

1. **Retrieval threshold, before the LLM.** If no retrieved chunk reaches `MIN_SCORE`, the API answers *"Not found in the document."* immediately. Questions like *"What's the weather in Paris?"* never reach the model, which is faster and costs nothing. The closest passages are still returned, so you can see why.
2. **Model-level refusal, after the LLM.** Some questions *sound* like the document but aren't answered by it. *"What is the CEO's salary?"*, for example, retrieves HR text about salaries and allowances. The system prompt tells the model to use only the given context and to reply with the exact sentinel `NOT_IN_DOCUMENT` when the answer isn't there. The API maps that sentinel to `answered: false`, so clients get a boolean rather than having to parse free text.

The UI shows both cases as the same amber *"Not found in the document"* card.

## Evaluation

`eval/questions.json` holds 12 answerable questions about the sample handbook, each with its expected page and the fact the answer must contain, plus 6 unanswerable ones. Some are obviously off-topic and some sound close to the handbook's content.

```bash
python eval/run_eval.py            # full run: prints the table below and saves eval/results.md
python eval/run_eval.py --sweep    # embeddings only: how each MIN_SCORE would split the questions
```

<!-- EVAL_RESULTS -->

## Design choices

- **Plain libraries instead of LangChain or LlamaIndex.** The pipeline is roughly six function calls: extract, chunk, embed, store, search, prompt. Writing them directly keeps every step visible and debuggable, avoids framework version churn, and makes it obvious where to plug in changes such as a reranker, another vector store or another model. Each module has one job, and the two provider interfaces (`Embedder`, `AnswerModel`) are small `Protocol`s, which is also what lets the tests run without an API key.
- **800-character chunks with 150 characters of overlap.** About 150–200 tokens: large enough to hold a complete policy rule with its context, small enough that retrieval stays precise and the prompt stays cheap. Chunks never cross page boundaries, so every source has exactly one page to cite.
- **Cosine similarity with a tuned threshold.** `MIN_SCORE` is tuned on the sample with `run_eval.py --sweep`. It is set just below the lowest-scoring answerable question, so it never blocks a real answer, while still blocking clearly off-topic questions for free. Near-miss questions are left to layer 2. A threshold depends on the embedding model and the corpus, so re-run the sweep on your own documents.
- **Cheap, fast default models** (`gpt-4.1-mini`, or `claude-haiku-5-5` with `LLM_PROVIDER=anthropic`). Grounded short answers don't need a frontier model. All model names are environment variables.
- **Content-hash document IDs.** Re-uploading a file is idempotent and never re-embeds it.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests run offline with no API key. A deterministic bag-of-words fake embedder and a recording fake LLM replace the providers. They cover chunk size, overlap and page metadata; the threshold refusal (asserting the LLM is never called); the sentinel mapping to `answered=false`; the upload validation errors (wrong type, empty, corrupt, scanned PDF); and end-to-end `/ask` through the API.

## Project structure

```
app/
  main.py          FastAPI routes
  config.py        settings from environment (pydantic-settings)
  ingest.py        text extraction + chunking
  store.py         Chroma wrapper
  llm.py           embedding + answer providers (OpenAI / Anthropic)
  rag.py           retrieve → threshold → prompt → answer
  static/index.html
sample/            make_sample.py + the generated acme_handbook.pdf
eval/              questions.json + run_eval.py
tests/             pytest suite with fake providers
scripts/           screenshots.py (Playwright) for the images above
```

## Limitations and next steps

- **Scanned PDFs** are rejected with a clear message. Adding OCR (e.g. Tesseract or a cloud OCR API) at the extraction step would cover them.
- **Reranking.** A cross-encoder or a reranking API applied to the top 20 would improve precision on long, repetitive documents.
- **Hybrid search.** Combining BM25 keyword search with vectors helps with exact terms such as policy numbers, product codes and names.
- **Multi-document questions.** Search is currently scoped to one document. Dropping the `doc_id` filter, or filtering on a set of IDs, enables corpus-wide Q&A.
- **Auth and multi-tenancy.** There is no authentication. A real deployment needs API keys or OAuth, and per-tenant collections.
- **Deployment.** For production: a managed vector database (or Chroma server), async background ingestion for large files, streaming answers, and request logging and metrics.

## License

MIT
