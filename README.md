# Multilingual RAG (Gemini + Qdrant + Cohere + Groq)

A simple, general-purpose RAG app. Ingest PDFs/TXT/DOCX in any language,
ask a question in any language, get an answer grounded only in your
documents — in the language of your choice (auto-detected by default).

Everything runs through free-tier APIs. No local model inference; your
machine only runs the Streamlit app and lightweight text processing
(chunking, language detection).

## Architecture

```
Ingestion:  file -> extract text -> chunk (~500 tokens, 50 overlap)
            -> Gemini embedding (768d) -> Qdrant Cloud

Query:      question -> detect language -> Gemini embedding (768d)
            -> Qdrant dense search (top 15) -> Cohere rerank (top 4)
            -> Groq LLM -> answer in chosen language
```

One Qdrant collection holds chunks from every language — no per-language
collections. Every point's payload carries `document_id`, `filename`,
`file_type`, `language`, `chunk_index`, `text`, `source`, `page`.

## Setup (using `uv`)

1. **Install dependencies and create the virtual environment:**
   ```bash
   cd multilang-rag
   uv sync
   ```
   This reads `pyproject.toml`, creates `.venv/`, and installs everything.

2. **Get your free-tier API keys:**
   - Gemini: https://aistudio.google.com/apikey
   - Qdrant Cloud (free cluster): https://cloud.qdrant.io
   - Cohere: https://dashboard.cohere.com/api-keys
   - Groq: https://console.groq.com/keys

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # then edit .env and paste in your four API keys / Qdrant URL
   ```

4. **Add documents to ingest:**
   Put your `.pdf` / `.txt` / `.docx` files (any mix of languages) into
   `data/raw/` (subfolders are fine — they're scanned recursively).

5. **Run the app:**
   ```bash
   uv run streamlit run app.py
   ```
   Open the URL Streamlit prints (usually http://localhost:8501),
   click **Run ingestion** in the sidebar once, then ask questions.

You can also run ingestion from the command line instead of the UI button:
```bash
uv run python -m ingestion.pipeline
```

## Notes / design choices

- **Same embedding model & dimensionality for docs and queries** —
  `gemini-embedding-001` at 768 dimensions everywhere (`retrieval/embedder.py`).
  Gemini's docs require L2-normalizing embeddings when using a truncated
  (non-3072) dimensionality for correct cosine similarity — handled for you.
- **No hybrid search** — dense retrieval (Qdrant) + Cohere rerank only, as
  requested, keeping the pipeline simple.
- **Dedup on ingestion** — each file is hashed (SHA-256); re-running
  ingestion skips files already present in Qdrant.
- **No conversation memory** — every question is answered independently;
  there's no chat history sent to the LLM.
- **Language handling** — `langdetect` (local, no API call) detects the
  query language for the default answer language; the dropdown lets you
  override it to answer in a different language than the question was
  asked in. The LLM is instructed to answer only from retrieved context
  and to say so explicitly if the answer isn't in the documents.
- All tunables (chunk size/overlap, candidate/final counts, model names,
  collection name) live in one place: `config.py` / `.env`.

## Project layout

```
config.py                  # all settings, loaded from .env
app.py                     # Streamlit UI
ingestion/
  loaders.py                # pdf / docx / txt -> text blocks (+ page numbers)
  chunking.py                # recursive, token-based chunker
  pipeline.py                 # discover -> extract -> chunk -> embed -> upsert
retrieval/
  language.py                # detect_language, dropdown options
  embedder.py                 # Gemini embed (documents + queries)
  qdrant_store.py              # collection mgmt, upsert, search, dedup
  reranker.py                   # Cohere rerank
  generator.py                   # Groq answer generation
data/raw/                  # put your source documents here
```
