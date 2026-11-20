# Ghanex — PDF document assistant

Ghanex lets you ask questions across multiple PDFs. It retrieves relevant passages,
asks Groq's `openai/gpt-oss-120b` to answer from those passages, and shows numbered
citations with document names and page numbers.

## RAG workflow

```text
PDF files → page text extraction → 500-character chunks (100 overlap)
          → all-MiniLM-L6-v2 embeddings → session-owned Chroma (cosine search)
Question → same embedding model → top relevant passages → Groq grounded answer
          → checked passage references and source pages
```

Each uploaded PDF is parsed independently, so one invalid file does not prevent
other valid PDFs in the same upload from indexing. Chunks retain document, page,
file fingerprint, and chunk metadata. Replacing a file refreshes its chunks.
Each browser session has its own in-memory Chroma collection.

The 500-character chunk size keeps passages focused. A 100-character overlap
retains nearby context when a sentence or idea crosses a chunk boundary, while
limiting duplicate text sent to the embedding model. These are starting values
that should be tuned against different PDFs.

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`: it runs locally,
requires no embedding API key, and is small enough for this demo. The first upload
downloads and loads model weights; Streamlit then caches that model for subsequent
uploads and questions. Groq is used only for answer generation. Short questions
go directly to vector retrieval without a separate query-rewrite LLM call.

## Run locally

Use Python 3.11 (Python 3.10+ is the assessment minimum):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` with your own Groq key:

```env
GROQ_API_KEY_1=your_groq_api_key
```

Alternatively, copy `.env.example` to `.env` and replace its placeholder.
Never commit `.env` or a real API key.

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload text-based PDFs, choose which
documents should be searched, and ask a question. Scanned PDFs need OCR before
upload. Password-protected, empty, damaged, and oversized PDFs receive an error
or warning. Each file is limited to 20 MB.

## Tests

Run the offline checks without Groq calls or model downloads:

```bash
python -m unittest discover -s tests -v
```

The suite covers PDF errors, upload cleanup, citation validation, session
isolation, replacement/deletion, and chat error handling. A separate live smoke
test uses synthetic documents and your configured Groq key:

```bash
python -B scripts/evaluate_live.py
```

That command sends synthetic test text to Groq and makes live API calls. It is
not part of the offline suite.

## Deployment

The Streamlit app entry point is `task-1-rag/app.py` from the assessment
repository root. Configure `GROQ_API_KEY_1` in the hosting provider's secrets
settings and point the app to this subdirectory. Deployment must be verified
after the assessment repository is published; no public demo URL is available
yet.

## Limitations

- Image-only pages require OCR before upload.
- Broad summaries use a sample of document chunks and may omit details.
- Citation validation checks citation syntax and passage-number bounds; it cannot
  prove that every claim is entailed by its cited passage. Open sources to verify
  important claims.
- Documents and conversation are in memory for one browser session. Refreshing
  starts a new session; this demo does not provide accounts or durable storage.
- Questions and retrieved excerpts are sent to Groq for answer generation.
- An intermittent filtered-search error has been observed with Chroma 1.0.20.
  The offline suite has passed on reruns, but this backend behavior still needs
  further investigation.

## Stack

Python, Streamlit, LangChain, ChromaDB, Sentence Transformers, PyPDF, and Groq.
All direct dependencies are version-pinned in `requirements.txt`.
