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
Preprocessing joins wrapped lines while preserving blank-line paragraph breaks,
so the recursive splitter can prefer paragraph boundaries. Page boundaries are
kept for source references.

The 500-character chunk size keeps passages focused. A 100-character overlap
retains nearby context when a sentence or idea crosses a chunk boundary, while
limiting duplicate text sent to the embedding model. These are starting values
that should be tuned against different PDFs.

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`: it runs locally,
requires no embedding API key, and is small enough for this demo. The first upload
downloads and loads model weights; Streamlit then caches that model for subsequent
uploads and questions. Groq is used only for answer generation. Short questions
go directly to vector retrieval without a separate query-rewrite LLM call.

Ordinary questions retrieve up to 8 passages and keep cosine relevance scores of
at least 0.35. This threshold is a starting heuristic, not a confidence estimate.
Broad summary requests sample up to 8 chunks across the selected files instead;
they do not read every chunk. Answers are buffered until citation syntax and
passage numbers have been checked, then displayed with the source passages.

## Run locally

Open the folder containing `app.py` and `requirements.txt`: `task-1-rag` in the
assessment repository, or `Desktop/doctok` in the original working copy.
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

The suite covers PDF errors, paragraph boundaries, upload cleanup, citation
validation, session isolation, replacement/deletion, retrieval recovery and file
scope, and chat error handling. A separate live smoke
test uses synthetic documents and your configured Groq key:

```bash
python -B scripts/evaluate_live.py
```

That command sends synthetic test text to Groq and makes live API calls. MiniLM
downloads on first use if it is not already cached. The test is not part of the
offline suite and does not measure answer quality on a representative PDF set.

Local verification on 2026-11-20: 34 offline tests and all 3 live smoke cases
passed using Python 3.11.6. The live cases checked two conflicting dates in
separately selected PDFs and refusal for an absent budget. `pip check` reported
no broken requirements in the existing environment; a fresh hosting build has
not yet been verified.

## Deployment

**Live demo:** [Ghanex PDF assistant](https://ghanextech-technical-assessment-anslrabcywwrbm4mm9tpqz.streamlit.app/)

In Streamlit Community Cloud, select the assessment repository, its `main`
branch, and `task-1-rag/app.py` as the entry point. Choose Python 3.11 in Advanced
settings and add your key to the Secrets field in TOML format:

```toml
GROQ_API_KEY_1 = "your_groq_api_key"
```

The repository-root `.streamlit/config.toml` supplies the theme and upload limit
when the app runs from the repository root. The app folder has the same config
for running locally from that folder. Dependencies are in
`task-1-rag/requirements.txt`.

## Limitations

- Image-only pages require OCR before upload.
- PDF text extraction may lose table structure or multi-column reading order.
- Broad summaries use at most 8 sampled chunks and may omit details or entire
  files when more files are selected than the passage budget allows.
- Each question is retrieved independently. The chat history is displayed but
  is not passed to the model; ask self-contained questions rather than "why?".
- Citation validation checks citation syntax and passage-number bounds; it cannot
  prove that every claim is entailed by its cited passage. Open sources to verify
  important claims.
- Documents and conversation are in memory for one browser session. Refreshing
  starts a new session; this demo does not provide accounts or durable storage.
- Questions and retrieved excerpts are sent to Groq for answer generation.
- Chroma 1.0.20 has intermittently returned `Error finding id` during retrieval.
  For that specific error, the app computes exact cosine similarity over stored
  vectors from the selected files. This keeps the selection and relevance
  threshold intact, but is slower for large collections. It is a recovery path,
  not a repair of the underlying Chroma index. Other database errors are surfaced.
- Groq availability and account rate limits can interrupt answers; the UI keeps
  the question and documents so the request can be retried.

## Development background

This assessment version builds on Aawaiz's earlier
[DocTOK project](https://github.com/Aawaizp/DocTOK). ChatGPT/Codex assisted with
debugging, the interface, refactoring, and automated tests. It is an extension
of that project, and the AI assistance should also be disclosed in the final
submission form. The candidate's own code review and walkthrough are still
required before submission.

## Stack

Python, Streamlit, LangChain, ChromaDB, Sentence Transformers, PyPDF, and Groq.
All direct dependencies are version-pinned in `requirements.txt`.
