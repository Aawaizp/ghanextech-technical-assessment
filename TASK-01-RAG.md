# Task 1: Document Assistant Using Retrieval-Augmented Generation (RAG)

## Overview

In this assignment, you will build a document question-answering assistant that allows users to upload PDF files and query them using natural language.

The goal is to test your understanding of core RAG fundamentals: document ingestion, text chunking, vector embeddings, vector indexing, semantic search, prompt construction, and grounded generation.

---

## Technical Requirements

Your application must implement the following pipeline:

### 1. Document Ingestion and Text Extraction
- Accept one or more PDF files via an upload component.
- Extract plain text while handling multi-page layouts cleanly.
- Gracefully handle scanned/empty documents, password-protected files, or unsupported formats by displaying informative error messages to the user.

### 2. Chunking and Preprocessing
- Split extracted text into logical passages or chunks (e.g., using recursive character splitting).
- Choose appropriate chunk sizes and overlap values to preserve contextual continuity without bloating embedding requests. Document your reasoning for your chosen parameters.

### 3. Embeddings and Vector Storage
- Convert text chunks into dense vector embeddings using an embedding model (e.g., Hugging Face embeddings, Google Generative AI embeddings, or alternative free models).
- Store embeddings in an in-memory or lightweight vector store such as ChromaDB or FAISS.
- Ensure the vector index refreshes appropriately when a new document is loaded.

### 4. Semantic Retrieval and Context Building
- When a user submits a question, generate its query embedding.
- Retrieve the top-$k$ most relevant text chunks using cosine similarity or Euclidean distance.
- Include source references (such as document name, chunk index, or page number) alongside retrieved passages.

### 5. Grounded Generation
- Pass retrieved context and the user query to an LLM with a system prompt instructing the model to answer based strictly on the provided context.
- The model should state clearly when the answer is not present in the document rather than hallucinating details.

### 6. User Interface
- Build an interactive web interface using Streamlit, Gradio, or a lightweight frontend connected to a FastAPI backend.
- Display the conversation history or query results, the generated response, and citations/source references.

---

## Suggested Stack

You are encouraged to use free or open-source components:

- **Language:** Python 3.10+
- **Framework:** Streamlit / Gradio / FastAPI
- **Orchestration:** LangChain, LlamaIndex, or vanilla Python
- **Embedding Models:** Sentence-Transformers (e.g., `all-MiniLM-L6-v2`), Hugging Face Inference API, or Google Gemini Embeddings
- **Vector Database:** ChromaDB or FAISS
- **LLM:** Google Gemini API (free tier), Groq API (free tier), or locally hosted models (e.g., Ollama)

You may substitute any technology as long as you document your architectural choices and rationale in your README.

---

## Deliverables

1. **Source Code:** Complete, clean, and well-structured source code in a GitHub repository.
2. **Requirements File:** A `requirements.txt` or `pyproject.toml` listing all dependencies with version pins.
3. **Documentation:** A comprehensive `README.md` covering:
   - System architecture and RAG workflow diagram/explanation.
   - Chunking strategy and embedding model selection rationale.
   - Setup and execution instructions for local testing.
   - Known limitations and areas for improvement.
4. **Live Deployment:** A deployed, publicly accessible demo hosted on Streamlit Community Cloud, Hugging Face Spaces, or similar free platforms.
