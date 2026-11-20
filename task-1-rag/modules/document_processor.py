import hashlib
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from modules.pdf_loader import load_pdf
from modules.text_chunker import chunk_documents
from modules.vector_store import store_documents
from config import MAX_UPLOAD_BYTES



def process_pdf(uploaded_file):
    """Parse a temporary upload and index it in the current session only."""
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("This file is empty. Upload a PDF with readable text.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("This PDF exceeds the 20 MB limit. Upload a smaller file.")
    document_hash = hashlib.sha256(file_bytes).hexdigest()
    filename = PureWindowsPath(uploaded_file.name).name
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Upload a PDF file.")

    # Never use an uploaded filename as a filesystem path. Each call gets a
    # separate temporary directory, removed on success and on parsing failure.
    with TemporaryDirectory(prefix="doctok-upload-") as temporary_directory:
        file_path = Path(temporary_directory) / "document.pdf"
        file_path.write_bytes(file_bytes)
        documents = load_pdf(str(file_path))

    # Keep user-facing citations meaningful after the temporary file is gone.
    for document in documents:
        document.metadata["source"] = filename
    chunks = chunk_documents(documents, document_hash)
    added = store_documents(chunks)
    empty_pages = sum(not document.page_content.strip() for document in documents)

    return {
        "filename": filename,
        "document_hash": document_hash,
        "pages": len(documents),
        "chunks": len(chunks),
        "indexed": added,
        "warnings": [
            f"{empty_pages} page(s) have no readable text and aren't included in search."
        ] if empty_pages else [],
    }

def process_pdfs(uploaded_files):
    """
    Save, load, chunk and index multiple PDFs.

    Each file is processed independently via process_pdf, so a failure
    on one file doesn't prevent the others from indexing. Returns a list
    of per-file result dicts in the same order as uploaded_files, with an
    "error" key set instead of the usual fields if that file failed.
    """

    results = []

    for uploaded_file in uploaded_files:
        try:
            info = process_pdf(uploaded_file)
        except Exception as e:
            info = {
                "filename": uploaded_file.name,
                "error": str(e)
            }
        results.append(info)

    return results
