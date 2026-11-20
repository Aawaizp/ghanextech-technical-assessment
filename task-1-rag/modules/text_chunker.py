import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.text_cleaner import clean_text

from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_documents(documents, document_hash):
    """Split PDF pages and attach source and document-version metadata."""

    for document in documents:
        document.page_content = clean_text(document.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")
        filename = os.path.basename(source)
        page = chunk.metadata.get("page", 0)

        chunk.metadata["filename"] = filename
        chunk.metadata["document_hash"] = document_hash
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = (
            f"{filename}:{document_hash}:{page}:{index}"
        )

    return chunks