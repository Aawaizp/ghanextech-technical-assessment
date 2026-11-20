
"""Document collections owned by a single Streamlit session."""

import logging
import weakref
from uuid import uuid4

import chromadb
from chromadb.config import Settings
import streamlit as st
from langchain_chroma import Chroma

from modules.embeddings import get_embedding_model

_INDEX_KEY = "_doctok_document_index"
_VERSION_KEY = "_doctok_session_storage_version"
_LOGGER = logging.getLogger(__name__)


def _release_abandoned_collection(client, collection_name):
    """Best-effort cleanup after Streamlit releases a disconnected session."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        _LOGGER.warning("Could not release an abandoned document collection.", exc_info=True)


class _SessionIndex:
    def __init__(self, embedding_model):
        # Ephemeral clients can share a backend. A random collection per session
        # is therefore essential; merely creating another client is not isolation.
        self.collection_name = f"doctok-{uuid4().hex}"
        self.client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        self.store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._cleanup = weakref.finalize(
            self, _release_abandoned_collection, self.client, self.collection_name
        )

    def close(self):
        self.store.delete_collection()
        self._cleanup.detach()


def clear_document_session():
    """Delete only this session's collection and reset its document/chat state."""
    index = st.session_state.get(_INDEX_KEY)
    if index is not None:
        index.close()
        del st.session_state[_INDEX_KEY]
    st.session_state["messages"] = []
    st.session_state["processed_versions"] = {}
    st.session_state["selected_files"] = {}
    st.session_state["document_stats"] = {}
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state.pop("upload_results", None)
    st.session_state.pop("processed_files", None)


def initialize_document_session():
    """Also discard legacy chat references to the formerly shared collection."""
    if st.session_state.get(_VERSION_KEY) != 1:
        clear_document_session()
        st.session_state[_VERSION_KEY] = 1


def get_vector_store():
    """Reuse a session-owned store; never cache document data across sessions."""
    initialize_document_session()
    index = st.session_state.get(_INDEX_KEY)
    if index is not None:
        return index.store
    try:
        index = _SessionIndex(get_embedding_model())
    except Exception as exc:
        raise RuntimeError(
            "Could not create this session's document index. "
            "Check the application logs for the underlying error."
        ) from exc
    st.session_state[_INDEX_KEY] = index
    return index.store

def store_documents(chunks):
    """Index one PDF version, then remove outdated chunks for its filename."""
    if not chunks:
        raise ValueError(
            "This PDF has no readable text. "
            "If it is scanned, run OCR before uploading it."
        )

    filename = chunks[0].metadata["filename"]
    document_hash = chunks[0].metadata["document_hash"]

    # This function processes one document version at a time.
    for chunk in chunks:
        if (
            chunk.metadata["filename"] != filename
            or chunk.metadata["document_hash"] != document_hash
        ):
            raise ValueError("Expected chunks from one document version.")

    db = get_vector_store()

    existing = db.get(
        where={"filename": filename},
        include=["metadatas"],
    )
    existing_ids = set(existing["ids"])

    current_ids = {
        chunk.metadata["chunk_id"]
        for chunk in chunks
    }

    new_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["chunk_id"] not in existing_ids
    ]

    # Store the new content before deleting any previous content.
    if new_chunks:
        db.add_documents(
            documents=new_chunks,
            ids=[chunk.metadata["chunk_id"] for chunk in new_chunks],
        )

    outdated_ids = list(existing_ids - current_ids)

    if outdated_ids:
        db.delete(ids=outdated_ids)

    return len(new_chunks)


def list_indexed_filenames():
    initialize_document_session()
    index = st.session_state.get(_INDEX_KEY)
    if index is None:
        return []
    data = index.store.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    return sorted({m.get("filename") for m in metadatas if m and m.get("filename")})


def delete_file(filename: str):
    """Remove every chunk belonging to `filename` from the Chroma collection."""
    index = st.session_state.get(_INDEX_KEY)
    if index is None:
        return
    ids = index.store.get(where={"filename": filename})["ids"]
    if ids:
        index.store.delete(ids=ids)
