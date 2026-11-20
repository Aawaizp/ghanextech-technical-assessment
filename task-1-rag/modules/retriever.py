import logging
import math
import os

from chromadb.errors import InternalError
from langchain_core.documents import Document

from config import TOP_K
from modules.vector_store import get_vector_store

SCORE_THRESHOLD = 0.35
_LOGGER = logging.getLogger(__name__)

SUMMARY_KEYWORDS = (
    "summarize", "summary", "key points", "main points",
    "overview", "main idea", "tl;dr", "what is this document about",
    "what is this about", "outline", "table of contents",
    "explain the document", "what does this cover",
)

DEBUG = os.getenv("DOCTOK_DEBUG", "false").lower() == "true"


def _build_filename_filter(filenames):
    if not filenames:
        return None
    filenames = list(filenames)
    if len(filenames) == 1:
        return {"filename": filenames[0]}
    return {"filename": {"$in": filenames}}


def _is_summary_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SUMMARY_KEYWORDS)


def _get_summary_chunks(filenames, k):
    db = get_vector_store()
    where = _build_filename_filter(filenames)

    data = db.get(where=where, include=["metadatas", "documents"])
    metadatas = data.get("metadatas") or []
    documents = data.get("documents") or []

    by_file = {}
    for meta, text in zip(metadatas, documents):
        fname = meta.get("filename", "unknown")
        by_file.setdefault(fname, []).append((meta, text))

    for fname in by_file:
        by_file[fname].sort(key=lambda item: item[0].get("chunk_index", 0))

    files = sorted(by_file)
    if not files:
        return []
    per_file, remainder = divmod(k, len(files))

    results = []
    for file_index, fname in enumerate(files):
        passages = by_file[fname]
        count = min(per_file + (file_index < remainder), len(passages))
        indices = (
            [round(i * (len(passages) - 1) / (count - 1)) for i in range(count)]
            if count > 1 else list(range(count))
        )
        for index in indices:
            meta, text = passages[index]
            results.append(Document(page_content=text, metadata=meta))

    return results


def _exact_cosine_search(db, query, k, where):
    """Recover from Chroma's HNSW ID error without widening the file scope."""
    data = db.get(where=where, include=["documents", "metadatas", "embeddings"])
    if not data["ids"]:
        return []
    query_vector = db.embeddings.embed_query(query)
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    if not query_norm:
        return []

    ranked = []
    for text, metadata, vector in zip(
        data["documents"], data["metadatas"], data["embeddings"], strict=True
    ):
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if not norm:
            continue
        score = sum(
            float(a) * float(b) for a, b in zip(query_vector, vector, strict=True)
        ) / (query_norm * norm)
        ranked.append((Document(page_content=text, metadata=metadata), score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def retrieve_documents(
    query: str,
    k: int | None = None,
    score_threshold: float = SCORE_THRESHOLD,
    filenames: list[str] | None = None,
):
    if k is None:
        k = TOP_K
    if k < 1 or not query.strip() or filenames == []:
        return []

    if _is_summary_query(query):
        docs = _get_summary_chunks(filenames, k=k)
        if DEBUG:
            print(f"[retrieval:summary] {len(docs)} chunks | filter={filenames or 'ALL'}")
        return docs

    db = get_vector_store()
    where_filter = _build_filename_filter(filenames)

    try:
        results = db.similarity_search_with_relevance_scores(
            query=query, k=k, filter=where_filter,
        )
    except InternalError as exc:
        if "Error finding id" not in str(exc):
            raise
        _LOGGER.warning("Chroma ID lookup failed; using exact cosine search for this request.")
        results = _exact_cosine_search(db, query, k, where_filter)

    filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]

    if DEBUG:
        print(f"[retrieval] query={query!r} | filter={filenames or 'ALL'} "
              f"| {len(filtered_results)}/{len(results)} passed threshold")
        for doc, score in filtered_results:
            print(f"    p.{doc.metadata.get('page', 0) + 1} score={score:.3f} {doc.metadata.get('filename')}")

    if filtered_results:
        return [doc for doc, score in filtered_results]

    if DEBUG:
        print("[retrieval] no chunks passed the relevance threshold")

    return []
