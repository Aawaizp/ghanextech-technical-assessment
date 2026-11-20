"""Offline integration checks using real Chroma and separate Streamlit sessions."""
import gc
import hashlib
from contextlib import contextmanager
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

from langchain_core.embeddings import Embeddings
from streamlit.testing.v1 import AppTest

from modules import vector_store


@contextmanager
def stub_generation(module):
    # Do not use patch.dict(sys.modules): it removes native Chroma modules
    # imported during the test, which cannot be initialized twice by PyO3.
    missing = object()
    previous = sys.modules.get("modules.rag_chain", missing)
    sys.modules["modules.rag_chain"] = module
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("modules.rag_chain", None)
        else:
            sys.modules["modules.rag_chain"] = previous


class FixedEmbeddings(Embeddings):
    """Small deterministic vectors: no model download or network."""
    def embed_documents(self, texts):
        return [[1.0, float("alpha" in text), float("beta" in text)] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


SESSION_APP = """
import hashlib
import streamlit as st
from langchain_core.documents import Document
from modules.retriever import retrieve_documents
from modules.vector_store import (
    initialize_document_session, get_vector_store, list_indexed_filenames,
    store_documents, delete_file, clear_document_session,
)

initialize_document_session()
if st.button("Index"):
    text = st.session_state["upload_text"]
    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    store_documents([Document(page_content=text, metadata={
        "filename": "same.pdf", "page": 0, "document_hash": fingerprint,
        "chunk_id": f"same.pdf:{fingerprint}:0:0", "chunk_index": 0,
    })])
    st.session_state.processed_versions["same.pdf"] = fingerprint
if st.button("Delete"):
    delete_file("same.pdf")
if st.button("Clear"):
    clear_document_session()

st.session_state["visible_files"] = list_indexed_filenames()
st.session_state["visible_texts"] = []
st.session_state["search_texts"] = []
st.session_state["filtered_texts"] = []
if st.session_state["visible_files"]:
    db = get_vector_store()
    st.session_state["visible_texts"] = db.get(include=["documents"])["documents"]
    st.session_state["search_texts"] = [
        d.page_content for d in retrieve_documents("alpha beta", k=1, score_threshold=0)
    ]
    st.session_state["filtered_texts"] = [
        d.page_content for d in retrieve_documents(
            "alpha beta", k=1, score_threshold=0, filenames=["same.pdf"]
        )
    ]
"""


class SessionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.embedding_patch = patch(
            "modules.vector_store.get_embedding_model", return_value=FixedEmbeddings()
        )
        self.embedding_patch.start()
        self.apps = []

    def tearDown(self):
        for app in self.apps:
            index = app.session_state.filtered_state.get("_doctok_document_index")
            if index is not None:
                index.close()
                del app.session_state["_doctok_document_index"]
        self.embedding_patch.stop()

    def new_session(self):
        app = AppTest.from_string(SESSION_APP).run(timeout=30)
        self.assertFalse(app.exception)
        self.apps.append(app)
        return app

    def upload(self, app, text):
        app.session_state["upload_text"] = text
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)

    def assert_content(self, app, expected):
        self.assertEqual(app.session_state["visible_files"], ["same.pdf"])
        for key in ("visible_texts", "search_texts", "filtered_texts"):
            self.assertEqual(app.session_state[key], [expected])

    def test_same_filename_search_and_reruns_are_isolated(self):
        first, second = self.new_session(), self.new_session()
        self.upload(first, "alpha private document")
        self.assertEqual(second.session_state["visible_files"], [])
        self.upload(second, "beta private document")
        self.assert_content(first, "alpha private document")
        self.assert_content(second, "beta private document")
        first_index = first.session_state["_doctok_document_index"]
        self.assertFalse(first_index.client.get_settings().is_persistent)
        self.assertNotEqual(
            first_index.collection_name,
            second.session_state["_doctok_document_index"].collection_name,
        )
        first.run(timeout=30)
        self.assertIs(first.session_state["_doctok_document_index"], first_index)
        self.assert_content(first, "alpha private document")

    def test_duplicate_replace_delete_and_clear_do_not_touch_other_session(self):
        first, second = self.new_session(), self.new_session()
        self.upload(first, "alpha old")
        self.upload(second, "beta unchanged")
        self.upload(first, "alpha old")
        self.assert_content(first, "alpha old")
        self.upload(first, "alpha replacement")
        self.assert_content(first, "alpha replacement")
        second.run(timeout=30)
        self.assert_content(second, "beta unchanged")
        first.button[1].click().run(timeout=30)
        self.assertFalse(first.exception)
        self.assertEqual(first.session_state["visible_files"], [])
        second.run(timeout=30)
        self.assert_content(second, "beta unchanged")
        self.upload(first, "alpha again")
        first.session_state["messages"] = [{"role": "assistant", "content": "old answer"}]
        first.button[2].click().run(timeout=30)
        self.assertFalse(first.exception)
        self.assertEqual(first.session_state["messages"], [])
        self.assertEqual(first.session_state["processed_versions"], {})
        self.assertEqual(first.session_state["visible_files"], [])
        second.run(timeout=30)
        self.assert_content(second, "beta unchanged")

    def test_identical_pdf_bytes_still_index_in_each_session(self):
        first, second = self.new_session(), self.new_session()
        self.upload(first, "alpha identical")
        self.upload(second, "alpha identical")
        first.button[2].click().run(timeout=30)
        second.run(timeout=30)
        self.assert_content(second, "alpha identical")

    def test_failed_replacement_keeps_existing_document(self):
        app = self.new_session()
        self.upload(app, "alpha original")
        store = app.session_state["_doctok_document_index"].store
        with patch.object(store, "add_documents", side_effect=RuntimeError("embedding failed")):
            app.session_state["upload_text"] = "alpha replacement"
            app.button[0].click().run(timeout=30)
            self.assertTrue(app.exception)
        app.run(timeout=30)
        self.assert_content(app, "alpha original")

    def test_new_session_discards_legacy_shared_history(self):
        app = AppTest.from_string(SESSION_APP)
        app.session_state["messages"] = [{"content": "legacy shared document"}]
        app.session_state["processed_files"] = {"legacy.pdf"}
        app.session_state["selected_files"] = {"legacy.pdf": True}
        app.run(timeout=30)
        self.apps.append(app)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["messages"], [])
        self.assertEqual(app.session_state["selected_files"], {})
        self.assertEqual(app.session_state["visible_files"], [])

    def test_releasing_session_owner_cleans_collection(self):
        index = vector_store._SessionIndex(FixedEmbeddings())
        name, client = index.collection_name, index.client
        owner = weakref.ref(index)
        del index
        gc.collect()
        self.assertIsNone(owner())
        self.assertNotIn(name, [collection.name for collection in client.list_collections()])


class AppStartupTests(unittest.TestCase):
    def test_real_app_upload_replace_delete_and_clear(self):
        from io import BytesIO
        from types import SimpleNamespace
        from pypdf import PdfWriter
        from pypdf.generic import (
            DecodedStreamObject, DictionaryObject, NameObject,
        )

        def pdf_upload(text):
            # A tiny real PDF fixture exercises PyPDFLoader and chunk metadata.
            writer = PdfWriter()
            page = writer.add_blank_page(width=612, height=792)
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({
                    NameObject("/F1"): writer._add_object(font)
                })
            })
            content = DecodedStreamObject()
            content.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(content)
            buffer = BytesIO()
            writer.write(buffer)
            data = buffer.getvalue()
            return SimpleNamespace(name="notes.pdf", getvalue=lambda: data)

        fake_rag = SimpleNamespace(
            ask_llm_stream=lambda *args, **kwargs: (None, []),
            NOT_FOUND_MESSAGE="I couldn't find that information in the uploaded documents.",
        )
        with stub_generation(fake_rag), patch(
            "modules.vector_store.get_embedding_model", return_value=FixedEmbeddings()
        ):
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[1] / "app.py")
            ).run(timeout=30)
            self.assertFalse(app.exception)
            old_pdf = pdf_upload("alpha original document")
            new_pdf = pdf_upload("alpha changed document")
            try:
                for uploaded in (old_pdf, old_pdf, new_pdf):
                    # Streamlit 1.49 AppTest has no upload setter. Supply only the
                    # widget value; the app and entire ingestion/storage path run.
                    with patch("streamlit.file_uploader", side_effect=[[uploaded], []]):
                        app.run(timeout=30)
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    store = app.session_state["_doctok_document_index"].store
                    rows = store.get(include=["documents", "metadatas"])
                    self.assertEqual(len(rows["ids"]), 1)
                    self.assertEqual(rows["metadatas"][0]["filename"], "notes.pdf")
                self.assertEqual(rows["documents"], ["alpha changed document"])
                app.checkbox[0].uncheck().run(timeout=30)
                self.assertFalse(app.exception)
                self.assertTrue(app.chat_input[0].disabled)
                app.checkbox[0].check().run(timeout=30)
                self.assertFalse(app.chat_input[0].disabled)
                self.assertEqual(
                    app.session_state["processed_versions"]["notes.pdf"],
                    hashlib.sha256(new_pdf.getvalue()).hexdigest(),
                )
                next(b for b in app.button if b.key == "delete_notes.pdf").click().run(timeout=30)
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["processed_versions"], {})
                self.assertEqual(store.get()["ids"], [])

                with patch("streamlit.file_uploader", side_effect=[[old_pdf], []]):
                    app.run(timeout=30)
                self.assertFalse(app.exception)
                app.session_state["messages"] = [{"role": "assistant", "content": "old answer"}]
                next(
                    b for b in app.button if b.label == "Clear documents and chat"
                ).click().run(timeout=30)
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["messages"], [])
                self.assertEqual(app.session_state["processed_versions"], {})
                self.assertTrue(any("Upload one or more PDFs" in item.value for item in app.markdown))
            finally:
                index = app.session_state.filtered_state.get("_doctok_document_index")
                if index is not None:
                    index.close()
                    del app.session_state["_doctok_document_index"]

    def test_real_app_starts_empty_without_loading_embeddings(self):
        from types import SimpleNamespace
        fake_rag = SimpleNamespace(
            ask_llm_stream=lambda *args, **kwargs: (None, []),
            NOT_FOUND_MESSAGE="I couldn't find that information in the uploaded documents.",
        )
        with stub_generation(fake_rag), patch(
            "modules.vector_store.get_embedding_model"
        ) as embedding_loader:
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[1] / "app.py")
            ).run(timeout=30)
        self.assertFalse(app.exception)
        embedding_loader.assert_not_called()
        self.assertTrue(app.chat_input[0].disabled)
        labels = {button.label for button in app.button}
        self.assertTrue(labels.isdisjoint({
            "Summarize this document", "What are the key points?", "List the main steps"
        }))
        self.assertFalse(any("Try asking" in item.value for item in app.markdown))
        clear_button = next(
            button for button in app.button if button.label == "Clear documents and chat"
        )
        self.assertTrue(clear_button.disabled)
        self.assertTrue(any("Upload one or more PDFs" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
