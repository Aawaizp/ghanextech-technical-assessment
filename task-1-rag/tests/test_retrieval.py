"""Exercise retrieval and its recovery path against real stored vectors."""
import unittest
from unittest.mock import patch

from chromadb.errors import InternalError
from langchain_core.documents import Document

from modules import retriever
from modules.vector_store import _SessionIndex
from test_session_isolation import FixedEmbeddings


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = _SessionIndex(FixedEmbeddings())
        self.addCleanup(self.index.close)
        self.db = self.index.store
        self.store_patch = patch.object(retriever, "get_vector_store", return_value=self.db)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        fixtures = [
            ("selected.pdf", "alpha selected", 0),
            ("selected.pdf", "beta selected", 1),
            ("other.pdf", "alpha unselected", 0),
        ]
        self.db.add_documents([
            Document(page_content=text, metadata={"filename": filename, "page": page,
                                                  "chunk_index": page})
            for filename, text, page in fixtures
        ], ids=["selected-alpha", "selected-beta", "other-alpha"])

    def test_id_error_recovers_with_correct_ranking_and_file_scope(self):
        with patch.object(self.db, "similarity_search_with_relevance_scores",
                          side_effect=InternalError("Error finding id")):
            docs = retriever.retrieve_documents("alpha", k=2, filenames=["selected.pdf"])
        self.assertEqual([d.page_content for d in docs], ["alpha selected", "beta selected"])
        self.assertTrue(all(d.metadata["filename"] == "selected.pdf" for d in docs))
        self.assertEqual(docs[0].metadata["page"], 0)

    def test_recovery_respects_threshold_top_k_and_missing_file(self):
        with patch.object(self.db, "similarity_search_with_relevance_scores",
                          side_effect=InternalError("Error finding id")):
            docs = retriever.retrieve_documents("alpha", k=2, score_threshold=0.9,
                                                filenames=["selected.pdf"])
            self.assertEqual([d.page_content for d in docs], ["alpha selected"])
            docs = retriever.retrieve_documents("alpha", k=1, score_threshold=0,
                                                filenames=["selected.pdf"])
            self.assertEqual([d.page_content for d in docs], ["alpha selected"])
            self.assertEqual(retriever.retrieve_documents("alpha", filenames=["missing.pdf"]), [])

    def test_unrelated_database_failure_is_not_hidden(self):
        with patch.object(self.db, "similarity_search_with_relevance_scores",
                          side_effect=InternalError("storage unavailable")):
            with self.assertRaisesRegex(InternalError, "storage unavailable"):
                retriever.retrieve_documents("alpha", filenames=["selected.pdf"])

    def test_empty_selection_never_searches_all_files(self):
        with patch.object(retriever, "get_vector_store") as get_store:
            for query in ("alpha", "summarize the documents"):
                self.assertEqual(retriever.retrieve_documents(query, filenames=[]), [])
            get_store.assert_not_called()

    def test_summary_respects_requested_budget_and_selected_files(self):
        docs = retriever.retrieve_documents("summarize the documents", k=1,
                                            filenames=["selected.pdf", "other.pdf"])
        self.assertEqual(len(docs), 1)
        docs = retriever.retrieve_documents("summarize the document", k=2,
                                            filenames=["selected.pdf"])
        self.assertEqual(len(docs), 2)
        self.assertTrue(all(d.metadata["filename"] == "selected.pdf" for d in docs))
