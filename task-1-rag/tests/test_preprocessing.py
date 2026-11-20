"""Keep PDF paragraph and page boundaries useful for retrieval and citations."""
import unittest

from langchain_core.documents import Document

from modules.text_chunker import chunk_documents
from modules.text_cleaner import clean_text


class PreprocessingTests(unittest.TestCase):
    def test_wrapped_lines_join_but_paragraphs_remain_separate(self):
        extracted = "  First\tparagraph\r\nwraps here.\r\n \r\nSecond   paragraph.  "
        self.assertEqual(
            clean_text(extracted), "First paragraph wraps here.\n\nSecond paragraph."
        )

    def test_chunks_follow_paragraphs_and_keep_page_references(self):
        first = "First topic. " * 25
        second = "Second topic. " * 25
        pages = [
            Document(page_content=first + "\n\n" + second,
                     metadata={"source": "notes.pdf", "page": 0}),
            Document(page_content="Last page.", metadata={"source": "notes.pdf", "page": 1}),
        ]
        chunks = chunk_documents(pages, "fixture-hash")
        self.assertEqual([c.page_content for c in chunks],
                         [first.strip(), second.strip(), "Last page."])
        self.assertEqual([c.metadata["page"] for c in chunks], [0, 0, 1])
        self.assertTrue(all(c.metadata["filename"] == "notes.pdf" for c in chunks))
        self.assertEqual(len({c.metadata["chunk_id"] for c in chunks}), 3)
