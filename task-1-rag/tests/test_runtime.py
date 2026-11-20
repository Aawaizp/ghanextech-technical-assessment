"""PDF validation and chat UI regressions; no provider calls."""
from contextlib import ExitStack
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from langchain_core.documents import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from streamlit.testing.v1 import AppTest
from modules import document_processor
from modules.pdf_loader import load_pdf
from test_session_isolation import FixedEmbeddings, stub_generation

NOT_FOUND = "I couldn't find that information in the uploaded documents."


def pdf_bytes(text=None, password=None, extra_blank=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
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
    if extra_blank:
        writer.add_blank_page(width=612, height=792)
    if password:
        writer.encrypt(password)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class PDFValidationTests(unittest.TestCase):
    def load_bytes(self, data):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.pdf"
            path.write_bytes(data)
            return load_pdf(str(path))

    def test_readable_pdf_keeps_page_metadata(self):
        documents = self.load_bytes(pdf_bytes("alpha document"))
        self.assertEqual(documents[0].page_content.strip(), "alpha document")
        self.assertEqual(documents[0].metadata["page"], 0)

    def test_blank_pdf_is_reported(self):
        with self.assertRaisesRegex(ValueError, "no readable text"):
            self.load_bytes(pdf_bytes())

    def test_password_protection_is_reported(self):
        with self.assertRaisesRegex(ValueError, "password-protected"):
            self.load_bytes(pdf_bytes("alpha document", password="test-only-password"))

    def test_malformed_pdf_is_reported(self):
        with self.assertRaisesRegex(ValueError, "fresh PDF"):
            self.load_bytes(b"%PDF-1.4 broken")

    def test_empty_and_oversized_uploads_do_not_reach_parser(self):
        with patch.object(document_processor, "load_pdf") as parser:
            with self.assertRaisesRegex(ValueError, "empty"):
                document_processor.process_pdf(SimpleNamespace(name="empty.pdf", getvalue=lambda: b""))
            with patch.object(document_processor, "MAX_UPLOAD_BYTES", 5):
                with self.assertRaisesRegex(ValueError, "20 MB"):
                    document_processor.process_pdf(SimpleNamespace(name="large.pdf", getvalue=lambda: b"123456"))
        parser.assert_not_called()

    def test_partly_blank_pdf_has_warning_and_keeps_readable_page(self):
        data = pdf_bytes("alpha document", extra_blank=True)
        with patch.object(document_processor, "store_documents", return_value=1) as store:
            result = document_processor.process_pdf(
                SimpleNamespace(name="mixed.pdf", getvalue=lambda: data)
            )
        self.assertEqual(result["pages"], 2)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("1 page(s)", result["warnings"][0])
        self.assertEqual(store.call_args.args[0][0].metadata["filename"], "mixed.pdf")


class ChatUITests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.doc = Document(page_content="Alpha is a test project.", metadata={"filename": "notes.pdf", "page": 0})
        self.fake_rag = SimpleNamespace(
            ask_llm_stream=Mock(return_value=(iter(["Alpha is a test project [1]."]), [self.doc])),
            NOT_FOUND_MESSAGE=NOT_FOUND,
        )
        self.stack.enter_context(stub_generation(self.fake_rag))
        self.stack.enter_context(patch("modules.vector_store.get_embedding_model", return_value=FixedEmbeddings()))
        self.app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=30)
        data = pdf_bytes("alpha document")
        uploaded = SimpleNamespace(name="notes.pdf", getvalue=lambda: data)
        with patch("streamlit.file_uploader", side_effect=[[uploaded], []]):
            self.app.run(timeout=30)
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)

    def tearDown(self):
        index = self.app.session_state.filtered_state.get("_doctok_document_index")
        if index is not None:
            index.close()
            del self.app.session_state["_doctok_document_index"]
        self.stack.close()

    def ask(self):
        self.app.chat_input[0].set_value("What does the document say about alpha?").run(timeout=30)
        self.assertFalse(self.app.exception)

    def test_answer_displays_sources(self):
        self.ask()
        self.assertFalse(self.app.error)
        self.assertTrue(any("View sources" in expander.label for expander in self.app.expander))
        self.assertEqual(self.app.session_state["messages"][-1]["content"], "Alpha is a test project [1].")
        self.assertFalse(any("Your documents are ready." in item.value for item in self.app.markdown))
        self.assertFalse(next(b for b in self.app.button if b.key == "new_chat").disabled)
        self.fake_rag.ask_llm_stream.assert_called_once_with(
            "What does the document say about alpha?", filenames=["notes.pdf"]
        )
        # Moving between the welcome composer and the conversation must retain
        # documents and accept the next question after starting a new chat.
        next(b for b in self.app.button if b.key == "new_chat").click().run(timeout=30)
        self.assertEqual(self.app.session_state["messages"], [])
        self.assertEqual(list(self.app.session_state["processed_versions"]), ["notes.pdf"])
        self.assertFalse(self.app.chat_input[0].disabled)
        self.fake_rag.ask_llm_stream.return_value = (iter(["Alpha is a test project [1]."]), [self.doc])
        self.ask()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertFalse(self.app.error)

    def test_refusal_does_not_display_sources(self):
        self.fake_rag.ask_llm_stream.return_value = (iter([NOT_FOUND]), [self.doc])
        self.ask()
        self.assertFalse(any("View sources" in expander.label for expander in self.app.expander))

    def test_stream_failure_preserves_question_and_shows_no_partial_answer(self):
        def broken_stream():
            yield "Partial answer [1]"
            raise RuntimeError("simulated network interruption")
        self.fake_rag.ask_llm_stream.return_value = (broken_stream(), [self.doc])
        self.ask()
        self.assertEqual(len(self.app.session_state["messages"]), 2)
        self.assertTrue(self.app.session_state["messages"][-1]["error"])
        self.assertFalse(any("Partial answer" in item.value for item in self.app.markdown))
        self.assertTrue(self.app.error)

    def test_source_html_is_escaped(self):
        self.doc.metadata["filename"] = "<script>not_executable</script>.pdf"
        self.ask()
        self.assertTrue(any("&lt;script&gt;" in item.value for item in self.app.markdown))
        self.assertFalse(any("<script>" in item.value for item in self.app.markdown))

    def test_missing_key_shows_configuration_message(self):
        self.fake_rag.ask_llm_stream.side_effect = RuntimeError("No Groq API key is configured.")
        self.ask()
        self.assertIn("aren't configured", self.app.error[0].value)


class NoKeyStartupTests(unittest.TestCase):
    def test_modules_import_without_api_key(self):
        code = '''
import os
from unittest.mock import patch
for name in list(os.environ):
    if name.startswith("GROQ_API_KEY"):
        del os.environ[name]
with patch("dotenv.load_dotenv", return_value=False):
    from modules import rag_chain, retriever
assert not rag_chain._API_KEYS
try:
    rag_chain.ask_llm_stream("What does the document say?")
except RuntimeError as exc:
    assert "API key" in str(exc)
else:
    raise AssertionError("Expected missing-key guidance")
print("Missing-key startup verified")
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=str(PROJECT), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
