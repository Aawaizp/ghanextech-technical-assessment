"""Upload parsing never shares a filesystem path or retains temporary PDFs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from modules import document_processor


def upload(name, content):
    return SimpleNamespace(name=name, getvalue=lambda: content)


class UploadIsolationTests(unittest.TestCase):
    def test_simultaneous_same_filename_uploads_use_distinct_temporary_paths(self):
        paths = []
        barrier = Barrier(2)

        def parse_pdf(path):
            paths.append(Path(path))
            barrier.wait(timeout=10)
            text = Path(path).read_bytes().decode()
            return [Document(page_content=text, metadata={"source": path, "page": 0})]

        with patch.object(document_processor, "load_pdf", side_effect=parse_pdf), patch.object(
            document_processor, "store_documents", return_value=1
        ) as store:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(document_processor.process_pdf, [
                    upload("same.pdf", b"alpha document"),
                    upload("same.pdf", b"beta document"),
                ]))
        self.assertEqual(len(set(paths)), 2)
        self.assertTrue(all(not path.parent.exists() for path in paths))
        self.assertEqual({result["filename"] for result in results}, {"same.pdf"})
        self.assertEqual(len({result["document_hash"] for result in results}), 2)
        chunks = [call.args[0][0] for call in store.call_args_list]
        self.assertEqual({chunk.page_content for chunk in chunks}, {"alpha document", "beta document"})
        self.assertTrue(all(chunk.metadata["source"] == "same.pdf" for chunk in chunks))

    def test_parse_failure_removes_temporary_file_and_does_not_index(self):
        paths = []

        def fail_parse(path):
            paths.append(Path(path))
            raise ValueError("Unreadable PDF")

        with patch.object(document_processor, "load_pdf", side_effect=fail_parse), patch.object(
            document_processor, "store_documents"
        ) as store:
            with self.assertRaisesRegex(ValueError, "Unreadable"):
                document_processor.process_pdf(upload("broken.pdf", b"invalid"))
        store.assert_not_called()
        self.assertFalse(paths[0].parent.exists())

    def test_uploaded_name_is_metadata_not_a_storage_path(self):
        paths = []

        def parse_pdf(path):
            paths.append(Path(path))
            return [Document(page_content="some text", metadata={"source": path, "page": 0})]

        with patch.object(document_processor, "load_pdf", side_effect=parse_pdf), patch.object(
            document_processor, "store_documents", return_value=1
        ) as store:
            result = document_processor.process_pdf(upload("../../other/notes.pdf", b"pdf bytes"))
        self.assertEqual(result["filename"], "notes.pdf")
        self.assertEqual(paths[0].name, "document.pdf")
        self.assertFalse(paths[0].exists())
        self.assertEqual(store.call_args.args[0][0].metadata["filename"], "notes.pdf")
        self.assertEqual(result["document_hash"], hashlib.sha256(b"pdf bytes").hexdigest())


if __name__ == "__main__":
    unittest.main()
