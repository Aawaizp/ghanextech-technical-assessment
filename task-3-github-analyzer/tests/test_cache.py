import tempfile
import unittest
from unittest.mock import patch

from analyzer.cache import LocalCache
from test_report import snapshot


class CacheTests(unittest.TestCase):
    def test_roundtrip_expiry_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = LocalCache(directory)
            data = snapshot()
            state = {"draft": "unfinished report"}
            with patch("analyzer.cache.time.time", return_value=1000):
                cache.save("example/project", data, state)
                self.assertEqual(cache.load("example/project"), (data, state))
            with patch("analyzer.cache.time.time", return_value=5000):
                self.assertIsNone(cache.load("example/project"))
            cache.path("example/project").write_text("broken json", encoding="utf-8")
            self.assertIsNone(cache.load("example/project"))

    def test_cache_path_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = LocalCache(directory)
            self.assertEqual(cache.path("../../anything").parent, cache.directory)
