"""Dashboard flow tested with mocked services and networking disabled."""
import os
import time
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from test_report import snapshot, report
from analyzer.report import AnalysisRateLimitError

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        network = patch("socket.socket.connect", side_effect=AssertionError("Network disabled in tests"))
        network.start()
        self.addCleanup(network.stop)

    def test_invalid_url_shows_notice(self):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        self.assertFalse(app.exception)
        app.text_input[0].set_value("https://example.com/a/b")
        next(b for b in app.button if b.label == "Inspect repository").click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)

    def test_example_only_fills_the_repository_input(self):
        with patch("analyzer.github.GitHubClient.snapshot") as fetch:
            app = AppTest.from_file(str(APP)).run(timeout=30)
            next(b for b in app.button if b.label == "FastAPI").click().run(timeout=30)
            self.assertEqual(next(t for t in app.text_input if t.label == "Public GitHub repository").value, "fastapi/fastapi")
            self.assertFalse(app.exception)
            fetch.assert_not_called()

    def test_successful_report_and_rerun_do_not_refetch(self):
        data = snapshot()
        data.update({"url": "https://github.com/example/project", "commit": "abc123", "description": "Example",
                     "stars": 1, "forks": 0, "open_issues_and_prs": 0, "sampled_files": 1,
                     "language": "Python", "license": "MIT", "topics": [], "archived": False,
                     "fetched_at": "2026-11-21T00:00:00Z", "rate_remaining": "100"})
        data["evidence"][0]["url"] = "https://github.com/example/project/blob/abc123/README.md"
        with patch.dict(os.environ, {"GROQ_API_KEY_1": "test-only"}), \
             patch("analyzer.github.GitHubClient.snapshot", return_value=data) as fetch, \
             patch("analyzer.report.analyze", return_value=report()) as generate:
            app = AppTest.from_file(str(APP)).run(timeout=30)
            app.text_input[0].set_value("example/project")
            next(b for b in app.button if b.label == "Inspect repository").click().run(timeout=30)
            next(b for b in app.button if b.label == "Generate analysis").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any(e.label == "Supporting excerpts" for e in app.expander))
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(next(b for b in app.button if b.label == "Generate analysis").disabled)

    def test_rate_limit_shows_cooldown_and_disables_generation(self):
        data = snapshot()
        data.update({"url": "https://github.com/example/project", "commit": "abc123", "description": "Example",
                     "stars": 1, "forks": 0, "open_issues_and_prs": 0, "sampled_files": 1,
                     "language": "Python", "license": "MIT", "topics": [], "archived": False,
                     "fetched_at": "2026-11-21T00:00:00Z", "rate_remaining": "100"})
        data["evidence"][0]["url"] = "https://github.com/example/project/blob/abc123/README.md"
        def limited(snapshot, keys, progress, **kwargs):
            progress["retry_at"] = time.time() + 22
            raise AnalysisRateLimitError("rate limited")
        with patch.dict(os.environ, {"GROQ_API_KEY_1": "test-only"}), \
             patch("analyzer.github.GitHubClient.snapshot", return_value=data), \
             patch("analyzer.report.analyze", side_effect=limited) as generate:
            app = AppTest.from_file(str(APP)).run(timeout=30)
            app.text_input[0].set_value("example/project")
            next(b for b in app.button if b.label == "Inspect repository").click().run(timeout=30)
            next(b for b in app.button if b.label == "Generate analysis").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertTrue(next(b for b in app.button if b.label == "Generate analysis").disabled)
            app.run(timeout=30)
            self.assertEqual(generate.call_count, 1)
