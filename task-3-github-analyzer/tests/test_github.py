import base64
import unittest
from unittest.mock import Mock

import requests

from analyzer.github import GitHubClient, GitHubError, parse_repository


class GitHubTests(unittest.TestCase):
    def test_repository_input(self):
        self.assertEqual(parse_repository("https://github.com/psf/requests.git"), "psf/requests")
        self.assertEqual(parse_repository("psf/requests"), "psf/requests")
        for value in ("", "https://example.com/a/b", "https://github.com/a/b/tree/main", "a/..", "https://github.com/a/b?token=secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_repository(value)

    def test_http_errors_are_safe_and_no_redirect_following(self):
        for status in (401, 403, 404, 429, 302):
            session = Mock()
            session.get.return_value = Mock(status_code=status, headers={})
            client = GitHubClient("secret", session)
            with self.subTest(status=status), self.assertRaises(GitHubError) as caught:
                client.get("/repos/a/b")
            self.assertNotIn("secret", str(caught.exception))
            self.assertFalse(session.get.call_args.kwargs["allow_redirects"])

    def test_timeout_is_actionable(self):
        session = Mock()
        session.get.side_effect = requests.Timeout()
        with self.assertRaisesRegex(GitHubError, "timed out"):
            GitHubClient(session=session).get("/repos/a/b")

    def test_private_repository_is_rejected(self):
        client = GitHubClient(session=Mock())
        client.get = Mock(return_value={"private": True})
        with self.assertRaisesRegex(GitHubError, "Only public"):
            client.snapshot("a/b")
        self.assertEqual(client.get.call_count, 1)

    def test_balanced_sample_prefers_ci_and_implementation(self):
        paths = ["README.md", "pyproject.toml", ".github/CONTRIBUTING.md", ".github/AI_POLICY.md",
                 ".github/workflows/close-issues.yml", ".github/workflows/tests.yml",
                 "frontend/test-globals.d.ts", "frontend/vitest-jest-dom.d.ts", "frontend/vite.config.ts",
                 "e2e_playwright/app.py", "scripts/build.py", "src/pkg/__version__.py", "src/pkg/client.py", "src/pkg/utils.py", "tests/test_client.py"]
        client = GitHubClient(session=Mock())
        calls = []
        def get(path, params=None, optional=False):
            calls.append((path, params))
            if path == "/repos/a/b":
                return {"private": False, "default_branch": "main", "html_url": "https://github.com/a/b"}
            if "/commits/" in path:
                return {"sha": "abc123"}
            if "/git/trees/" in path:
                return {"tree": [{"path": p, "type": "blob", "size": 100} for p in paths]}
            return {"encoding": "base64", "type": "file", "content": base64.b64encode(b"Example source content.").decode()}
        client.get = get
        result = client.snapshot("a/b")
        selected = [e["path"] for e in result["evidence"]]
        self.assertEqual(result["sampled_files"], 8)
        self.assertIn(".github/workflows/tests.yml", selected)
        self.assertIn("src/pkg/client.py", selected)
        self.assertNotIn("src/pkg/__version__.py", selected)
        for excluded in ("frontend/test-globals.d.ts", "frontend/vitest-jest-dom.d.ts",
                         "frontend/vite.config.ts", "e2e_playwright/app.py", "scripts/build.py"):
            self.assertNotIn(excluded, selected)
        for path, params in calls:
            if "/contents/" in path:
                self.assertEqual(params, {"ref": "abc123"})
