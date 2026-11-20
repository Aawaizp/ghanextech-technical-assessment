import unittest
from unittest.mock import Mock, patch

import requests

from agent.tools.knowledge import KnowledgeSearchError, search_knowledge


def response(payload):
    result = Mock()
    result.json.return_value = payload
    return result


def session_for(search=None, pages=None):
    session = Mock()
    session.get.side_effect = [
        response({"query": {"search": [{"pageid": 12}]}} if search is None else search),
        response({"query": {"pages": {"12": {
            "title": "Example", "extract": "An introductory fact.",
            "fullurl": "https://en.wikipedia.org/wiki/Example",
        }}}} if pages is None else pages),
    ]
    return session


class KnowledgeTests(unittest.TestCase):
    def test_returns_excerpt_sources_and_honest_scope(self):
        session = session_for()
        result = search_knowledge(" Example ", session=session)
        self.assertEqual(result["query"], "Example")
        self.assertEqual(result["results"][0]["summary"], "An introductory fact.")
        self.assertEqual(result["results"][0]["url"], "https://en.wikipedia.org/wiki/Example")
        self.assertIn("not complete articles", result["scope_note"])
        for request in session.get.call_args_list:
            self.assertIn("github.com/Aawaizp/", request.kwargs["headers"]["User-Agent"])
            self.assertEqual(request.kwargs["timeout"], 10)
        session.close.assert_not_called()

    def test_preserves_search_order_and_bounds_excerpts(self):
        pages = {str(value): {"title": f"Page {value}", "extract": "a" * 1600,
                             "fullurl": f"https://en.wikipedia.org/wiki/Page_{value}"}
                 for value in (1, 2, 3, 4)}
        session = session_for(
            search={"query": {"search": [{"pageid": value} for value in (3, 1, 2, 4)]}},
            pages={"query": {"pages": pages}},
        )
        result = search_knowledge("example", session=session)
        self.assertEqual([page["title"] for page in result["results"]], ["Page 3", "Page 1", "Page 2"])
        self.assertTrue(all(len(page["summary"]) == 1500 and page["truncated"] for page in result["results"]))
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["pageids"], "3|1|2")

    def test_empty_search_avoids_an_extract_request(self):
        session = session_for(search={"query": {"search": []}})
        result = search_knowledge("no matches", session=session)
        self.assertEqual(result["results"], [])
        self.assertEqual(session.get.call_count, 1)

    def test_skips_deleted_pages_and_empty_excerpts(self):
        for pages in ({}, {"12": {"missing": ""}}, {"12": {
            "title": "Example", "extract": "", "fullurl": "https://en.wikipedia.org/wiki/Example",
        }}):
            with self.subTest(pages=pages):
                result = search_knowledge("example", session=session_for(pages={"query": {"pages": pages}}))
                self.assertEqual(result["results"], [])

    def test_rejects_bad_query_types_and_lengths(self):
        for query in (None, 1, [], "", " ", "a" * 301):
            session = Mock()
            with self.subTest(query=query), self.assertRaises(KnowledgeSearchError):
                search_knowledge(query, session=session)
            session.get.assert_not_called()

    def test_rejects_api_errors_and_malformed_search_payloads(self):
        for payload in (
            [], {"error": {"code": "ratelimited"}}, {}, {"query": []},
            {"query": {"search": {}}}, {"query": {"search": [None]}},
            {"query": {"search": [{"pageid": True}]}},
        ):
            with self.subTest(payload=payload), self.assertRaises(KnowledgeSearchError):
                search_knowledge("example", session=session_for(search=payload))

    def test_rejects_malformed_excerpts_and_untrusted_links(self):
        valid_page = {"title": "Example", "extract": "Fact", "fullurl": "https://en.wikipedia.org/wiki/Example"}
        for page in ([], dict(valid_page, extract=[]), dict(valid_page, title=None),
                     dict(valid_page, fullurl="https://en.wikipedia.org.evil.test/wiki/Example"),
                     dict(valid_page, fullurl="javascript:alert(1)")):
            with self.subTest(page=page), self.assertRaises(KnowledgeSearchError):
                search_knowledge("example", session=session_for(pages={"query": {"pages": {"12": page}}}))

    def test_closes_owned_session_on_success_and_failure(self):
        session = session_for()
        with patch("agent.tools.knowledge.requests.Session", return_value=session):
            search_knowledge("example")
        session.close.assert_called_once()
        session = Mock()
        session.get.side_effect = requests.Timeout("private implementation detail")
        with patch("agent.tools.knowledge.requests.Session", return_value=session):
            with self.assertRaisesRegex(KnowledgeSearchError, "timed out"):
                search_knowledge("example")
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
