"""Concise factual lookup using Wikipedia's public API."""

from __future__ import annotations

from urllib.parse import urlsplit

import requests


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
EXCERPT_LIMIT = 1_500
SCOPE_NOTE = (
    "These are introductory Wikipedia excerpts, not complete articles or a general web search. "
    "A missing fact may appear elsewhere in an article. Only claim facts stated in these excerpts."
)
HEADERS = {
    "User-Agent": (
        "GhanexAssessmentAgent/0.1 "
        "(https://github.com/Aawaizp/ghanextech-technical-assessment)"
    )
}


class KnowledgeSearchError(RuntimeError):
    pass


def _query(http: requests.Session, params: dict) -> dict:
    response = http.get(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise KnowledgeSearchError("Wikipedia returned an invalid response")
    if "error" in payload:
        raise KnowledgeSearchError("Wikipedia could not process the search; please retry")
    query = payload.get("query")
    if not isinstance(query, dict):
        raise KnowledgeSearchError("Wikipedia returned an incomplete response")
    return query


def search_knowledge(query: str, session: requests.Session | None = None) -> dict:
    """Search Wikipedia and return up to three introductory extracts."""
    if not isinstance(query, str):
        raise KnowledgeSearchError("Search query must be a string")
    query = query.strip()
    if not query:
        raise KnowledgeSearchError("Search query cannot be empty")
    if len(query) > 300:
        raise KnowledgeSearchError("Search query cannot exceed 300 characters")
    http = requests.Session() if session is None else session
    try:
        search = _query(
            http,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
                "utf8": 1,
            },
        )
        matches = search.get("search")
        if not isinstance(matches, list):
            raise KnowledgeSearchError("Wikipedia returned invalid search results")
        if not matches:
            return {"query": query, "results": [], "scope_note": SCOPE_NOTE}
        page_ids = []
        for match in matches[:3]:
            if not isinstance(match, dict) or type(match.get("pageid")) is not int or match["pageid"] <= 0:
                raise KnowledgeSearchError("Wikipedia returned an invalid page identifier")
            page_id = str(match["pageid"])
            if page_id not in page_ids:
                page_ids.append(page_id)
        extracts = _query(
            http,
            {
                "action": "query",
                "prop": "extracts|info",
                "pageids": "|".join(page_ids),
                "exintro": 1,
                "exlimit": 3,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
            },
        )
        pages = extracts.get("pages")
        if not isinstance(pages, dict):
            raise KnowledgeSearchError("Wikipedia returned invalid article excerpts")
        results = []
        for page_id in page_ids:
            page = pages.get(page_id)
            # An article can disappear between the search and the extract request.
            if page is None:
                continue
            if not isinstance(page, dict):
                raise KnowledgeSearchError("Wikipedia returned an invalid article")
            if "missing" in page:
                continue
            title, excerpt, url = page.get("title"), page.get("extract"), page.get("fullurl")
            if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
                raise KnowledgeSearchError("Wikipedia returned incomplete source information")
            source = urlsplit(url)
            if source.scheme != "https" or source.netloc != "en.wikipedia.org":
                raise KnowledgeSearchError("Wikipedia returned an invalid source URL")
            if excerpt is None or excerpt == "":
                continue
            if not isinstance(excerpt, str):
                raise KnowledgeSearchError("Wikipedia returned an invalid article excerpt")
            results.append({
                "title": title,
                "summary": excerpt[:EXCERPT_LIMIT],
                "url": url,
                "truncated": len(excerpt) > EXCERPT_LIMIT,
            })
        return {"query": query, "results": results, "scope_note": SCOPE_NOTE}
    except KnowledgeSearchError:
        raise
    except requests.Timeout as exc:
        raise KnowledgeSearchError("Wikipedia search timed out; please retry") from exc
    except requests.RequestException as exc:
        raise KnowledgeSearchError("Wikipedia is unavailable; please retry") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise KnowledgeSearchError("Wikipedia returned an invalid response; please retry") from exc
    finally:
        if session is None:
            http.close()
