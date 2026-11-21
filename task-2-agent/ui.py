"""Presentation helpers; model routing and tool execution live in agent/."""

import json
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st

TOOL_LABELS = {
    "get_weather": "Weather",
    "calculate": "Calculator",
    "search_knowledge": "Wikipedia",
    "convert_currency": "Currency conversion",
}
SOURCE_HOSTS = {"en.wikipedia.org", "api.open-meteo.com", "api.frankfurter.dev"}


def apply_styles():
    st.markdown("<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def render_sources(trace):
    sources = {}
    for call in trace:
        if not call["result"].get("ok"):
            continue
        data = call["result"].get("data", {})
        candidates = data.get("results", []) if call["name"] == "search_knowledge" else [
            {"url": data.get("source_url"), "title": data.get("source", "Source")}
        ]
        for source in candidates:
            url = source.get("url") or ""
            parsed = urlsplit(url)
            if parsed.scheme == "https" and parsed.netloc in SOURCE_HOSTS:
                sources.setdefault(url, source.get("title", "Source"))
    if sources:
        with st.expander(f"Sources · {len(sources)}"):
            for url, title in sources.items():
                st.link_button(title, url)


def render_tool_activity(trace):
    if not trace:
        return
    with st.expander(f"Tool activity · {len(trace)}"):
        for call in trace:
            name = TOOL_LABELS.get(call["name"], call["name"])
            ok = call["result"].get("ok", False)
            status = "Completed" if ok else "Could not complete"
            if call.get("reused"):
                status += " · reused result"
            st.markdown(f"**{name}**")
            st.caption(status)
            if call["name"] == "get_weather" and ok:
                data = call["result"].get("data", {})
                cache = data.get("cache", {})
                if cache.get("hit"):
                    st.caption(f"Cached weather · {cache['age_seconds']} seconds old")
                if data.get("fetched_at_utc"):
                    st.caption(f"Fetched at {data['fetched_at_utc']} (UTC)")
            if not ok:
                st.caption(call["result"].get("error", "The tool returned an error."))
            st.code(json.dumps({
                "arguments": call["arguments"], "result": call["result"],
            }, indent=2, ensure_ascii=False), language="json")
