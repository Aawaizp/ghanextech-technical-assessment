"""Conversation records owned by one Streamlit session, never globally cached."""

import json
from datetime import datetime, timezone
from uuid import uuid4

MAX_CHATS = 30


def new_chat(messages=None, title="New chat"):
    return {
        "id": uuid4().hex, "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": list(messages or []),
    }


def export_chat(chat):
    # Portable transcript only: do not trust imported tool traces as fresh evidence.
    return json.dumps({
        "format": "ghanex-chat-v1", "title": chat["title"],
        "messages": [
            {"role": m["role"], "content": m["content"], "is_error": bool(m.get("is_error"))}
            for m in chat["messages"]
        ],
    }, indent=2, ensure_ascii=False)


def import_chat(raw):
    if len(raw) > 2_000_000:
        raise ValueError("Choose a chat file smaller than 2 MB.")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("format") != "ghanex-chat-v1":
            raise ValueError()
        title, messages = data["title"], data["messages"]
        if not isinstance(title, str) or not title.strip() or len(title) > 80:
            raise ValueError()
        if not isinstance(messages, list) or len(messages) > 500:
            raise ValueError()
        cleaned = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                raise ValueError()
            content = message.get("content")
            if not isinstance(content, str) or len(content) > 20_000:
                raise ValueError()
            cleaned.append({
                "role": message["role"], "content": content,
                "is_error": message.get("is_error") is True,
            })
        return new_chat(cleaned, title.strip())
    except (ValueError, KeyError, TypeError, RecursionError, UnicodeDecodeError) as exc:
        raise ValueError("This is not a valid Ghanex chat export.") from exc


def matches_search(chat, query):
    query = query.strip().casefold()
    return not query or query in chat["title"].casefold() or any(
        query in message["content"].casefold() for message in chat["messages"]
    )
