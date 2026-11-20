"""Minimal chat interface for Ghanex Agent."""

import logging
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Community Cloud starts from the repository root; resolve this app's modules.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from agent import GroqToolAgent
from chat_store import MAX_CHATS, export_chat, import_chat, matches_search, new_chat
from ui import apply_styles, render_sources, render_tool_activity

load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="Ghanex Agent", page_icon=":material/explore:", layout="centered", initial_sidebar_state="expanded")
apply_styles()
logger = logging.getLogger(__name__)


def configured_keys() -> list[str]:
    """Read local configuration or Streamlit Cloud secrets, without displaying keys."""
    keys = {
        name: value for name, value in os.environ.items()
        if name.startswith("GROQ_API_KEY_") and value
    }
    try:
        keys.update({
            name: str(value) for name, value in st.secrets.items()
            if name.startswith("GROQ_API_KEY_") and value
        })
    except FileNotFoundError:
        pass
    return list(dict.fromkeys(value.strip() for _, value in sorted(keys.items()) if value.strip()))


if "chats" not in st.session_state:
    initial = new_chat(st.session_state.pop("agent_messages", []))
    if initial["messages"]:
        initial["title"] = initial["messages"][0]["content"][:60]
    st.session_state.chats = {initial["id"]: initial}
    st.session_state.active_chat = initial["id"]
chats = st.session_state.chats
api_keys = configured_keys()

with st.sidebar:
    with st.container(key="sidebar_controls"):
        st.markdown('<div class="sidebar-brand"><span class="brand-symbol">G</span> Ghanex <span class="brand-tag">AGENT</span></div>', unsafe_allow_html=True)
        if st.button("New chat", icon=":material/add:", use_container_width=True, key="new_chat", disabled=len(chats) >= MAX_CHATS):
            empty = next((chat for chat in chats.values() if not chat["messages"]), None)
            chat = empty or new_chat()
            chats[chat["id"]] = chat
            st.session_state.active_chat = chat["id"]
            st.rerun()
        search = st.text_input("Search chats", placeholder="Search chats…", label_visibility="collapsed", key="chat_search")
        st.markdown('<div class="sidebar-label">YOUR CONVERSATIONS</div>', unsafe_allow_html=True)
    visible = [c for c in reversed(list(chats.values())) if matches_search(c, search)]
    with st.container(border=False, key="history_list"):
        if not visible:
            st.caption("No conversations match your search.")
        for chat in visible:
            active = chat["id"] == st.session_state.active_chat
            label = chat["title"] if len(chat["title"]) <= 26 else chat["title"][:25].rstrip() + "…"
            if st.button(label, key=f"chat_{chat['id']}", help=chat["title"], icon=":material/chat_bubble_outline:",
                         type="primary" if active else "tertiary", use_container_width=True):
                st.session_state.active_chat = chat["id"]
                st.rerun()
    st.caption("Session history · Download chats to keep")
    import_column, about_column = st.columns(2, gap="small")
    with import_column.popover("Import", icon=":material/upload:", use_container_width=True):
        uploaded = st.file_uploader("Ghanex chat export", type=["json"], key="chat_import")
        st.caption("Restores messages; tool activity is not included.")
        if st.button("Import chat", disabled=uploaded is None or len(chats) >= MAX_CHATS):
            try:
                imported = import_chat(uploaded.getvalue())
                chats[imported["id"]] = imported
                st.session_state.active_chat = imported["id"]
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if len(chats) >= MAX_CHATS:
        st.caption("30-chat limit reached. Download and delete an older chat to make room.")
    with about_column.popover("About", icon=":material/info:", use_container_width=True):
        st.write("**Calculate · Weather · Wikipedia · Currency**")
        st.caption("Ghanex chooses tools when needed. Weather covers seven forecast dates. Currency uses dated daily reference rates, excluding fees.")
        st.caption("Messages and tool results are sent to Groq. Imported transcripts are conversation context, not verified source evidence.")
        st.caption("History lasts for this browser session. Use Chat → Download chat to keep it after reloading; Import restores the saved messages.")

chat = chats[st.session_state.active_chat]
messages = chat["messages"]
with st.container(key="chat_header"):
    heading, actions = st.columns([5, 2], vertical_alignment="center")
    with heading:
        st.markdown('<div class="chat-brand">Ghanex <span>YOUR AI WORKSPACE</span></div>', unsafe_allow_html=True)
        st.caption(chat["title"])
    with actions:
        with st.popover("Chat", icon=":material/more_horiz:", use_container_width=True):
            with st.form(f"rename_{chat['id']}"):
                title = st.text_input("Conversation name", value=chat["title"], max_chars=80)
                if st.form_submit_button("Save name"):
                    if title.strip():
                        chat["title"] = title.strip()
                        st.rerun()
                    else:
                        st.error("Enter a conversation name.")
            st.download_button("Download chat", data=export_chat(chat), file_name=f"ghanex-chat-{chat['id'][:8]}.json",
                               mime="application/json", icon=":material/download:", on_click="ignore",
                               use_container_width=True)
            confirm = st.checkbox("Confirm deletion", key=f"delete_confirm_{chat['id']}")
            if st.button("Delete chat", icon=":material/delete:", disabled=not confirm, key=f"delete_{chat['id']}", use_container_width=True):
                del chats[chat["id"]]
                if not chats:
                    replacement = new_chat()
                    chats[replacement["id"]] = replacement
                st.session_state.active_chat = next(reversed(chats))
                st.rerun()
    

if not api_keys:
    st.warning("Add a Groq API key to .env or Streamlit Cloud secrets to start chatting.")

if not messages:
    st.markdown(
        '<div class="welcome"><div class="welcome-mark">G</div>'
        '<div class="welcome-eyebrow">A LITTLE LESS EFFORT. A LITTLE MORE CLARITY.</div><h1>What’s on your mind?</h1>'
        '<p>A question, a calculation, or a little clarity.</p>'
        '<div class="capabilities">'
        '<div><b>Calculate</b><span>Expressions &amp; everyday math</span></div>'
        '<div><b>Weather</b><span>Conditions &amp; forecasts</span></div>'
        '<div><b>Discover</b><span>Background &amp; sources</span></div>'
        '<div><b>Convert</b><span>Currencies &amp; dated rates</span></div></div></div>',
        unsafe_allow_html=True,
    )

for message in messages:
    avatar = ":material/person:" if message["role"] == "user" else ":material/explore:"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("is_error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("tool_trace", []))
            render_tool_activity(message.get("tool_trace", []))

prompt = st.chat_input("Message Ghanex Agent", disabled=not api_keys, max_chars=4000)
if prompt and prompt.strip():
    if not messages and chat['title'] == 'New chat':
        chat['title'] = ' '.join(prompt.split())[:60]
    messages.append({"role": "user", "content": prompt.strip()})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt.strip())
    try:
        with st.spinner("Working on it…"):
            with GroqToolAgent(api_keys) as agent:
                result = agent.run(messages)
        answer, tool_trace, is_error = result.answer, result.tool_trace, result.is_error
    except RuntimeError as exc:
        answer, tool_trace, is_error = str(exc), [], True
    except Exception:
        logger.exception("Agent request failed")
        answer, tool_trace, is_error = "Something went wrong. Please try again.", [], True
    messages.append({
        "role": "assistant", "content": answer,
        "tool_trace": tool_trace, "is_error": is_error,
    })
    st.rerun()
