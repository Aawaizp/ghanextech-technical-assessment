"""Ghanex's document workspace and chat interface."""
import hashlib
import logging
import os
from html import escape
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Ghanex · Document workspace",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    for name, value in st.secrets.items():
        if name.startswith("GROQ_API_KEY") and value:
            os.environ[name] = str(value)
except FileNotFoundError:
    pass

from modules.rag_chain import ask_llm_stream, NOT_FOUND_MESSAGE
from modules.document_processor import process_pdfs
from modules.vector_store import (
    initialize_document_session,
    clear_document_session,
    list_indexed_filenames,
    delete_file,
)

_LOGGER = logging.getLogger(__name__)
style = (Path(__file__).parent / "assets" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

initialize_document_session()
st.session_state.setdefault("document_stats", {})
try:
    all_filenames = list_indexed_filenames()
except Exception:
    st.error("We couldn't open your document library. Restart this session and upload your PDFs again.")
    st.stop()


def selection_key(filename):
    return "include_" + hashlib.sha256(filename.encode()).hexdigest()


def render_sources(sources):
    with st.expander(f"View sources · {len(sources)} passages"):
        st.caption("Match each [number] in the answer to its passage below.")
        for number, doc in enumerate(sources, start=1):
            filename = escape(str(doc.metadata.get("filename", "Document")))
            page = int(doc.metadata.get("page", 0)) + 1
            st.markdown(
                f'<div class="source-heading"><span class="source-number">[{number}]</span>'
                f'<span>{filename}</span><span class="source-page">Page {page}</span></div>'
                f'<div class="source-text">{escape(doc.page_content)}</div>',
                unsafe_allow_html=True,
            )


def render_message(message):
    is_user = message["role"] == "user"
    with st.chat_message(
        message["role"],
        avatar=":material/person:" if is_user else ":material/description:",
    ):
        st.caption("YOU" if is_user else "Ghanex")
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        if (
            not is_user
            and message.get("sources")
            and message["content"].strip() != NOT_FOUND_MESSAGE
        ):
            render_sources(message["sources"])


def answer_question(question, selected_filenames):
    question = question.strip()
    if not question:
        return
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.spinner("Finding relevant passages and preparing your answer…"):
        try:
            stream, sources = ask_llm_stream(question, filenames=selected_filenames)
            answer = NOT_FOUND_MESSAGE if stream is None else "".join(stream)
            message = {"role": "assistant", "content": answer, "sources": sources}
        except Exception as exc:
            _LOGGER.exception("Document answer request failed")
            if isinstance(exc, ValueError):
                explanation = (
                    "The answer's source references couldn't be verified. "
                    "Please try rephrasing your question."
                )
            elif "API key" in str(exc):
                explanation = "AI answers aren't configured yet. Add a Groq API key and restart the app."
            else:
                explanation = (
                    "We couldn't finish this answer. Your documents are still available. "
                    "Please try again in a moment."
                )
            message = {"role": "assistant", "content": explanation, "error": True}
    st.session_state.messages.append(message)
    render_message(message)


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-name">Ghanex<span>·</span></div>'
        '<div class="brand-description">A little clarity, from your documents.</div></div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "New chat", icon=":material/edit_square:", key="new_chat",
        use_container_width=True, help="Start a fresh conversation with the same documents.",
    ):
        st.session_state.messages = []
        st.rerun()
    st.markdown('<div class="section-label">DOCUMENTS</div>', unsafe_allow_html=True)
    st.caption("Add PDFs to your conversation.")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        help="PDFs up to 20 MB each. Scanned documents need OCR first.",
    )

    if uploaded_files:
        new_files = [
            item for item in uploaded_files
            if hashlib.sha256(item.getvalue()).hexdigest()
            != st.session_state.processed_versions.get(item.name)
        ]
        results = []
        if new_files:
            with st.spinner(f"Reading {len(new_files)} document(s)…"):
                results = process_pdfs(new_files)
            for info in results:
                if not info.get("error"):
                    filename = info["filename"]
                    st.session_state.processed_versions[filename] = info["document_hash"]
                    st.session_state.selected_files[filename] = True
                    st.session_state[selection_key(filename)] = True
                    st.session_state.document_stats[filename] = info
            if any(not info.get("error") for info in results):
                st.session_state.messages = []
        st.session_state.upload_results = results
        st.session_state.uploader_key += 1
        st.rerun()

    for info in st.session_state.pop("upload_results", []):
        if info.get("error"):
            st.error(f"{info['filename']}: {info['error']}")
        else:
            st.success(f"Ready: {info['filename']}")
            for warning in info.get("warnings", []):
                st.warning(warning)

    for filename in list(st.session_state.selected_files):
        if filename not in all_filenames:
            del st.session_state.selected_files[filename]

    if all_filenames:
        st.markdown(
            f'<div class="library-heading">Your library <span>{len(all_filenames):02d}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Checked files are included in your search.")
        for filename in all_filenames:
            st.session_state.selected_files.setdefault(filename, True)
            widget_key = selection_key(filename)
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state.selected_files[filename]
            name_column, remove_column = st.columns([5, 1], vertical_alignment="center")
            with name_column:
                st.session_state.selected_files[filename] = st.checkbox(
                    filename, key=widget_key, help=f"Include {filename} in search"
                )
            with remove_column:
                if st.button("✕", key=f"delete_{filename}", help=f"Remove {filename}"):
                    try:
                        delete_file(filename)
                    except Exception:
                        st.error("Couldn't remove this document. Please retry.")
                        st.stop()
                    st.session_state.processed_versions.pop(filename, None)
                    st.session_state.selected_files.pop(filename, None)
                    st.session_state.document_stats.pop(filename, None)
                    st.session_state.messages = []
                    st.rerun()
            stats = st.session_state.document_stats.get(filename)
            if stats:
                st.caption(f"{stats['pages']} pages · PDF")
    else:
        st.markdown(
            '<div class="library-empty">No documents yet.<br>'
            '<span>Your PDFs will appear here.</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    with st.expander("About this session"):
        st.write("Your documents are separate from other browser sessions.")
        st.caption(
            "Refreshing starts a new session. Questions and relevant excerpts "
            "are sent to Groq for answers. Scanned pages aren't read without OCR."
        )
    if st.button(
        "Clear documents and chat",
        disabled=not all_filenames,
        use_container_width=True,
        help="Remove this session's documents and conversation.",
    ):
        try:
            clear_document_session()
        except Exception:
            st.error("Couldn't clear the library. Please retry.")
        else:
            st.rerun()
    st.markdown('<div class="sidebar-footer"><span class="profile-mark">G</span><div>Ghanex<span>Document assistant</span></div></div>', unsafe_allow_html=True)


selected_filenames = [
    filename for filename, selected in st.session_state.selected_files.items() if selected
]
has_messages = bool(st.session_state.messages)
selected_count = len(selected_filenames)
scope_label = (
    f"{selected_count} of {len(all_filenames)} documents selected"
    if all_filenames else "PDF workspace"
)
st.markdown(
    '<div class="workspace-bar"><span>Document chat</span>'
    f'<span class="workspace-scope">{scope_label}</span></div>',
    unsafe_allow_html=True,
)

if all_filenames and not selected_filenames:
    st.warning("Select at least one document in the library to ask a question.")

input_options = {
    "placeholder": (
        "Ask Ghanex about your documents…" if all_filenames
        else "Upload a PDF to start a conversation…"
    ),
    "disabled": not selected_filenames,
    "max_chars": 2000,
    "key": "document_question",
}

if not has_messages:
    description = (
        "Upload one or more PDFs from the sidebar to get started."
        if not all_filenames
        else "Your documents are ready. Ask a question and explore the sources."
    )
    st.markdown(
        '<section class="welcome">'
        '<div class="welcome-mark" aria-hidden="true">g<span>·</span></div>'
        '<h1>What would you like to know?</h1>'
        f'<p>{description}</p></section>',
        unsafe_allow_html=True,
    )
    with st.container(key="welcome_composer"):
        query = st.chat_input(**input_options)
    st.markdown(
        '<div class="composer-note"><span>Answers from your PDFs</span>'
        '<span>Check the sources for every answer</span></div>',
        unsafe_allow_html=True,
    )
else:
    for message in st.session_state.messages:
        render_message(message)
    query = st.chat_input(**input_options)

if query and selected_filenames:
    answer_question(query, selected_filenames)
    st.rerun()
