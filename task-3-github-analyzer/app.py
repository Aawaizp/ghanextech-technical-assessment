"""Ghanex repository analyzer dashboard."""

import json
import os
from pathlib import Path
import sys
import time
import math

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analyzer.github import GitHubClient, GitHubError, parse_repository
from analyzer.report import analyze, AnalysisError, AnalysisRateLimitError
from analyzer.export import export_report

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Ghanex Â· Repository Analyzer", page_icon=":material/code:", layout="wide")
st.markdown("<style>" + (ROOT / "ui.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def settings():
    config = dict(os.environ)
    try:
        config.update(dict(st.secrets))
    except FileNotFoundError:
        pass
    keys = [str(v) for k, v in sorted(config.items()) if k.startswith("GROQ_API_KEY_") and v]
    return keys, str(config.get("GITHUB_TOKEN", "")), config


keys, token, config = settings()
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-name">Ghanex</span><span class="brand-sub">workspace</span></div>', unsafe_allow_html=True)
    if st.button("New inspection", icon=":material/add:", use_container_width=True):
        for name in ("snapshot", "report", "analysis_progress"):
            st.session_state.pop(name, None)
        st.rerun()
    st.divider()
    st.caption("MODEL")
    st.write("Groq")
    st.caption("openai/gpt-oss-120b Â· Concise reports")
    st.divider()
    current = st.session_state.get("snapshot")
    if current:
        st.caption("CURRENT REPOSITORY")
        st.write(current["repository"])
        st.caption("Commit " + current["commit"][:12])
    else:
        st.caption("INSPECTIONS")
        st.caption("Your repository will appear here.")
    with st.expander("Settings & about", icon=":material/tune:"):
        st.caption("Read public files, inspect the evidence, then generate a report. Repository code is never executed.")
        st.caption("GitHub token configured" if token else "Public GitHub access Â· optional token supported")


if st.session_state.get("analysis_version") != 12:
    st.session_state.pop("report", None)
    st.session_state.pop("analysis_progress", None)
    st.session_state.analysis_version = 12

def use_example(repository):
    st.session_state.repo_input = repository


if not st.session_state.get("snapshot"):
    st.markdown(
        '<div class="welcome"><div class="welcome-mark" aria-hidden="true">G</div>'
        '<h1>What would you like to explore?</h1>'
        '<p>Start with a repository. Find your next contribution.</p></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="workspace-heading">Repository analysis</div>', unsafe_allow_html=True)

with st.form("repository"):
    value = st.text_input(
        "Public GitHub repository", placeholder="Paste a GitHub link or owner/repositoryâ€¦",
        max_chars=250, key="repo_input", label_visibility="collapsed",
    )
    hint, submit = st.columns([3, 1.4], vertical_alignment="center")
    with hint:
        st.caption("GitHub repository Â· Public access")
    with submit:
        inspect = st.form_submit_button("Inspect repository", type="primary", use_container_width=True)
if inspect:
    st.session_state.pop("snapshot", None)
    st.session_state.pop("report", None)
    st.session_state.pop("analysis_progress", None)
    client = None
    try:
        repository = parse_repository(value)
        with st.spinner("Fetching repository metadata and a bounded file sampleâ€¦"):
            client = GitHubClient(token)
            st.session_state.snapshot = client.snapshot(repository)
            st.rerun()
    except (ValueError, GitHubError) as exc:
        st.error(str(exc))
    except (KeyError, TypeError):
        st.error("GitHub returned an unexpected repository response. Please retry.")
    finally:
        if client:
            client.close()

snapshot = st.session_state.get("snapshot")
if not snapshot:
    with st.container(key="examples"):
        for column, label, repository in zip(
            st.columns(3), ["Requests", "Streamlit", "FastAPI"],
            ["psf/requests", "streamlit/streamlit", "fastapi/fastapi"],
        ):
            column.button(label, key="example_" + label, on_click=use_example,
                          args=(repository,), use_container_width=True, icon=":material/code:")
    st.markdown('<p class="workspace-footnote">Read-only inspection. Source code stays unexecuted.</p>', unsafe_allow_html=True)
if snapshot:
    st.subheader(snapshot["repository"])
    st.write(snapshot["description"])
    st.link_button("Open on GitHub", snapshot["url"])
    columns = st.columns(4)
    for col, label, value in zip(columns, ["Stars", "Forks", "Open issues + PRs", "Files inspected"],
                                  [snapshot["stars"], snapshot["forks"], snapshot["open_issues_and_prs"], snapshot["sampled_files"]]):
        col.metric(label, value)
    st.caption(f"Language: {snapshot['language']} Â· License: {snapshot['license']} Â· Commit: {snapshot['commit'][:12]}")
    if snapshot["topics"]:
        st.caption("Topics: " + " Â· ".join(snapshot["topics"]))
    if snapshot["archived"]:
        st.warning("This repository is archived. Choose an active project for the contribution requirement.")
    with st.expander("Inspection scope", expanded=False):
        for warning in snapshot["warnings"]:
            st.caption(warning)
        st.caption(f"Fetched: {snapshot['fetched_at']} Â· GitHub requests remaining: {snapshot['rate_remaining']}")
    if not keys:
        st.info("Configure GROQ_API_KEY_1 in .env or Streamlit secrets to generate the AI report.")
    progress = st.session_state.setdefault("analysis_progress", {})
    cooling_down = progress.get("retry_at", 0) > time.time()
    if cooling_down:
        @st.fragment(run_every=1)
        def cooldown_notice():
            seconds = max(0, math.ceil(progress["retry_at"] - time.time()))
            if not seconds:
                st.rerun()
            st.warning(f"Groq is temporarily rate limited. You can retry in {seconds} seconds. Your progress is saved.")
            st.caption("No automatic API retries are made. Smaller requests can still be rate limited.")
        cooldown_notice()
    if progress.get("draft"):
        st.info("Your draft is saved. Resume to validate its sources.")
    st.caption("Groq Â· Short report Â· Source quotes checked Â· Verify interpretations before acting")
    if st.button("Resume analysis" if progress.get("draft") else "Generate analysis", type="primary", disabled=cooling_down or not keys or bool(st.session_state.get("report"))):
        st.session_state.pop("report", None)
        try:
            with st.status("Analyzing the inspected evidenceâ€¦", expanded=False) as status:
                st.session_state.report = analyze(snapshot, keys, progress, on_progress=lambda message: status.update(label=message))
                status.update(label="Analysis complete", state="complete")
        except AnalysisRateLimitError:
            st.rerun()
        except AnalysisError as exc:
            st.error(str(exc))
    if progress.get("failed"):
        if st.button("Start a fresh analysis"):
            st.session_state.analysis_progress = {}
            st.rerun()
    overview, recommendations, evidence_tab = st.tabs(["Analysis", "Recommendations & contributions", "Inspected files"])
    evidence = {e["id"]: e for e in snapshot["evidence"]}

    def render_finding(item):
        st.markdown("#### " + item["title"])
        st.write(item["detail"])
        if "action" in item:
            st.write("**Action:** " + item["action"])
            st.write("**Validate:** " + item["validation"])
        for ref in dict.fromkeys(item["evidence_ids"]):
            source = evidence[ref]
            st.link_button(source["path"], source["url"])
        with st.expander("Supporting excerpts"):
            for reference in item["evidence_quotes"]:
                st.caption(evidence[reference["evidence_id"]]["path"])
                st.text(reference["quote"])
        st.divider()

    report = st.session_state.get("report")
    with overview:
        if report:
            st.subheader("Executive summary")
            st.write(report["summary"])
            st.markdown('<div class="report-note">A sampled inspection, not a full code audit. Review the excerpts before acting on a recommendation.</div>', unsafe_allow_html=True)
            st.write("**Tech stack:** " + ", ".join(report["tech_stack"]))
            st.subheader("Strengths")
            for item in report["strengths"]:
                render_finding(item)
            st.subheader("Architecture & organization")
            for item in report["architecture"]:
                render_finding(item)
            with st.expander("Analysis limitations"):
                for item in report["limitations"]:
                    st.write("- " + item)
        else:
            st.info("Inspect the files, then generate an analysis.")
    with recommendations:
        if report:
            st.subheader("Practical improvements")
            st.caption("Investigate existing behavior first. Make changes only after confirming a gap; quoted text alone does not verify the recommendation.")
            for item in report["recommendations"]:
                render_finding(item)
            st.subheader("Contribution opportunities")
            st.caption("Suggestions to investigate, not verified bugs or existing issues. Check CONTRIBUTING and duplicates before submitting.")
            for item in report["contribution_opportunities"]:
                render_finding(item)
    with evidence_tab:
        for source in snapshot["evidence"]:
            with st.expander(source["path"]):
                st.link_button("View source", source["url"])
                if source.get("truncated"):
                    st.caption("Excerpt truncated to 4,500 characters.")
                st.code(source["text"], language=None)
    if report:
        try:
            download = export_report(snapshot, report)
            st.download_button("Download report & evidence", download,
                               file_name="ghanex-repository-report.json", mime="application/json")
        except AnalysisError:
            st.error("This saved report failed source validation and cannot be exported. Start a new inspection.")
