# Task 3: GitHub Repository AI Analyzer & Open-Source Contribution

## Overview

This assignment has two parts:
1. **Engineering Component:** Build an intelligent GitHub repository analyzer that inspects a public repository via the GitHub REST API and generates structured technical insights using an LLM.
2. **Open-Source Contribution Component:** Identify, investigate, and submit a meaningful contribution (an issue or pull request) to an established open-source project.

This task assesses your ability to work with REST APIs, analyze unfamiliar codebases, synthesize technical documentation, and collaborate within open-source communities.

---

## Part A: GitHub Repository AI Analyzer

### Technical Requirements

#### 1. Repository Ingestion via GitHub REST API
- Accept a public GitHub repository URL or `owner/repo` string.
- Fetch repository metadata using the GitHub REST API (e.g., description, stars, forks, primary language, license, topics, open issues count).
- Retrieve repository content files, notably the `README.md`, directory structure, and package descriptors (e.g., `package.json`, `pyproject.toml`, or `requirements.txt`).
- Respect GitHub API rate limits gracefully (encourage candidates to support optional personal access tokens for unauthenticated limits).

#### 2. AI-Powered Technical Analysis
- Pass the repository metadata, README, and project structure to an LLM.
- Generate a concise executive summary explaining the project's purpose and tech stack.
- Analyze strengths, architectural patterns, and code organization.
- Produce at least **three practical, actionable recommendations** for improvements (e.g., testing coverage, documentation gaps, CI/CD pipeline, architectural modularity).
- Highlight potential contribution opportunities or good first issues for prospective developers.

#### 3. User Interface
- Provide an intuitive web dashboard where users input the repository URL and view formatted insights, charts, or structured sections.
- Handle invalid URLs, private repositories, and rate-limit errors gracefully with user-friendly notices.

---

## Part B: Open-Source Contribution Requirement

As a GenAI & Open-Source Developer Intern, contributing to public software repositories is a core responsibility. To demonstrate this capability:

### Instructions:
1. Select an active, public open-source repository on GitHub (preferably related to Python, AI, machine learning tools, or developer tooling).
2. Review the project's `README.md` and `CONTRIBUTING.md` guidelines.
3. Identify a genuine need:
   - A bug report with clear reproduction steps.
   - A detailed feature proposal with user rationale.
   - A documentation enhancement or typo fix that improves readability.
   - A code improvement, test addition, or bug fix via Pull Request.
4. Submit either:
   - A well-written, informative GitHub Issue, **OR**
   - A well-crafted GitHub Pull Request adhering strictly to the maintainers' contribution standards.
5. Record the link to your contribution for submission.

*Note: Avoid spam, superficial changes (such as adding spaces or trivial punctuation), or unsolicited automated edits. Maintainers' time is valuable, and contributions should add genuine utility.*

---

## Suggested Stack

- **Language:** Python 3.10+
- **API Client:** `requests`, `httpx`, or `PyGithub`
- **Frontend:** Streamlit, Gradio, or FastAPI
- **LLM:** Google Gemini API, Groq, or OpenAI-compatible endpoints

---

## Deliverables

1. **Analyzer Source Code:** GitHub repository containing the complete analyzer application.
2. **Analyzer Deployment:** Public URL for the deployed analyzer (Streamlit Cloud, Hugging Face, etc.).
3. **Open-Source Contribution Links:**
   - Link to the target open-source repository.
   - Direct link to your submitted Issue or Pull Request.
   - A brief summary explaining the contribution and why it was needed.
