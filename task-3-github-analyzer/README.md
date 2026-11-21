# Ghanex Repository Analyzer

Inspect a public GitHub repository and generate a technical report from a bounded
sample of its files. Built with Python, Streamlit, requests, Pydantic and Groq.
Repository code is never cloned or executed.

## Run locally

Use Python 3.11. From this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run app.py --server.port 8503
```

Copy the example only when creating a new `.env`; preserve an existing one.
Set `GROQ_API_KEY_1`. Optional numbered keys are accepted, but keys in the same
organization share limits. `GITHUB_TOKEN` is optional and can improve the GitHub
request allowance. Never commit `.env` or Streamlit secrets.

## Analysis and limits

Groq-only concise reports use up to two excerpts per source and an output budget of
1,800 tokens. A normal report needs one model request. There is no separate model
review request. One correction is allowed if schema or source validation fails.
Selection favors short function blocks, configuration settings and explanatory prose,
skipping logo/badge blocks and standalone headings. Partial excerpts are marked and accompanied by bounded surrounding text.
Core source sampling excludes e2e apps and developer scripts; nested package
manifests and test/build workflows receive priority over maintenance workflows.
This heuristic selects context; it does not determine whether a claim is true.
Python restores exact source quotes from excerpt IDs. This checks quote existence,
not whether the model interpreted evidence correctly. Review findings before acting.

The report retains summary, tech stack, strengths, architecture, three actionable
recommendations and a contribution proposal. Recommendations must be conditional
investigations when deficiencies cannot be established from the sample.
The prompt asks for an inspection first, a change conditional on its findings, and
scenario-based verification instead of arbitrary percentage targets. File-role
hints distinguish example/e2e code from library implementation. These are model
instructions, not a guarantee of semantic accuracy; no extra review call is added.

The JSON download includes commit-pinned evidence, the report and explicit
validation limits. Schema and quotes are checked again before export. Only known
public snapshot fields are exported, not configuration or cached draft state.

GitHub reads at most eight files at an immutable commit, with up to 4,500 characters
per file and 500 directory paths. No repository code is executed. Truncation and
sampling limits are disclosed. Smaller model requests can still hit organization
limits; the app honors Retry-After, retains progress and requires manual retry.

CLI checkpoints are stored in the git-ignored `.cache/` folder for one hour.
Use `python check_analysis.py --refresh` to fetch fresh evidence. Browser state
lasts for the session. Ollama support was removed from this project; the installed
Ollama application and models on your computer are untouched.

## Verification

Run `python -B -m unittest discover -s tests -v`. Tests use mocked services.
Offline tests passed locally. Generated reports use a bounded source sample and
must be reviewed before acting on recommendations. Deployment and the open-source
contribution are still pending.

## Deployment preparation

After live verification and pushing the Task 3 files, configure the deployment:

- Repository: `Aawaizp/ghanextech-technical-assessment`
- Branch: `main` (ensure it contains the Task 3 files)
- Entry point: `task-3-github-analyzer/app.py`
- Python: `3.11`
- Dependencies: `task-3-github-analyzer/requirements.txt`

Set secrets in the hosting dashboard, not Git:

```toml
GROQ_API_KEY_1 = "replace-with-your-key"
GITHUB_TOKEN = "optional-public-repository-token"
```

Before submission, check the public URL in a signed-out browser. Inspect a public
repository, generate a report, inspect its sources and download the JSON. Confirm
invalid input produces a notice. Avoid repeatedly generating reports while limits
are low. Record the real deployment URL once these checks pass.

## Limits and contribution work

A bounded sample cannot establish missing features, full test coverage, runtime
performance or security. GitHub trees and excerpts may be truncated. Recommendations
should propose investigations where evidence is insufficient. Repository metadata
can change independently of the pinned file commit. Renamed repositories need their
canonical URL. Scanning archived repositories is allowed, with a warning to choose
an active project for the contribution requirement.

Task 3 Part B requires a meaningful issue or pull request to an active open-source
project. Read its full README, contribution instructions and AI-use policy; verify a
real need and check for duplicates before submitting. An analyzer suggestion alone
is not a verified bug. This application does not submit issues or pull requests.

Still required: target repository link, public contribution link, a brief account
of the verified problem/change, and the deployed analyzer URL. Do not describe
an unsubmitted proposal as a completed contribution.

Developed with AI assistance. Review and understand the implementation before
submitting the assessment.

### Evidence regression checks

The sampler excludes declaration files, mock/config files, e2e apps and scripts
from implementation slots. Dependency constraints are selected directly so a long
comment cannot hide an exclusion. The model receives the default branch explicitly.
Validation rejects recognized default-branch contradictions and specific unsupported
absence assertions; these narrow checks do not establish overall semantic accuracy.
Three concrete investigations are allowed when the sample cannot establish defects.
The one-correction limit remains unchanged; no extra reviewer request is added.
