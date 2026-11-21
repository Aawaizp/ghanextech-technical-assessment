"""Bounded, read-only GitHub REST client. Never clones or executes repository code."""

import base64
from datetime import datetime, timezone
import re
from urllib.parse import quote, urlsplit
import requests


class GitHubError(RuntimeError):
    pass


def parse_repository(value: str) -> str:
    """Convert a GitHub repository URL or owner/repo into owner/repo."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter a GitHub repository URL or owner/repo.")

    value = value.strip()

    if "://" in value:
        parsed = urlsplit(value)

        if (
            parsed.scheme != "https"
            or parsed.netloc.lower() != "github.com"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Use https://github.com/owner/repo or owner/repo."
            )

        value = parsed.path.strip("/")

    value = value.removesuffix(".git")

    pattern = (
        r"[A-Za-z0-9][A-Za-z0-9-]{0,38}"
        r"/[A-Za-z0-9_.-]{1,100}"
    )

    if not re.fullmatch(pattern, value):
        raise ValueError(
            "Enter a repository, such as psf/requests. "
            "Do not include a branch or file path."
        )

    if value.split("/")[1] in {".", ".."}:
        raise ValueError("Invalid repository name.")

    return value


class GitHubClient:
    def __init__(self, token="", session=None):
        self.http = session or requests.Session()
        self.owned = session is None
        self.headers = {
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "GhanexRepositoryAnalyzer/1.0",
        }
        if token.strip():
            self.headers["Authorization"] = f"Bearer {token.strip()}"
        self.remaining = None

    def close(self):
        if self.owned:
            self.http.close()

    def get(self, path, params=None, optional=False):
        try:
            response = self.http.get(
                "https://api.github.com" + path, params=params,
                headers=self.headers, timeout=(5, 20), allow_redirects=False,
            )
            self.remaining = response.headers.get("X-RateLimit-Remaining")
            if response.status_code == 404:
                if optional:
                    return None
                raise GitHubError("Repository not found or not publicly accessible. Check the owner and name.")
            if response.status_code == 401:
                raise GitHubError("GitHub rejected the token. Check GITHUB_TOKEN or remove it for public access.")
            if response.status_code in (403, 429):
                if self.remaining == "0" or response.status_code == 429 or response.headers.get("Retry-After"):
                    raise GitHubError("GitHub rate limit reached. Wait before retrying; an optional GitHub token may provide a higher allowance.")
                raise GitHubError("GitHub denied access to this request. Check token permissions and repository visibility.")
            if 300 <= response.status_code < 400:
                raise GitHubError("GitHub redirected this repository. Enter its current canonical owner/repository name.")
            response.raise_for_status()
            if len(response.content) > 10_000_000:
                raise GitHubError("GitHub response is too large for this analyzer.")
            return response.json()
        except GitHubError:
            raise
        except requests.Timeout as exc:
            raise GitHubError("GitHub timed out. Please retry later.") from exc
        except requests.RequestException as exc:
            raise GitHubError("Could not reach GitHub successfully. Please retry later.") from exc
        except ValueError as exc:
            raise GitHubError("GitHub returned invalid JSON.") from exc

    def snapshot(self, repository):
        repository = parse_repository(repository)
        prefix = "/repos/" + repository
        metadata = self.get(prefix)
        if metadata.get("private"):
            raise GitHubError("Only public repositories are supported, even when a token is configured.")
        branch = metadata["default_branch"]
        commit = self.get(prefix + "/commits/" + quote(branch, safe=""))
        sha = commit["sha"]
        tree = self.get(prefix + "/git/trees/" + sha, {"recursive": "1"})
        entries = tree.get("tree", [])
        files = [e for e in entries if e.get("type") == "blob"]
        warnings = []
        if tree.get("truncated"):
            warnings.append("GitHub truncated the directory tree. Missing paths are not proof that a file does not exist.")
        descriptors = {"pyproject.toml", "requirements.txt", "package.json", "Cargo.toml", "go.mod", "pom.xml", "setup.cfg"}
        def priority(entry):
            path = entry["path"]
            name = path.rsplit("/", 1)[-1]
            lower = name.lower()
            if lower.startswith("readme"):
                return 0 if "/" not in path else 4
            if name in descriptors:
                return 1 if "/" not in path else 3
            if lower in {"contributing.md", "contributing.rst"}:
                return 2
            if path.startswith(".github/workflows/") and name.endswith((".yml", ".yaml")):
                return 5
            if name.endswith((".py", ".js", ".ts", ".rs", ".go")):
                return 6
            return 99
        # Reserve slots for documentation, configuration, implementation and tests.
        eligible = sorted(
            [entry for entry in files if entry.get("size", 0) <= 100_000],
            key=lambda entry: (entry["path"].count("/"), entry["path"]),
        )
        candidates = []
        selected_paths = set()

        def select_files(predicate, count=1, rank=None):
            added = 0
            for entry in sorted(eligible, key=rank) if rank else eligible:
                path = entry["path"]
                if len(candidates) >= 8 or added >= count:
                    break
                if path not in selected_paths and predicate(path):
                    candidates.append(entry)
                    selected_paths.add(path)
                    added += 1

        def is_test_file(path):
            name = path.rsplit("/", 1)[-1]
            return (
                (name.startswith("test_") and name.endswith(".py"))
                or name.endswith(("_test.py", "_test.go", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
            )

        def is_source_file(path):
            parts = path.lower().split("/")
            name = parts[-1]
            excluded = {"tests", "test", "docs", "examples", "fixtures", "vendor", "node_modules", "__tests__", "__mocks__", "mocks"}
            return (
                not excluded.intersection(parts[:-1])
                and not any(part.startswith("e2e") or part in {"scripts", "tools", "demos", "benchmarks"} for part in parts[:-1])
                and not is_test_file(path)
                and not name.endswith(".d.ts")
                and not re.search(r"(^|[._-])(test|mock|fixture|config|setup)([._-]|$)", name)
                and name not in {"setup.py", "__init__.py", "conftest.py", "__version__.py", "_version.py", "version.py", "__main__.py"}
                and name.endswith((".py", ".js", ".ts", ".rs", ".go"))
            )

        select_files(lambda path: "/" not in path and path.lower().startswith("readme"))
        select_files(lambda path: "/" not in path and path in descriptors)
        select_files(lambda path: path.rsplit("/", 1)[-1].lower() in {"contributing.md", "contributing.rst"})
        select_files(lambda path: path.rsplit("/", 1)[-1].lower() == "ai_policy.md")
        select_files(
            lambda path: path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")),
            rank=lambda entry: (
                not bool(re.match(r"(?:ci|test|tests|lint|python|build)(?:[-_.]|$)", entry["path"].rsplit("/", 1)[-1].lower())),
                entry["path"],
            ),
        )
        # A nested manifest often describes the published package, unlike the root dev environment.
        select_files(lambda path: "/" in path and path.rsplit("/", 1)[-1] in descriptors
                     and path.split("/")[0] in {"lib", "src", "packages"})
        select_files(is_source_file, rank=lambda e: (not any(p in {"src", "lib"} for p in e["path"].split("/")[:-1]), e["path"].count("/"), e["path"]))
        select_files(is_test_file)
        select_files(is_source_file, count=2)

        for entry in sorted(eligible, key=lambda item: (priority(item), item["path"].count("/"), item["path"])):
            if len(candidates) >= 8:
                break
            if (entry["path"] not in selected_paths and priority(entry) < 99
                    and (priority(entry) != 6 or is_source_file(entry["path"]) or is_test_file(entry["path"]))):
                candidates.append(entry)
                selected_paths.add(entry["path"])

        evidence = [{
            "id": "metadata", "path": "Repository metadata", "url": metadata["html_url"],
            "text": str({k: metadata.get(k) for k in ("description", "language", "topics", "archived", "pushed_at", "default_branch")}),
        }, {
            "id": "tree", "path": "Directory structure",
            "url": f"https://github.com/{repository}/tree/{sha}",
            "text": "\n".join(e["path"] for e in entries[:500]),
        }]
        if len(entries) > 500:
            warnings.append("Only the first 500 paths are included in the model's directory view.")
        for entry in candidates:
            payload = self.get(prefix + "/contents/" + quote(entry["path"], safe="/"), {"ref": sha}, optional=True)
            if not payload or payload.get("encoding") != "base64" or payload.get("type") != "file":
                warnings.append(f"Could not read {entry['path']}.")
                continue
            try:
                content = base64.b64decode(payload["content"]).decode("utf-8")
            except (ValueError, UnicodeError, KeyError):
                warnings.append(f"Skipped unreadable text: {entry['path']}.")
                continue
            limit = 4500
            evidence.append({
                "id": f"file-{len(evidence)-1}", "path": entry["path"],
                "url": f"https://github.com/{repository}/blob/{sha}/{quote(entry['path'], safe='/')}",
                "text": content[:limit], "truncated": len(content) > limit,
            })
        if not any(e["path"].lower().startswith("readme") for e in evidence):
            warnings.append("A root README was not retrieved. Purpose and setup details may be incomplete.")
        warnings.append("This is a bounded file sample, not a full code audit. No code or tests were executed.")
        return {
            "repository": repository, "url": metadata["html_url"], "commit": sha,
            "default_branch": branch, "description": metadata.get("description") or "",
            "stars": metadata.get("stargazers_count", 0), "forks": metadata.get("forks_count", 0),
            "open_issues_and_prs": metadata.get("open_issues_count", 0),
            "language": metadata.get("language") or "Not specified",
            "license": (metadata.get("license") or {}).get("spdx_id") or "Not specified",
            "topics": metadata.get("topics", []), "archived": metadata.get("archived", False),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "tree_paths": [e["path"] for e in entries[:500]],
            "tree_truncated": bool(tree.get("truncated")) or len(entries) > 500,
            "sampled_files": len(evidence) - 2, "evidence": evidence,
            "warnings": warnings, "rate_remaining": self.remaining,
        }
