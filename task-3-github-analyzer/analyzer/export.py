"""Export only repository evidence and a structurally validated report."""

import json

from analyzer.report import validate_report


def export_report(snapshot, report):
    fields = (
        "repository", "url", "commit", "default_branch", "description", "stars",
        "forks", "open_issues_and_prs", "language", "license", "topics", "archived",
        "fetched_at", "tree_paths", "tree_truncated", "sampled_files", "warnings",
    )
    public = {key: snapshot[key] for key in fields if key in snapshot}
    public["evidence"] = [
        {key: source[key] for key in ("id", "path", "url", "text", "truncated") if key in source}
        for source in snapshot["evidence"]
    ]
    return json.dumps({
        "export_version": 1,
        "validation": {
            "schema_and_quotes_checked": True,
            "interpretations_independently_verified": False,
            "recommendations": "Proposals to investigate, not established deficiencies.",
        },
        "snapshot": public,
        "analysis": validate_report(json.dumps(report), public),
    }, indent=2, ensure_ascii=False)
