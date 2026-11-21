"""Test GitHub ingestion and AI analysis together."""

import json
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from analyzer.github import GitHubClient, GitHubError, parse_repository
from analyzer.cache import LocalCache
from analyzer.report import AnalysisError, analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Fetch fresh evidence and discard saved analysis progress.")
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))

    keys = [
        value
        for name, value in sorted(os.environ.items())
        if name.startswith("GROQ_API_KEY_") and value.strip()
    ]

    if not keys:
        print("Add GROQ_API_KEY_1 to your .env file first.")
        return

    repository = input("GitHub repository URL or owner/repo: ")
    client = GitHubClient(token=os.getenv("GITHUB_TOKEN", ""))
    cache = LocalCache(Path(__file__).parent / ".cache")
    snapshot = None
    progress = {}

    try:
        repository = parse_repository(repository)
        saved = None if args.refresh else cache.load(repository)
        if saved:
            snapshot, progress = saved
            print(f"\nUsing saved evidence at commit {snapshot['commit'][:12]}. Use --refresh for new evidence.")
        else:
            print("\nFetching repository evidence...")
            snapshot = client.snapshot(repository)

        print(
            f"Retrieved {snapshot['sampled_files']} files. "
            "Generating analysis..."
        )

        report = analyze(snapshot, keys, progress)

        print("\nAI REPORT\n")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    except (ValueError, GitHubError, AnalysisError) as error:
        print(f"\nAnalysis failed: {error}")

    finally:
        client.close()
        if snapshot:
            try:
                cache.save(repository, snapshot, progress)
            except OSError:
                print("Could not save the local checkpoint; the next run will start fresh.")


if __name__ == "__main__":
    main()
