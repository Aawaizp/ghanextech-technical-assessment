"""Inspect the evidence that will later be sent to the LLM."""

import os
from pathlib import Path

from dotenv import load_dotenv

from analyzer.github import GitHubClient, GitHubError


def main():
    load_dotenv(Path(__file__).with_name(".env"))

    repository = input("GitHub repository URL or owner/repo: ")
    client = GitHubClient(token=os.getenv("GITHUB_TOKEN", ""))

    try:
        snapshot = client.snapshot(repository)

        print("\nRepository:", snapshot["repository"])
        print("Commit:", snapshot["commit"])
        print("Files inspected:", snapshot["sampled_files"])

        print("\nFirst 20 directory paths:")
        for path in snapshot["tree_paths"][:20]:
            print(" -", path)

        print("\nRetrieved evidence:")
        for source in snapshot["evidence"]:
            print(f"\n[{source['id']}] {source['path']}")
            print("Source:", source["url"])
            print("Characters:", len(source["text"]))
            print("Truncated:", source.get("truncated", False))

        print("\nInspection limitations:")
        for warning in snapshot["warnings"]:
            print(" -", warning)

    except (ValueError, GitHubError) as error:
        print(f"\nInspection failed: {error}")

    finally:
        client.close()


if __name__ == "__main__":
    main()