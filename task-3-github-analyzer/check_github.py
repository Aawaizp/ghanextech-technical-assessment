"""Check GitHub access without calling the LLM."""

import os
from pathlib import Path

from dotenv import load_dotenv

from analyzer.github import GitHubClient, GitHubError, parse_repository


def main():
    load_dotenv(Path(__file__).with_name(".env"))

    repository_input = input("GitHub repository URL or owner/repo: ")
    client = GitHubClient(token=os.getenv("GITHUB_TOKEN", ""))

    try:
        repository = parse_repository(repository_input)
        data = client.get(f"/repos/{repository}")

        if data.get("private"):
            print("Only public repositories are supported.")
            return

        license_info = data.get("license") or {}

        print("\nRepository:", data["full_name"])
        print("Description:", data.get("description") or "Not provided")
        print("Primary language:", data.get("language") or "Not specified")
        print("Stars:", data.get("stargazers_count", 0))
        print("Forks:", data.get("forks_count", 0))
        print("Open issues + PRs:", data.get("open_issues_count", 0))
        print("License:", license_info.get("spdx_id") or "Not specified")
        print("Default branch:", data["default_branch"])
        print("API requests remaining:", client.remaining)

    except (ValueError, GitHubError) as error:
        print(f"\nCould not inspect repository: {error}")

    finally:
        client.close()


if __name__ == "__main__":
    main()