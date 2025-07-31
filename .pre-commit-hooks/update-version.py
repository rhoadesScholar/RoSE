#!/usr/bin/env python3
"""
Pre-commit hook to update version in CITATION.cff before commit.
This ensures the version is updated automatically when committing to main.
"""

from datetime import datetime, timezone
import os
import subprocess
import sys


def get_current_branch():
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def generate_pep440_date_tag():
    """Generate a PEP440 compliant date-based version tag."""
    now = datetime.now(timezone.utc)
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    return f"{year}.{month}.{day}.{hour}{minute:02d}"


def update_citation_file(version):
    """Update CITATION.cff with new version and release date."""
    citation_file = "CITATION.cff"

    if not os.path.exists(citation_file):
        print(f"Warning: {citation_file} not found, skipping version update")
        return False

    try:
        with open(citation_file, "r") as f:
            content = f.read()

        # Update version line
        import re

        content = re.sub(
            r"^version: .*$", f"version: {version}", content, flags=re.MULTILINE
        )

        # Update date-released line
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        content = re.sub(
            r"^date-released: .*$",
            f"date-released: {today}",
            content,
            flags=re.MULTILINE,
        )

        with open(citation_file, "w") as f:
            f.write(content)

        # Stage the updated file
        subprocess.run(["git", "add", citation_file], check=True)
        print(f"Updated {citation_file} with version {version}")
        return True

    except Exception as e:
        print(f"Error updating {citation_file}: {e}")
        return False


def main():
    """Main function for the pre-commit hook."""
    # Only run on main branch commits
    current_branch = get_current_branch()

    if current_branch != "main":
        print("Not on main branch, skipping version update")
        return 0

    # Check if this is a merge commit or has [skip ci] in message
    try:
        # Get the commit message if it exists (for amend operations)
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"], capture_output=True, text=True
        )
        if result.returncode == 0 and "[skip ci]" in result.stdout:
            print("Found [skip ci] in commit message, skipping version update")
            return 0
    except Exception:
        pass  # No existing commit, continue

    # Generate new version
    version = generate_pep440_date_tag()

    # Update CITATION.cff
    if update_citation_file(version):
        print(f"Version updated to {version}")
    else:
        print("Failed to update version")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
