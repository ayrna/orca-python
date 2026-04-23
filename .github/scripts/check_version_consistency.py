"""Check that the project version is consistent across all declarations."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def extract_version_from_pyproject() -> str | None:
    content = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(.+?)"', content, re.MULTILINE)
    return match.group(1) if match else None


def extract_version_from_init() -> str | None:
    content = Path("skordinal/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\'](.+?)["\']', content)
    return match.group(1) if match else None


def extract_version_from_release_title(title: str) -> str | None:
    match = re.search(r"([0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.]+)?)", title)
    return match.group(1) if match else None


def main() -> int:
    pyproject_version = extract_version_from_pyproject()
    init_version = extract_version_from_init()

    print(f"pyproject.toml : {pyproject_version}")
    print(f"__init__.py    : {init_version}")

    versions = {pyproject_version, init_version}

    release_title = os.environ.get("RELEASE_TITLE")
    if release_title:
        release_version = extract_version_from_release_title(release_title)
        print(f"release title  : {release_title!r} -> {release_version}")
        versions.add(release_version)

    if None in versions or len(versions) != 1:
        print("Version mismatch detected.", file=sys.stderr)
        return 1

    print("All versions match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
