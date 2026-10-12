#!/usr/bin/env python3
"""Fail when the catalog's pinned sha no longer matches the repo HEAD.

The catalog pins the commit the plugin content lives at; this guard makes a
stale pin loud in CI instead of silently shipping a diff mismatch.

Usage: python3 scripts/check_catalog_sha.py   (from a repo checkout)
"""
import subprocess
import sys
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "catalog" / "context-audit.yaml"


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        print("not a git checkout — skipping sha check", file=sys.stderr)
        return 0
    actual = head.stdout.strip()
    text = CATALOG.read_text()
    pinned = ""
    for line in text.splitlines():
        if line.strip().startswith("sha:"):
            pinned = line.split(":", 1)[1].strip()
            break
    if not pinned:
        print("FAIL: catalog has no sha: pin", file=sys.stderr)
        return 1
    if pinned == actual:
        print(f"OK: catalog sha matches HEAD ({actual[:12]})")
        return 0
    print(f"FAIL: catalog sha {pinned[:12]} != HEAD {actual[:12]} — update the pin", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
