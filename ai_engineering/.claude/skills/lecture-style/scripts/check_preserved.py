#!/usr/bin/env python3
"""Check that a rewritten lecture kept everything that must survive a style rewrite.

Usage: check_preserved.py LECTURE.md [--base REV]

Compares LECTURE.md with its version at REV (default HEAD). Missing display formulas,
section headings, images and code blocks are errors (exit 1). Missing numbers are listed
for manual review: a dropped repeat is fine, a dropped fact is not.
"""
import argparse
import re
import subprocess
import sys
from collections import Counter

DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
HEADING_NUM = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)*)\.", re.M)
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
CODE_BLOCK = re.compile(r"^```.*?^```", re.S | re.M)
NUMBER = re.compile(r"(?<![\w.])[−-]?\d+(?:[.,]\d+)?(?:·10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?")


def norm(s):
    return " ".join(s.split())


def extract(text):
    code = CODE_BLOCK.findall(text)
    prose = CODE_BLOCK.sub("", text)
    return {
        "formulas": Counter(norm(m) for m in DISPLAY_MATH.findall(prose)),
        "headings": Counter(f"{h} {n}" for h, n in HEADING_NUM.findall(prose)),
        "images": Counter(IMAGE.findall(prose)),
        "code": Counter(norm(c) for c in code),
        "numbers": Counter(n.replace("−", "-") for n in NUMBER.findall(DISPLAY_MATH.sub("", prose))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--base", default="HEAD")
    args = ap.parse_args()

    old = subprocess.run(
        ["git", "show", f"{args.base}:./{args.path}"], capture_output=True, text=True, check=True
    ).stdout
    with open(args.path, encoding="utf-8") as f:
        new = f.read()

    before, after = extract(old), extract(new)
    failed = False
    for kind in ("headings", "formulas", "images", "code"):
        lost = before[kind] - after[kind]
        if lost:
            failed = True
            print(f"ERROR: lost {kind} ({sum(lost.values())}):")
            for item, n in lost.items():
                print(f"  {n}× {item[:160]}")

    lost_nums = before["numbers"] - after["numbers"]
    if lost_nums:
        print(f"REVIEW: numbers that occur fewer times than before ({sum(lost_nums.values())}):")
        print("  " + ", ".join(f"{k}×{v}" if v > 1 else k for k, v in sorted(lost_nums.items())))

    if not failed:
        print("OK: headings, formulas, images and code blocks preserved")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
