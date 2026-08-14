#!/usr/bin/env python3
"""Numeric-aware comparison of two experiment output directories.

`diff` is the wrong tool for checking a reproduction: it reports a failure when a
label changes and passes a run whose third decimal has drifted. This compares
positionally-extracted floats under per-field tolerances instead, and can ignore
labels entirely so that translating printed strings does not invalidate the
comparison.

usage:
  python tools/compare_outputs.py REF NEW               # byte + numeric report
  python tools/compare_outputs.py REF NEW --numbers-only # ignore all label text
  python tools/compare_outputs.py REF NEW --tol 0.002
"""

import argparse
import os
import re
import sys

NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def numbers(line):
    return [float(x) for x in NUM.findall(line)]


def labels(line):
    return NUM.sub("#", line)


def compare_file(ref_path, new_path, tol, numbers_only):
    ref = open(ref_path, encoding="utf-8").read().splitlines()
    new = open(new_path, encoding="utf-8").read().splitlines()

    if ref == new:
        return True, "byte-identical", []

    issues = []
    if len(ref) != len(new):
        issues.append(f"line count {len(ref)} -> {len(new)}")

    worst = 0.0
    for i, (a, b) in enumerate(zip(ref, new), 1):
        na, nb = numbers(a), numbers(b)
        if not numbers_only and labels(a) != labels(b):
            issues.append(f"L{i}: label text differs")
        if len(na) != len(nb):
            issues.append(f"L{i}: {len(na)} numbers -> {len(nb)}")
            continue
        for x, y in zip(na, nb):
            d = abs(x - y)
            worst = max(worst, d)
            if d > tol:
                issues.append(f"L{i}: {x} -> {y} (delta {d:.6f})")

    ok = not issues
    return ok, f"max delta {worst:.6f}", issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="absolute tolerance per number (default 0 = exact)")
    ap.add_argument("--numbers-only", action="store_true",
                    help="ignore label text; compare extracted numbers only")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.ref) if f.endswith(".txt"))
    failures = 0
    for name in names:
        new_path = os.path.join(args.new, name)
        if not os.path.exists(new_path):
            print(f"MISSING  {name}")
            failures += 1
            continue
        ok, summary, issues = compare_file(
            os.path.join(args.ref, name), new_path, args.tol, args.numbers_only
        )
        print(f"{'PASS' if ok else 'FAIL':6s} {name:34s} {summary}")
        for issue in issues[:8]:
            print(f"         {issue}")
        if len(issues) > 8:
            print(f"         ... and {len(issues) - 8} more")
        failures += not ok

    print(f"\n{len(names) - failures}/{len(names)} files pass "
          f"(tol={args.tol}, numbers_only={args.numbers_only})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
