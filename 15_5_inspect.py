#!/usr/bin/env python3
"""
Print top +/- loading features per factor, per language.

For each language subdirectory under fa_results/, reads loadings.csv and
prints the features loading above the threshold (default |0.35|, Biber's
standard salience cutoff) on each factor, sorted by loading magnitude.

Usage:
    python show_dimensions.py
    python show_dimensions.py --threshold 0.4
    python show_dimensions.py --langs eng_Latn cmn
"""

import argparse
from pathlib import Path

import pandas as pd


def show_language(fa_dir: Path, threshold: float):
    loadings_path = fa_dir / "loadings.csv"
    if not loadings_path.exists():
        print(f"[skip] {fa_dir.name}: no loadings.csv")
        return

    loadings = pd.read_csv(loadings_path, index_col=0)

    print()
    print("=" * 70)
    print(
        f"  {fa_dir.name}   ({loadings.shape[0]} features, {loadings.shape[1]} factors)"
    )
    print("=" * 70)

    for factor in loadings.columns:
        s = loadings[factor]
        pos = s[s >= threshold].sort_values(ascending=False)
        neg = s[s <= -threshold].sort_values()

        print(f"\n{factor}")
        print("-" * 70)

        print("  + positive pole")
        if len(pos):
            for feat, val in pos.items():
                print(f"    {val:+.3f}  {feat}")
        else:
            print("    (none)")

        print("  - negative pole")
        if len(neg):
            for feat, val in neg.items():
                print(f"    {val:+.3f}  {feat}")
        else:
            print("    (none)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa-dir", default="fa_results")
    ap.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="Subdir names under fa-dir. Default: all.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Minimum |loading| to display. Default: 0.35 (Biber).",
    )
    args = ap.parse_args()

    root = Path(args.fa_dir)
    if args.langs:
        subdirs = [root / lang for lang in args.langs]
    else:
        subdirs = sorted(d for d in root.iterdir() if d.is_dir())

    if not subdirs:
        print(f"No subdirectories in {root}")
        return

    for sub in subdirs:
        show_language(sub, args.threshold)


if __name__ == "__main__":
    main()
