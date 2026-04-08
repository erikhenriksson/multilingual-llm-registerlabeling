"""
Compare LLM register annotations between two JSONL files.

Matches documents by URL ("u" field), compares line-level annotations,
and reports agreement statistics.

Usage:
    python compare_annotations.py file1.jsonl file2.jsonl [--max-docs 5000]
"""

import argparse
import json
import sys
from collections import Counter


def load_docs(path, max_docs):
    docs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            row = json.loads(line)
            url = row.get("u", "")
            if not url:
                continue
            docs[url] = row
    return docs


def compare_annotations(ann1, ann2):
    """Compare two annotation lists line by line.

    Returns per-field agreement counts: {field: {agree: N, disagree: N}}
    and a list of disagreement details.
    """
    fields = ["mode_medium", "mode_turn", "field_activity", "tenor_formality"]

    lookup1 = {a["line"]: a for a in ann1}
    lookup2 = {a["line"]: a for a in ann2}

    common_lines = sorted(set(lookup1) & set(lookup2))

    counts = {f: Counter() for f in fields}
    disagreements = []

    for ln in common_lines:
        a1 = lookup1[ln]
        a2 = lookup2[ln]
        for f in fields:
            v1 = a1.get(f)
            v2 = a2.get(f)
            if v1 == v2:
                counts[f]["agree"] += 1
            else:
                counts[f]["disagree"] += 1
                disagreements.append(
                    {
                        "line": ln,
                        "field": f,
                        "file1": v1,
                        "file2": v2,
                    }
                )

    return counts, disagreements, len(common_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare register annotations between two JSONL files."
    )
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument(
        "--show-disagreements",
        type=int,
        default=20,
        help="Print first N disagreements (default: 20, 0=none)",
    )
    args = parser.parse_args()

    print(f"Loading {args.file1}...")
    docs1 = load_docs(args.file1, args.max_docs)
    print(f"  {len(docs1)} docs loaded")

    print(f"Loading {args.file2}...")
    docs2 = load_docs(args.file2, args.max_docs)
    print(f"  {len(docs2)} docs loaded")

    common_urls = sorted(set(docs1) & set(docs2))
    only1 = len(docs1) - len(common_urls)
    only2 = len(docs2) - len(common_urls)
    print(f"\nMatched by URL: {len(common_urls)} docs")
    if only1 or only2:
        print(f"  Only in file1: {only1}, only in file2: {only2}")

    # Accumulate across all docs
    fields = ["mode_medium", "mode_turn", "field_activity", "tenor_formality"]
    total_counts = {f: Counter() for f in fields}
    all_disagreements = []
    total_lines = 0
    n_skipped = 0
    n_both_failed = 0

    # Per-field confusion matrices
    confusion = {f: Counter() for f in fields}

    for url in common_urls:
        d1 = docs1[url]
        d2 = docs2[url]

        # Skip if either failed
        if d1.get("classification_failed") or d2.get("classification_failed"):
            if d1.get("classification_failed") and d2.get("classification_failed"):
                n_both_failed += 1
            else:
                n_skipped += 1
            continue

        ann1 = d1.get("llm_register_annotation") or {}
        ann2 = d2.get("llm_register_annotation") or {}

        for section in ("main", "comments"):
            s1 = ann1.get(section, [])
            s2 = ann2.get(section, [])
            if not s1 and not s2:
                continue

            counts, disag, n_lines = compare_annotations(s1, s2)
            total_lines += n_lines
            for f in fields:
                total_counts[f] += counts[f]
            for d in disag:
                d["url"] = url
                d["section"] = section
                confusion[d["field"]][(d["file1"], d["file2"])] += 1
            all_disagreements.extend(disag)

    # --- Report ---
    print(f"\nCompared {total_lines} lines across {len(common_urls)} matched docs")
    if n_skipped:
        print(f"  {n_skipped} docs skipped (one side failed)")
    if n_both_failed:
        print(f"  {n_both_failed} docs skipped (both failed)")

    print(f"\n{'Field':<20} {'Agree':>8} {'Disagree':>8} {'Agreement%':>10}")
    print("-" * 50)
    for f in fields:
        a = total_counts[f]["agree"]
        d = total_counts[f]["disagree"]
        total = a + d
        pct = (a / total * 100) if total else 0
        print(f"{f:<20} {a:>8} {d:>8} {pct:>9.1f}%")

    # Confusion pairs for fields with disagreements
    for f in fields:
        if not confusion[f]:
            continue
        top = confusion[f].most_common(10)
        print(f"\n  Top disagreements for {f}:")
        for (v1, v2), count in top:
            print(f"    {v1} vs {v2}: {count}")

    # Sample disagreements
    if args.show_disagreements and all_disagreements:
        n = min(args.show_disagreements, len(all_disagreements))
        print(f"\nFirst {n} disagreements:")
        for d in all_disagreements[:n]:
            print(
                f"  [{d['section']}] line {d['line']} | {d['field']}: "
                f"{d['file1']} vs {d['file2']} | {d['url'][:60]}"
            )


if __name__ == "__main__":
    main()
