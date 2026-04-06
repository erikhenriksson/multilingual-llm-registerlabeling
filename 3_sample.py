"""
Sample segments from annotated JSONL for human evaluation.

Simple random sampling of classifiable segments. Outputs a JSON file
containing the sampled segments with context, ready to load into the
annotation tool.

Usage:
    python sample_for_annotation.py \
        --input annotated.jsonl \
        --output sampled.jsonl \
        --n 500

Options:
    --n             Total segments to sample (default: 500)
    --seed          Random seed (default: 42)
    --section       Which section to sample from: main, comments, both (default: both)
    --stats         Print label statistics and exit (no output file)
    --max-line-chars  Truncate lines beyond this length (default: 500)
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_lines(md):
    if not md or not md.strip():
        return []
    lines = []
    for raw in md.split("\n"):
        m = re.match(r"\[(\d+)]\s*(.*)", raw)
        if m:
            lines.append({"num": int(m.group(1)), "text": m.group(2)})
    return lines


def is_structural(text):
    t = text.strip()
    return t.startswith("# ") or t.startswith("TABLE: ") or t.startswith("CODE: ")


def trunc(text, n):
    return text if len(text) <= n else text[:n] + "…"


def label_key(ann):
    """Summary key for reporting label distribution."""
    if not ann:
        return "no_ann"
    medium = ann.get("mode_medium", "unknown")
    activity = ann.get("field_activity", "none")
    if medium == "cannot_rate":
        activity = "_"
    if activity is None:
        activity = "none"
    return f"{medium}|{activity}"


# ---------------------------------------------------------------------------
# Extract all candidate segments
# ---------------------------------------------------------------------------


def extract_segments(input_path, sections, max_chars):
    """Yield all classifiable segments with their metadata."""
    with open(input_path) as f:
        for doc_idx, raw_line in enumerate(f):
            row = json.loads(raw_line)
            url = row.get("u", "")
            ann_data = row.get("llm_register_annotation", {})

            for sec in sections:
                md = row.get(f"markdown_{sec}", "")
                lines = parse_lines(md)
                if not lines:
                    continue

                anns = ann_data.get(sec, [])
                by_line = {}
                for a in anns:
                    if a.get("line") is not None:
                        by_line[a["line"]] = a

                for i, ln in enumerate(lines):
                    if is_structural(ln["text"]):
                        continue
                    la = by_line.get(ln["num"])
                    if la and la.get("is_structural"):
                        continue

                    yield {
                        "doc_idx": doc_idx,
                        "section": sec,
                        "url": url,
                        "line_idx": i,
                        "line_num": ln["num"],
                        "line_text": trunc(ln["text"], max_chars),
                        "llm": la,
                        "_lines_ref": lines,
                        "_i": i,
                    }


# ---------------------------------------------------------------------------
# Build context and output
# ---------------------------------------------------------------------------

CONTEXT = 5


def add_context(seg, max_chars):
    """Add context_before / context_after from the lines reference."""
    lines = seg.pop("_lines_ref")
    i = seg.pop("_i")

    before = []
    for j in range(max(0, i - CONTEXT), i):
        before.append(
            {"num": lines[j]["num"], "text": trunc(lines[j]["text"], max_chars)}
        )

    after = []
    for j in range(i + 1, min(len(lines), i + 1 + CONTEXT)):
        after.append(
            {"num": lines[j]["num"], "text": trunc(lines[j]["text"], max_chars)}
        )

    seg["context_before"] = before
    seg["context_after"] = after

    # Reshape LLM annotation for the annotation tool
    la = seg.pop("llm", None)
    if la:
        seg["llm"] = {
            "mode_medium": la.get("mode_medium"),
            "mode_turn": la.get("mode_turn"),
            "field_activity": la.get("field_activity"),
            "tenor_formality": la.get("tenor_formality"),
        }
    else:
        seg["llm"] = None

    return seg


def print_distribution(segments, title, file=sys.stderr):
    """Print label distribution for a set of segments."""
    counts = defaultdict(int)
    for seg in segments:
        la = seg.get("llm") if "llm" in seg else seg.get("llm")
        counts[label_key(la)] += 1

    total = len(segments)
    print(f"\n{title}:", file=file)
    print(f"{'Label':<35} {'Count':>8} {'%':>7}", file=file)
    print("─" * 52, file=file)
    for k in sorted(counts, key=lambda x: -counts[x]):
        pct = 100 * counts[k] / total
        print(f"  {k:<33} {counts[k]:>8} {pct:>6.1f}%", file=file)
    print("─" * 52, file=file)
    print(f"  {'TOTAL':<33} {total:>8}", file=file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Random sampling for annotation")
    parser.add_argument("--input", required=True, help="Annotated JSONL file")
    parser.add_argument(
        "--output", default=None, help="Output JSON for annotation tool"
    )
    parser.add_argument("--n", type=int, default=500, help="Total segments to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--section", default="both", choices=["main", "comments", "both"]
    )
    parser.add_argument("--stats", action="store_true", help="Print stats and exit")
    parser.add_argument("--max-line-chars", type=int, default=500)
    args = parser.parse_args()

    sections = ["main", "comments"] if args.section == "both" else [args.section]

    print(f"Scanning {args.input}...", file=sys.stderr)
    segments = list(extract_segments(args.input, sections, args.max_line_chars))
    print(f"Found {len(segments)} classifiable segments.", file=sys.stderr)

    if not segments:
        print("No segments found.", file=sys.stderr)
        sys.exit(1)

    print_distribution(segments, title="Population")

    if args.stats:
        return

    if not args.output:
        print(
            "\nNo --output specified. Use --output to write sampled tasks.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Sample
    rng = random.Random(args.seed)
    n = min(args.n, len(segments))
    sampled = rng.sample(segments, n)

    # Add context and assign sequential task IDs
    for i, seg in enumerate(sampled):
        add_context(seg, args.max_line_chars)
        seg["task_id"] = i

    print_distribution(sampled, title="Sampled")

    # Write as JSON array
    with open(args.output, "w") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\nWritten to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()