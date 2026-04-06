"""
Sample segments from annotated JSONL for human evaluation.

Stratified sampling by mode_medium × field_activity ensures coverage
of rare label combinations. Outputs a JSONL file containing only the
sampled documents (with all their segments), ready to load into the
annotation tool.

Usage:
    python sample_for_annotation.py \
        --input annotated.jsonl \
        --output sampled.jsonl \
        --n 500

Options:
    --n             Total segments to sample (default: 500)
    --min-per-stratum  Minimum segments per stratum (default: 5)
    --seed          Random seed (default: 42)
    --section       Which section to sample from: main, comments, both (default: both)
    --stats         Print stratum statistics and exit (no output file)
    --max-line-chars  Truncate lines beyond this length (default: 500)
"""

import argparse
import json
import math
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


def stratum_key(ann):
    """Build stratification key from an annotation dict."""
    medium = ann.get("mode_medium", "unknown") if ann else "no_ann"
    activity = ann.get("field_activity", "none") if ann else "none"
    # Collapse nulls
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
                        "stratum": stratum_key(la),
                        # We'll fill context later from the sampled docs
                        "_lines_ref": lines,
                        "_i": i,
                    }


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def stratified_sample(segments, n, min_per_stratum, rng):
    """
    Stratified sampling with proportional allocation and a minimum floor.

    1. Group segments by stratum.
    2. Guarantee min_per_stratum from each stratum (or all if fewer available).
    3. Distribute remaining budget proportionally to stratum size.
    4. If a stratum can't fill its proportional share, redistribute surplus.
    """
    strata = defaultdict(list)
    for seg in segments:
        strata[seg["stratum"]].append(seg)

    # Shuffle within each stratum
    for k in strata:
        rng.shuffle(strata[k])

    num_strata = len(strata)
    allocation = {}
    remaining_budget = n

    # Phase 1: guarantee floor
    for k, segs in strata.items():
        floor = min(min_per_stratum, len(segs))
        allocation[k] = floor
        remaining_budget -= floor

    if remaining_budget < 0:
        # Floor alone exceeds budget — scale down proportionally
        total_floor = sum(allocation.values())
        for k in allocation:
            allocation[k] = max(1, round(allocation[k] * n / total_floor))
        remaining_budget = 0

    # Phase 2: proportional allocation of remaining budget
    if remaining_budget > 0:
        total_available = sum(max(0, len(strata[k]) - allocation[k]) for k in strata)
        if total_available > 0:
            # May need multiple passes if some strata are exhausted
            for _ in range(3):
                surplus = 0
                for k in strata:
                    avail = len(strata[k]) - allocation[k]
                    if avail <= 0:
                        continue
                    share = remaining_budget * (avail / total_available)
                    give = min(int(math.floor(share)), avail)
                    allocation[k] += give
                    surplus += share - give

                # Distribute fractional remainder
                remaining_budget = round(surplus)
                total_available = sum(
                    max(0, len(strata[k]) - allocation[k]) for k in strata
                )
                if remaining_budget <= 0 or total_available <= 0:
                    break

            # Final leftover: give one each to random strata
            leftover_keys = [k for k in strata if len(strata[k]) > allocation[k]]
            rng.shuffle(leftover_keys)
            for k in leftover_keys:
                if remaining_budget <= 0:
                    break
                allocation[k] += 1
                remaining_budget -= 1

    # Phase 3: collect samples
    sampled = []
    for k, segs in strata.items():
        count = min(allocation.get(k, 0), len(segs))
        sampled.extend(segs[:count])

    # Final shuffle so output isn't grouped by stratum
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Build context and output
# ---------------------------------------------------------------------------

CONTEXT = 5


def add_context(seg, max_chars):
    """Add context_before / context_after from the lines reference."""
    lines = seg.pop("_lines_ref")
    i = seg.pop("_i")
    seg.pop("stratum", None)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Stratified sampling for annotation")
    parser.add_argument("--input", required=True, help="Annotated JSONL file")
    parser.add_argument(
        "--output", default=None, help="Output JSONL for annotation tool"
    )
    parser.add_argument("--n", type=int, default=500, help="Total segments to sample")
    parser.add_argument(
        "--min-per-stratum", type=int, default=5, help="Minimum per stratum"
    )
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

    # Compute stats
    strata = defaultdict(int)
    for seg in segments:
        strata[seg["stratum"]] += 1

    print(f"\n{'Stratum':<35} {'Count':>8} {'%':>7}", file=sys.stderr)
    print("─" * 52, file=sys.stderr)
    for k in sorted(strata, key=lambda x: -strata[x]):
        pct = 100 * strata[k] / len(segments)
        print(f"  {k:<33} {strata[k]:>8} {pct:>6.1f}%", file=sys.stderr)
    print("─" * 52, file=sys.stderr)
    print(f"  {'TOTAL':<33} {len(segments):>8}", file=sys.stderr)

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
    sampled = stratified_sample(segments, n, args.min_per_stratum, rng)

    # Add context and reassign sequential task IDs
    for i, seg in enumerate(sampled):
        add_context(seg, args.max_line_chars)
        seg["task_id"] = i

    # Report sampling result
    sampled_strata = defaultdict(int)
    for seg in sampled:
        la = seg.get("llm")
        sampled_strata[stratum_key(la)] += 1

    print(f"\nSampled {len(sampled)} segments:", file=sys.stderr)
    print(f"{'Stratum':<35} {'Sampled':>8} {'Population':>12}", file=sys.stderr)
    print("─" * 57, file=sys.stderr)
    for k in sorted(sampled_strata, key=lambda x: -sampled_strata[x]):
        print(
            f"  {k:<33} {sampled_strata[k]:>8} {strata.get(k, 0):>12}", file=sys.stderr
        )
    print("─" * 57, file=sys.stderr)
    print(f"  {'TOTAL':<33} {len(sampled):>8} {len(segments):>12}", file=sys.stderr)

    # Write as JSON array (annotation tool accepts both JSON array and JSONL)
    with open(args.output, "w") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\nWritten to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
