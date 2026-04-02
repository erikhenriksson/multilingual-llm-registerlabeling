"""
Review register annotations interactively in the terminal.

Usage:
    python review_annotations.py --input sample_10k_annotated.jsonl
    python review_annotations.py --input sample_10k_annotated.jsonl --random
    python review_annotations.py --input sample_10k_annotated.jsonl --start-from 500
    python review_annotations.py --input sample_10k_annotated.jsonl --section comments
    python review_annotations.py --input sample_10k_annotated.jsonl --filter promoting
    python review_annotations.py --input sample_10k_annotated.jsonl --filter transcribed

Controls:
    Enter / n   Next segment
    p           Previous segment
    d           Next document
    D           Previous document
    r           Toggle random mode
    g <N>       Go to document N
    s           Toggle section (main / comments)
    q           Quit
"""

import argparse
import json
import os
import random
import re
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONTEXT_LINES = 5
MAX_DISPLAY_CHARS = 500

# ANSI colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"
C_BG_DARK = "\033[48;5;236m"

# Colors per field_activity
ACTIVITY_COLORS = {
    "explaining": C_BLUE,
    "recounting": C_GREEN,
    "directing": C_CYAN,
    "evaluating": C_YELLOW,
    "promoting": C_RED,
    "creating": C_MAGENTA,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_index(path):
    """Build byte-offset index for a JSONL file."""
    offsets = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    return offsets


def read_doc(path, offset):
    """Read a single document by byte offset."""
    with open(path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline().decode("utf-8"))


def parse_lines(markdown_text):
    """Parse '[N] content' lines into list of {num, text}."""
    if not markdown_text or not markdown_text.strip():
        return []
    lines = []
    for raw in markdown_text.split("\n"):
        m = re.match(r"\[(\d+)\]\s*(.*)", raw)
        if m:
            lines.append({"num": int(m.group(1)), "text": m.group(2)})
    return lines


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def truncate(text, max_chars=MAX_DISPLAY_CHARS):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def format_annotation(ann):
    """One-line annotation summary."""
    if ann is None:
        return f"{C_DIM}(no annotation){C_RESET}"
    medium = ann.get("mode_medium", "?")
    if medium == "cannot_rate" or medium == "structural":
        return f"{C_DIM}cannot_rate{C_RESET}"
    turn = ann.get("mode_turn", "?")
    activity = ann.get("field_activity", "?")
    formality = ann.get("tenor_formality", "?")
    color = ACTIVITY_COLORS.get(activity, C_WHITE)
    structural = " [structural]" if ann.get("is_structural") else ""
    return (
        f"{color}{C_BOLD}{activity}{C_RESET} "
        f"{C_DIM}|{C_RESET} {medium} {C_DIM}|{C_RESET} {turn} "
        f"{C_DIM}|{C_RESET} {formality}{structural}"
    )


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(doc_idx, total_docs, url, section, seg_idx, total_segs, random_mode):
    """Print the top bar."""
    mode_str = f"{C_YELLOW}RANDOM{C_RESET}" if random_mode else "SEQUENTIAL"
    print(f"{C_BOLD}{'─' * 80}{C_RESET}")
    print(
        f"  Doc {C_BOLD}{doc_idx}{C_RESET}/{total_docs}  │  "
        f"Section: {C_BOLD}{section}{C_RESET}  │  "
        f"Segment {C_BOLD}{seg_idx + 1}{C_RESET}/{total_segs}  │  "
        f"Mode: {mode_str}"
    )
    print(f"  {C_DIM}{truncate(url, 76)}{C_RESET}")
    print(f"{C_BOLD}{'─' * 80}{C_RESET}")


def print_segment(lines, annotations, center_idx, context=CONTEXT_LINES):
    """Print a segment with context lines around center_idx."""
    ann_by_line = {}
    for ann in annotations:
        if ann and ann.get("line") is not None:
            ann_by_line[ann["line"]] = ann

    start = max(0, center_idx - context)
    end = min(len(lines), center_idx + context + 1)

    for i in range(start, end):
        line = lines[i]
        num = line["num"]
        text = truncate(line["text"])
        ann = ann_by_line.get(num)

        if i == center_idx:
            # Highlighted target line
            ann_str = format_annotation(ann)
            print(f"\n  {C_BOLD}{C_BG_DARK} [{num:3d}] {text} {C_RESET}")
            print(f"         {ann_str}")
            print()
        else:
            # Context line (dimmed)
            print(f"  {C_DIM} [{num:3d}] {text}{C_RESET}")


def print_controls():
    print(f"{C_DIM}{'─' * 80}{C_RESET}")
    print(
        f"  {C_DIM}Enter/n{C_RESET} next  "
        f"{C_DIM}p{C_RESET} prev  "
        f"{C_DIM}d/D{C_RESET} next/prev doc  "
        f"{C_DIM}r{C_RESET} toggle random  "
        f"{C_DIM}g N{C_RESET} goto doc  "
        f"{C_DIM}s{C_RESET} toggle section  "
        f"{C_DIM}q{C_RESET} quit"
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def matches_filter(ann, filter_value):
    """Check if an annotation matches the filter on any field."""
    if filter_value is None:
        return True
    if ann is None:
        return False
    for key in ("mode_medium", "mode_turn", "field_activity", "tenor_formality"):
        if ann.get(key) == filter_value:
            return True
    return False


def get_matching_indices(lines, annotations, filter_value):
    """Return line indices (into `lines` list) that match the filter."""
    ann_by_line = {}
    for ann in annotations:
        if ann and ann.get("line") is not None:
            ann_by_line[ann["line"]] = ann

    indices = []
    for i, line in enumerate(lines):
        ann = ann_by_line.get(line["num"])
        if filter_value is None:
            # No filter: show every non-structural, non-cannot_rate line
            if ann and ann.get("mode_medium") not in (
                "cannot_rate",
                "structural",
                None,
            ):
                indices.append(i)
        else:
            if matches_filter(ann, filter_value):
                indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Review register annotations")
    parser.add_argument("--input", required=True, help="Annotated JSONL file")
    parser.add_argument("--random", action="store_true", help="Start in random mode")
    parser.add_argument("--start-from", type=int, default=0, help="Starting doc index")
    parser.add_argument(
        "--section",
        default="main",
        choices=["main", "comments"],
        help="Which section to review (default: main)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only show lines matching this value (e.g. 'promoting', 'transcribed', 'informal')",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=CONTEXT_LINES,
        help=f"Number of context lines above/below (default: {CONTEXT_LINES})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Building index for {args.input}...", file=sys.stderr)
    offsets = load_index(args.input)
    total_docs = len(offsets)
    print(f"Loaded {total_docs} documents.", file=sys.stderr)

    if total_docs == 0:
        print("No documents found.", file=sys.stderr)
        sys.exit(1)

    random_mode = args.random
    section = args.section
    doc_idx = args.start_from
    seg_idx = 0
    filter_value = args.filter
    context = args.context

    current_lines = None
    current_annotations = None
    current_matching = None

    def load_doc(idx):
        nonlocal current_lines, current_annotations, current_matching, seg_idx
        doc = read_doc(args.input, offsets[idx])
        md = doc.get(f"markdown_{section}", "")
        current_lines = parse_lines(md)
        ann_data = doc.get("llm_register_annotation", {})
        current_annotations = ann_data.get(section, [])
        current_matching = get_matching_indices(
            current_lines, current_annotations, filter_value
        )
        seg_idx = 0
        return doc

    def find_next_doc_with_matches(start, direction=1):
        """Find the first doc at or after `start` (in `direction`) with matches."""
        for i in range(total_docs):
            idx = (start + i * direction) % total_docs
            doc = read_doc(args.input, offsets[idx])
            md = doc.get(f"markdown_{section}", "")
            lines = parse_lines(md)
            ann_data = doc.get("llm_register_annotation", {})
            anns = ann_data.get(section, [])
            matching = get_matching_indices(lines, anns, filter_value)
            if matching:
                return idx
        return None

    if random_mode:
        doc_idx = random.randint(0, total_docs - 1)
    doc = load_doc(doc_idx)

    if not current_matching:
        next_idx = find_next_doc_with_matches(doc_idx)
        if next_idx is None:
            print("No documents match the filter.", file=sys.stderr)
            sys.exit(1)
        doc_idx = next_idx
        doc = load_doc(doc_idx)

    while True:
        url = doc.get("u", "(no URL)")

        clear_screen()
        print_header(
            doc_idx,
            total_docs,
            url,
            section,
            seg_idx,
            len(current_matching),
            random_mode,
        )

        if current_matching:
            line_idx = current_matching[seg_idx]
            print_segment(current_lines, current_annotations, line_idx, context=context)
        else:
            print(f"\n  {C_DIM}(no matching segments in this doc/section){C_RESET}\n")

        print_controls()

        try:
            raw_cmd = input(f"\n  {C_BOLD}>{C_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        cmd = raw_cmd.lower()

        if cmd in ("q", "quit", "exit"):
            break

        elif cmd in ("", "n", "next"):
            if current_matching and seg_idx < len(current_matching) - 1:
                seg_idx += 1
            else:
                if random_mode:
                    doc_idx = random.randint(0, total_docs - 1)
                else:
                    doc_idx = (doc_idx + 1) % total_docs
                next_idx = find_next_doc_with_matches(doc_idx)
                if next_idx is not None:
                    doc_idx = next_idx
                doc = load_doc(doc_idx)

        elif cmd in ("p", "prev"):
            if current_matching and seg_idx > 0:
                seg_idx -= 1
            else:
                next_idx = find_next_doc_with_matches(doc_idx, direction=-1)
                if next_idx is not None:
                    doc_idx = next_idx
                    doc = load_doc(doc_idx)
                    if current_matching:
                        seg_idx = len(current_matching) - 1

        elif raw_cmd == "D":
            # Previous document (case-sensitive uppercase)
            next_idx = find_next_doc_with_matches(doc_idx, direction=-1)
            if next_idx is not None:
                doc_idx = next_idx
            doc = load_doc(doc_idx)

        elif cmd == "d":
            # Next document
            if random_mode:
                doc_idx = random.randint(0, total_docs - 1)
            else:
                doc_idx = (doc_idx + 1) % total_docs
            next_idx = find_next_doc_with_matches(doc_idx)
            if next_idx is not None:
                doc_idx = next_idx
            doc = load_doc(doc_idx)

        elif cmd in ("r", "random"):
            random_mode = not random_mode
            if random_mode:
                doc_idx = random.randint(0, total_docs - 1)
                next_idx = find_next_doc_with_matches(doc_idx)
                if next_idx is not None:
                    doc_idx = next_idx
                doc = load_doc(doc_idx)

        elif cmd in ("s", "section"):
            section = "comments" if section == "main" else "main"
            doc = load_doc(doc_idx)
            if not current_matching:
                next_idx = find_next_doc_with_matches(doc_idx)
                if next_idx is not None:
                    doc_idx = next_idx
                    doc = load_doc(doc_idx)

        elif cmd.startswith("g"):
            try:
                target = int(raw_cmd.split(None, 1)[1])
                if 0 <= target < total_docs:
                    doc_idx = target
                    doc = load_doc(doc_idx)
                else:
                    print(f"  Index out of range (0-{total_docs - 1})")
                    input("  Press Enter...")
            except (ValueError, IndexError):
                print("  Usage: g <doc_index>")
                input("  Press Enter...")

    print(f"\nReviewed up to doc {doc_idx}. Goodbye!")


if __name__ == "__main__":
    main()
