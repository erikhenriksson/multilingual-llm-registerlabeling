"""
Add CORE register labels + segment text to annotated JSONL rows.

Merges main + comments sections, derives CORE labels from facet annotations,
and collapses consecutive segments sharing the same CORE label.

Usage:
    python add_core_labels.py --input annotated.jsonl
    # Output goes to core_mapped/<original_filename>
"""

import argparse
import json
import os
import re
import sys

ACTIVITY_TO_REGISTER = {
    "recounting": "Narrative",
    "explaining": "Informational Description",
    "evaluating": "Opinion",
    "promoting": "Informational Persuasion",
    "directing": "How-to or Instructional",
    "creating": "Lyrical",
}


def derive_core_label(ann):
    medium = ann.get("mode_medium")
    turn = ann.get("mode_turn")
    activity = ann.get("field_activity")

    if medium == "cannot_rate":
        return "Cannot rate"
    if medium == "transcribed":
        return "Spoken"
    if turn == "dialogic":
        return "Interactive Discussion"
    if activity in ACTIVITY_TO_REGISTER:
        return ACTIVITY_TO_REGISTER[activity]
    return "Cannot rate"


def parse_lines(markdown_text):
    if not markdown_text or not markdown_text.strip():
        return []
    result = []
    for raw in markdown_text.split("\n"):
        m = re.match(r"\[(\d+)\]\s*(.*)", raw)
        if m:
            result.append((int(m.group(1)), m.group(2)))
    return result


def build_segments(markdown_text, annotations):
    lines = parse_lines(markdown_text)
    if not lines:
        return []

    ann_by_line = {a["line"]: a for a in annotations}

    segments = []
    for line_num, text in lines:
        ann = ann_by_line.get(line_num)
        if ann:
            label = derive_core_label(ann)
        else:
            label = "Cannot rate"
        segments.append((label, text))
    return segments


def merge_consecutive(segments):
    if not segments:
        return []

    merged = []
    current_label, current_text = segments[0]

    for label, text in segments[1:]:
        if label == current_label:
            current_text += " " + text
        else:
            merged.append({"label": current_label, "text": current_text})
            current_label = label
            current_text = text

    merged.append({"label": current_label, "text": current_text})
    return merged


def process_row(row):
    ann = row.get("llm_register_annotation", {})

    main_segments = build_segments(
        row.get("markdown_main", ""),
        ann.get("main", []),
    )
    comments_segments = build_segments(
        row.get("markdown_comments", ""),
        ann.get("comments", []),
    )

    all_segments = main_segments + comments_segments
    return merge_consecutive(all_segments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    os.makedirs("core_mapped", exist_ok=True)
    output_path = os.path.join("core_mapped", os.path.basename(args.input))

    count = 0
    with open(args.input) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            core_segments = process_row(row)
            fout.write(json.dumps(core_segments, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  {count} docs processed", file=sys.stderr)

    print(f"Done. {count} docs written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
