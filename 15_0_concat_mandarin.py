#!/usr/bin/env python3
"""
Preprocessing: concatenate Mandarin Traditional and Simplified parse files
into a single cmn_parsed.jsonl before MDA.

Rationale: cmn_Hant and cmn_Hans are script variants of the same language
(Mandarin), parsed with the same Stanza 'zh' model. For MDA purposes they
should be treated as one language so the factor analysis sees all Mandarin
data jointly.

Preserves doc_id uniqueness by offsetting doc indices (not strictly needed
since extract_features.py assigns doc_id by line number within a file, but
good to keep a source column so you can trace back).

Input:  parses/cmn_Hant_parsed.jsonl
        parses/cmn_Hans_parsed.jsonl
Output: parses/cmn_parsed.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def concat_files(inputs, output_path):
    total_lines = 0
    per_source = {}
    with open(output_path, "w", encoding="utf-8") as out:
        for src_path in inputs:
            src_name = Path(src_path).stem
            count = 0
            with open(src_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        out.write("[]\n")
                        count += 1
                        continue
                    # Tag each segment with its source script variant.
                    # Parse, inject, re-dump — keeps downstream code simple.
                    try:
                        segments = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning(f"  {src_name}: invalid JSON line, writing []")
                        out.write("[]\n")
                        count += 1
                        continue
                    if isinstance(segments, list):
                        for seg in segments:
                            if isinstance(seg, dict):
                                seg["_source"] = src_name
                    out.write(json.dumps(segments, ensure_ascii=False) + "\n")
                    count += 1
            per_source[src_name] = count
            total_lines += count
            log.info(f"  {src_name}: {count} lines")
    log.info(f"Wrote {total_lines} total lines → {output_path}")
    return per_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parses-dir", default="parses")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=["cmn_Hant_parsed.jsonl", "cmn_Hans_parsed.jsonl"],
        help="File names (in parses-dir) to concatenate.",
    )
    ap.add_argument(
        "--output",
        default="cmn_parsed.jsonl",
        help="Output file name (in parses-dir).",
    )
    args = ap.parse_args()

    parses_dir = Path(args.parses_dir)
    input_paths = [parses_dir / f for f in args.inputs]
    for p in input_paths:
        if not p.exists():
            log.error(f"Missing: {p}")
            return
    output_path = parses_dir / args.output

    log.info(f"Concatenating {len(input_paths)} files → {output_path}")
    concat_files(input_paths, output_path)


if __name__ == "__main__":
    main()
