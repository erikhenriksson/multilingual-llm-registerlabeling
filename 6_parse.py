#!/usr/bin/env python3
"""
Multilingual linguistic parser using Stanza.

Reads JSONL files from data/ directory, parses the "text" field,
and writes 1:1 matching JSONL rows to parses/ directory.

File naming convention: {lang}_{script}_annotated.jsonl
Output: parses/{lang}_{script}_parsed.jsonl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import stanza

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Map your filename prefixes to Stanza language codes ──────────────────────
LANG_MAP = {
    #"cmn_Hans": "zh-hans",  # Simplified Chinese
    "cmn_Hant": "zh-hant",  # Traditional Chinese
    "eng_Latn": "en",  # English
    "swe_Latn": "sv",  # Swedish
    "pes_Arab": "fa",  # Persian (Farsi)
    "fin_Latn": "fi",  # Finnish
}

# Case-insensitive lookup (your files use "pes_arab" not "pes_Arab")
LANG_MAP_LOWER = {k.lower(): v for k, v in LANG_MAP.items()}


def resolve_stanza_lang(filename: str) -> str | None:
    """Extract the language prefix from filename and map to Stanza code."""
    # e.g. "cmn_Hans_annotated.jsonl" → "cmn_Hans"
    stem = Path(filename).stem  # "cmn_Hans_annotated"
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    prefix = f"{parts[0]}_{parts[1]}"
    return LANG_MAP_LOWER.get(prefix.lower())


def download_models(lang_codes: list[str]):
    """Pre-download all needed Stanza models."""
    for code in lang_codes:
        log.info(f"Downloading Stanza model: {code}")
        stanza.download(code, logging_level="WARNING")


def build_pipeline(lang_code: str) -> stanza.Pipeline:
    """Build a Stanza pipeline for the given language."""
    processors = "tokenize,mwt,pos,lemma,depparse"
    log.info(f"Building pipeline for: {lang_code}")
    return stanza.Pipeline(
        lang=lang_code,
        processors=processors,
        logging_level="WARNING",
        use_gpu=True,  # falls back to CPU if unavailable
    )


def doc_to_dict(doc) -> dict:
    """
    Convert a Stanza Document to a dictionary.

    Structure:
    {
        "sentences": [
            {
                "text": "The cat sat.",
                "tokens": [
                    {
                        "id": 1,
                        "text": "The",
                        "lemma": "the",
                        "upos": "DET",
                        "xpos": "DT",
                        "feats": "Definite=Def|PronType=Art",
                        "head": 2,
                        "deprel": "det",
                        "start_char": 0,
                        "end_char": 3
                    },
                    ...
                ]
            },
            ...
        ]
    }
    """
    sentences = []
    for sent in doc.sentences:
        tokens = []
        for word in sent.words:
            tokens.append(
                {
                    "id": word.id,
                    "text": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats if word.feats else None,
                    "head": word.head,
                    "deprel": word.deprel,
                    "start_char": word.start_char,
                    "end_char": word.end_char,
                }
            )
        sentences.append(
            {
                "text": sent.text,
                "tokens": tokens,
            }
        )
    return {"sentences": sentences}


def parse_file(input_path: Path, output_path: Path, nlp: stanza.Pipeline):
    """Parse a single JSONL file, writing 1:1 output rows."""
    log.info(f"Parsing: {input_path.name}")

    # Read all lines first to know total count
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    log.info(f"  {total} rows to parse")

    parsed_count = 0
    error_count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # Empty line in input → empty dict in output (preserve 1:1)
                out.write("{}\n")
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"  Row {i}: invalid JSON ({e}), writing empty dict")
                out.write("{}\n")
                error_count += 1
                continue

            text = row.get("text", "")
            if not text or not isinstance(text, str) or not text.strip():
                # No text to parse → empty dict
                out.write("{}\n")
                error_count += 1
                continue

            try:
                doc = nlp(text)
                result = doc_to_dict(doc)
                result["_source_index"] = i  # for traceability
            except Exception as e:
                log.warning(f"  Row {i}: parse failed ({e}), writing empty dict")
                result = {}
                error_count += 1

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            parsed_count += 1

            if (i + 1) % 500 == 0:
                log.info(f"  {i + 1}/{total} rows processed")

    log.info(
        f"  Done: {parsed_count} parsed, {error_count} errors/empty, "
        f"{total} total rows written"
    )

    # Sanity check: output lines == input lines
    with open(output_path, "r", encoding="utf-8") as f:
        out_count = sum(1 for _ in f)
    assert out_count == total, (
        f"1:1 VIOLATION: input has {total} lines but output has {out_count}"
    )
    log.info(f"  ✓ 1:1 check passed ({total} == {out_count})")


def main():
    parser = argparse.ArgumentParser(description="Parse JSONL files with Stanza")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Input directory containing *_annotated.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="parses",
        help="Output directory for parsed JSONL files",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=None,
        help="Specific files to parse (default: all matching files in data-dir)",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if GPU is available"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        log.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Discover files
    if args.files:
        files = [data_dir / f for f in args.files]
    else:
        files = sorted(data_dir.glob("*_annotated.jsonl"))

    if not files:
        log.error(f"No *_annotated.jsonl files found in {data_dir}")
        sys.exit(1)

    # Resolve languages and check all are supported
    file_lang_pairs = []
    for f in files:
        lang_code = resolve_stanza_lang(f.name)
        if lang_code is None:
            log.error(
                f"Cannot determine language for: {f.name}\n"
                f"  Known prefixes: {list(LANG_MAP.keys())}"
            )
            sys.exit(1)
        file_lang_pairs.append((f, lang_code))

    log.info(f"Files to process:")
    for f, lc in file_lang_pairs:
        log.info(f"  {f.name} → Stanza lang: {lc}")

    # Download all needed models
    unique_langs = list(set(lc for _, lc in file_lang_pairs))
    download_models(unique_langs)

    # Process files, reusing pipelines per language
    pipelines: dict[str, stanza.Pipeline] = {}

    for input_path, lang_code in file_lang_pairs:
        if lang_code not in pipelines:
            if args.cpu:
                import torch

                # override
                pipelines[lang_code] = stanza.Pipeline(
                    lang=lang_code,
                    processors="tokenize,mwt,pos,lemma,depparse",
                    logging_level="WARNING",
                    use_gpu=False,
                )
            else:
                pipelines[lang_code] = build_pipeline(lang_code)

        # Output filename: replace _annotated with _parsed
        out_name = input_path.name.replace("_annotated.jsonl", "_parsed.jsonl")
        output_path = output_dir / out_name

        parse_file(input_path, output_path, pipelines[lang_code])

    log.info("All files processed.")


if __name__ == "__main__":
    main()
