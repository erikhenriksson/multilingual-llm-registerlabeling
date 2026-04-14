#!/usr/bin/env python3
"""
Multilingual linguistic parser using Stanza.

Reads JSONL files from core_mapped/ directory. Each row is a list of
{"label": "...", "text": "..."} segment dicts. Parses the "text" of each
segment and writes 1:1 matching JSONL rows to parses/ directory.

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
    "cmn_Hant": "zh-hant",
    "eng_Latn": "en",
    "swe_Latn": "sv",
    "pes_Arab": "fa",
    "fin_Latn": "fi",
}

LANG_MAP_LOWER = {k.lower(): v for k, v in LANG_MAP.items()}


def resolve_stanza_lang(filename: str) -> str | None:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    prefix = f"{parts[0]}_{parts[1]}"
    return LANG_MAP_LOWER.get(prefix.lower())


def download_models(lang_codes: list[str]):
    for code in lang_codes:
        log.info(f"Downloading Stanza model: {code}")
        stanza.download(code, logging_level="WARNING")


def build_pipeline(lang_code: str) -> stanza.Pipeline:
    processors = "tokenize,mwt,pos,lemma,depparse"
    log.info(f"Building pipeline for: {lang_code}")
    return stanza.Pipeline(
        lang=lang_code,
        processors=processors,
        logging_level="WARNING",
        use_gpu=True,
    )


def doc_to_dict(doc) -> dict:
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
    log.info(f"Parsing: {input_path.name}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    log.info(f"  {total} rows to parse")

    parsed_count = 0
    segment_count = 0
    error_count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                out.write("[]\n")
                continue

            try:
                segments = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"  Row {i}: invalid JSON ({e}), writing empty list")
                out.write("[]\n")
                error_count += 1
                continue

            if not isinstance(segments, list):
                log.warning(f"  Row {i}: expected list, got {type(segments).__name__}")
                out.write("[]\n")
                error_count += 1
                continue

            parsed_segments = []
            for seg in segments:
                text = seg.get("text", "")
                label = seg.get("label", "")

                if not text or not isinstance(text, str) or not text.strip():
                    parsed_segments.append({"label": label, "text": text, "parse": {}})
                    continue

                try:
                    doc = nlp(text)
                    parse = doc_to_dict(doc)
                except Exception as e:
                    log.warning(f"  Row {i}, segment '{label}': parse failed ({e})")
                    parse = {}

                parsed_segments.append({"label": label, "text": text, "parse": parse})
                segment_count += 1

            out.write(json.dumps(parsed_segments, ensure_ascii=False) + "\n")
            parsed_count += 1

            if (i + 1) % 10 == 0:
                log.info(f"  {i + 1}/{total} rows processed")

    log.info(
        f"  Done: {parsed_count} rows, {segment_count} segments parsed, "
        f"{error_count} errors, {total} total rows written"
    )

    with open(output_path, "r", encoding="utf-8") as f:
        out_count = sum(1 for _ in f)
    assert out_count == total, (
        f"1:1 VIOLATION: input has {total} lines but output has {out_count}"
    )
    log.info(f"  ✓ 1:1 check passed ({total} == {out_count})")


def main():
    parser = argparse.ArgumentParser(
        description="Parse core-mapped JSONL segment files with Stanza"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="core_mapped",
        help="Input directory containing JSONL files (default: core_mapped)",
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
        help="Specific files to parse (default: all .jsonl files in data-dir)",
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

    if args.files:
        files = [data_dir / f for f in args.files]
    else:
        files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        log.error(f"No .jsonl files found in {data_dir}")
        sys.exit(1)

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

    unique_langs = list(set(lc for _, lc in file_lang_pairs))
    download_models(unique_langs)

    pipelines: dict[str, stanza.Pipeline] = {}

    for input_path, lang_code in file_lang_pairs:
        if lang_code not in pipelines:
            if args.cpu:
                pipelines[lang_code] = stanza.Pipeline(
                    lang=lang_code,
                    processors="tokenize,mwt,pos,lemma,depparse",
                    logging_level="WARNING",
                    use_gpu=False,
                )
            else:
                pipelines[lang_code] = build_pipeline(lang_code)

        out_name = input_path.stem.replace("_annotated", "_parsed") + ".jsonl"
        output_path = output_dir / out_name

        parse_file(input_path, output_path, pipelines[lang_code])

    log.info("All files processed.")


if __name__ == "__main__":
    main()
