#!/usr/bin/env python3
"""
Extract MDA features from Stanza UD parses.

Feature philosophy: enumerate everything actually present in the data.
  - Every upos value → one rate feature
  - Every deprel value (full, including subtypes) → one rate feature
  - Every (feats_key, feats_value) pair → one rate feature
  - Plus non-enumerable structural features (sentence length, dep distance,
    MATTR, word length)

No hand-curated feature list. The SMC filter and factor analysis do the
feature selection downstream.

Two-pass design:
  Pass 1: scan all files for the union of tag vocabularies (per file).
  Pass 2: compute features using that vocabulary.

Input:  parses/{lang}_{script}_parsed.jsonl
Output: features/{lang}_{script}_features.parquet
        features/{lang}_{script}_vocabulary.json  (what was discovered)
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Merge groups ─────────────────────────────────────────────────────────────
# Map file prefix (e.g. "cmn_Hant") → group name for output.
# Files sharing the same group name are concatenated into a single feature
# matrix. Prefixes not listed here become their own group.
MERGE_GROUPS = {
    "cmn_Hant": "cmn",
    "cmn_Hans": "cmn",
}


def prefix_of(path):
    """Extract 'lang_Script' prefix from 'lang_Script_parsed.jsonl'."""
    stem = path.stem.replace("_parsed", "")
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def group_of(path):
    """Group name for a file (from MERGE_GROUPS, else the file's own prefix)."""
    pref = prefix_of(path)
    return MERGE_GROUPS.get(pref, pref)


def parse_feats(feats_str):
    """Parse UD feats string 'Mood=Ind|Number=Sing|...' into list of (k,v)."""
    if not feats_str:
        return []
    out = []
    for kv in feats_str.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out.append((k, v))
    return out


def mattr(tokens, window=50):
    """Moving-Average Type-Token Ratio. Length-robust.

    If shorter than window, fall back to overall TTR.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n < window:
        return len(set(tokens)) / n
    total = 0.0
    for i in range(n - window + 1):
        total += len(set(tokens[i : i + window])) / window
    return total / (n - window + 1)


def discover_vocabulary(input_paths):
    """Pass 1: collect all upos, deprel, (feat_key, feat_val) values
    across one or more input files."""
    upos_set = set()
    deprel_set = set()
    feat_pair_set = set()

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    segments = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(segments, list):
                    continue
                for seg in segments:
                    parse = seg.get("parse") or {}
                    for sent in parse.get("sentences", []):
                        for tok in sent.get("tokens", []):
                            if tok.get("upos"):
                                upos_set.add(tok["upos"])
                            if tok.get("deprel"):
                                deprel_set.add(tok["deprel"])
                            for k, v in parse_feats(tok.get("feats")):
                                feat_pair_set.add((k, v))

    return {
        "upos": sorted(upos_set),
        "deprel": sorted(deprel_set),
        "feats": sorted(feat_pair_set),
    }


def safe_col(prefix, name):
    """Build a safe column name: replace UD separators with underscore."""
    return f"{prefix}_{name}".replace(":", "_").replace("=", "_").replace("|", "_")


def extract_segment_features(parse, vocab):
    """Extract features from one parsed segment using the discovered vocab.

    Returns (features_dict, n_content_tokens, n_sents) or None if empty.
    """
    sentences = parse.get("sentences") or []
    if not sentences:
        return None

    upos_counts = Counter()
    deprel_counts = Counter()
    feat_counts = Counter()

    word_tokens = []
    lemma_tokens = []
    word_lens = []
    dep_distances = []
    sent_lengths = []
    n_tokens_total = 0
    n_punct = 0

    for sent in sentences:
        tokens = sent.get("tokens") or []
        if not tokens:
            continue
        sent_content = 0

        for tok in tokens:
            n_tokens_total += 1
            upos = tok.get("upos") or "X"
            deprel = tok.get("deprel") or ""

            upos_counts[upos] += 1
            if deprel:
                deprel_counts[deprel] += 1

            for k, v in parse_feats(tok.get("feats")):
                feat_counts[(k, v)] += 1

            # Dep distance (skip root; skip multiword-token tuple ids)
            head = tok.get("head")
            tid = tok.get("id")
            if isinstance(head, int) and head > 0 and isinstance(tid, int):
                dep_distances.append(abs(head - tid))

            if upos == "PUNCT":
                n_punct += 1
            else:
                sent_content += 1
                text = tok.get("text") or ""
                lemma = tok.get("lemma") or text
                if text:
                    word_tokens.append(text.lower())
                    word_lens.append(len(text))
                if lemma:
                    lemma_tokens.append(lemma.lower())

        if sent_content > 0:
            sent_lengths.append(sent_content)

    n_content = n_tokens_total - n_punct
    n_sents = len(sent_lengths)
    if n_content == 0:
        return None

    def rate(count):
        return 1000.0 * count / n_content

    feats_out = {}

    # Enumerated tag rates (always present, zero if tag didn't occur in segment)
    for upos in vocab["upos"]:
        feats_out[safe_col("pos", upos.lower())] = rate(upos_counts.get(upos, 0))
    for dr in vocab["deprel"]:
        feats_out[safe_col("dep", dr)] = rate(deprel_counts.get(dr, 0))
    for k, v in vocab["feats"]:
        feats_out[safe_col("morph", f"{k}_{v}")] = rate(feat_counts.get((k, v), 0))

    # Structural features (not tag-enumerable)
    feats_out["mean_sent_len"] = sum(sent_lengths) / n_sents if n_sents else 0.0
    feats_out["mean_dep_dist"] = (
        sum(dep_distances) / len(dep_distances) if dep_distances else 0.0
    )
    feats_out["mean_word_len"] = sum(word_lens) / len(word_lens) if word_lens else 0.0
    feats_out["mattr50_words"] = mattr(word_tokens, window=50)
    feats_out["mattr50_lemmas"] = mattr(lemma_tokens, window=50)

    return feats_out, n_content, n_sents


def process_group(group_name, input_paths, output_path, vocab_path, min_tokens):
    """Process one language group. Multiple inputs are concatenated
    with source-prefixed doc_ids to keep documents distinct.
    """
    log.info(f"Processing group: {group_name}")
    for p in input_paths:
        log.info(f"  Input: {p.name}")

    # Pass 1
    log.info("  Pass 1: discovering tag vocabulary")
    vocab = discover_vocabulary(input_paths)
    log.info(
        f"  Found: {len(vocab['upos'])} UPOS, "
        f"{len(vocab['deprel'])} deprels, "
        f"{len(vocab['feats'])} morph (key,val) pairs"
    )

    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "group": group_name,
                "source_files": [p.name for p in input_paths],
                "upos": vocab["upos"],
                "deprel": vocab["deprel"],
                "feats": [list(p) for p in vocab["feats"]],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info(f"  Vocabulary → {vocab_path}")

    # Pass 2
    log.info("  Pass 2: computing features")
    rows = []
    for input_path in input_paths:
        # Use file stem (minus "_parsed") as a source tag for doc_id uniqueness
        source_tag = input_path.stem.replace("_parsed", "")
        n_docs_from_file = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for doc_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    segments = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(f"  {source_tag} doc {doc_idx}: invalid JSON, skipping")
                    continue
                if not isinstance(segments, list):
                    continue

                # Unique doc_id across merged sources
                doc_id = f"{source_tag}:{doc_idx}"

                for seg_idx, seg in enumerate(segments):
                    parse = seg.get("parse") or {}
                    label = seg.get("label", "")
                    if not parse:
                        continue

                    result = extract_segment_features(parse, vocab)
                    if result is None:
                        continue
                    feats, n_tok, n_sents = result
                    if n_tok < min_tokens:
                        continue

                    row = {
                        "doc_id": doc_id,
                        "source": source_tag,
                        "segment_idx": seg_idx,
                        "label": label,
                        "n_tokens": n_tok,
                        "n_sents": n_sents,
                        **feats,
                    }
                    rows.append(row)

                n_docs_from_file = doc_idx + 1
                if (doc_idx + 1) % 500 == 0:
                    log.info(
                        f"  {source_tag}: {doc_idx + 1} docs, "
                        f"{len(rows)} total segments so far"
                    )
        log.info(f"  {source_tag}: {n_docs_from_file} docs read")

    if not rows:
        log.error(f"  No segments extracted for group {group_name}")
        return

    df = pd.DataFrame(rows)
    meta_cols = ["doc_id", "source", "segment_idx", "label", "n_tokens", "n_sents"]
    feat_cols = sorted(c for c in df.columns if c not in meta_cols)
    df = df[meta_cols + feat_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info(f"  Wrote {len(df)} segments × {len(feat_cols)} features → {output_path}")
    log.info(f"  Labels: {dict(Counter(df['label']))}")
    if len(input_paths) > 1:
        log.info(f"  Per-source: {dict(Counter(df['source']))}")
    log.info(
        f"  Token stats: min={df['n_tokens'].min()}, "
        f"median={int(df['n_tokens'].median())}, "
        f"max={df['n_tokens'].max()}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parses-dir", default="parses")
    ap.add_argument("--output-dir", default="features")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument(
        "--skip",
        nargs="*",
        default=["cmn_Hant_parsed.jsonl", "cmn_Hans_parsed.jsonl"],
        help="Filenames to skip. Default skips cmn_Hant and cmn_Hans since "
        "they should be merged via concat_mandarin.py into cmn_parsed.jsonl. "
        "Pass --skip with no arguments to disable.",
    )
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Drop segments with fewer than this many content tokens. "
        "Recommend 50-100 for stable rate features.",
    )
    args = ap.parse_args()

    parses_dir = Path(args.parses_dir)
    output_dir = Path(args.output_dir)

    if args.files:
        files = [parses_dir / f for f in args.files]
    else:
        files = sorted(parses_dir.glob("*_parsed.jsonl"))

    if args.skip:
        skip_set = set(args.skip)
        before = len(files)
        files = [f for f in files if f.name not in skip_set]
        log.info(f"Skipping {before - len(files)} file(s): {sorted(skip_set)}")

    if not files:
        log.error(f"No *_parsed.jsonl files in {parses_dir}")
        return

    # Group files by language (merging Hans/Hant, etc.)
    groups = {}
    for f in files:
        gname = group_of(f)
        groups.setdefault(gname, []).append(f)

    log.info("File groupings:")
    for gname, group_files in sorted(groups.items()):
        log.info(f"  {gname}: {[f.name for f in group_files]}")

    for gname, group_files in sorted(groups.items()):
        out = output_dir / f"{gname}_features.parquet"
        vocab_out = output_dir / f"{gname}_vocabulary.json"
        process_group(gname, group_files, out, vocab_out, min_tokens=args.min_tokens)


if __name__ == "__main__":
    main()
