#!/usr/bin/env python3
"""
Compare factor solutions at multiple k values.

Parallel analysis massively over-retains at large N. Biber-style MDA
conventionally uses 3-7 factors chosen by scree + interpretability. This
script fits the FA at several candidate k values and reports diagnostics
so you can pick one by inspection.

For each k:
  - cumulative variance explained
  - number of features with |loading| >= threshold on ANY factor ("salient")
  - number of factors with at least min_feats salient features
    ("interpretable factors")
  - number of complex features (salient on multiple factors)

Input:  features_filtered/{lang}_{script}_filtered.parquet
Output: reports/{lang}_{script}_k_comparison.csv
        (console table)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = ["doc_id", "source", "segment_idx", "label", "n_tokens", "n_sents"]


def evaluate_k(X_z, k, rotation, loading_threshold, min_feats_per_factor):
    fa = FactorAnalyzer(n_factors=k, rotation=rotation, method="principal")
    fa.fit(X_z)

    loadings = np.abs(fa.loadings_)
    _, _, cum_var = fa.get_factor_variance()
    cumvar = cum_var[-1]

    # Salient features: any |loading| >= threshold on any factor
    max_load_per_feat = loadings.max(axis=1)
    n_salient_feats = int(np.sum(max_load_per_feat >= loading_threshold))

    # Interpretable factors: factors with at least min_feats salient features
    salient_matrix = loadings >= loading_threshold
    feats_per_factor = salient_matrix.sum(axis=0)
    n_interpretable = int(np.sum(feats_per_factor >= min_feats_per_factor))

    # Complex features (loading salient on 2+ factors → cross-loading)
    complexity = salient_matrix.sum(axis=1)
    n_complex = int(np.sum(complexity >= 2))

    # Smallest factor size (how marginal is the last factor?)
    min_factor_size = int(feats_per_factor.min()) if len(feats_per_factor) else 0

    return {
        "k": k,
        "cumulative_var": round(cumvar, 3),
        "salient_features": n_salient_feats,
        "interpretable_factors": n_interpretable,
        "complex_features": n_complex,
        "min_factor_size": min_factor_size,
        "feats_per_factor": feats_per_factor.tolist(),
    }


def process_file(
    input_path, report_path, k_values, rotation, loading_threshold, min_feats_per_factor
):
    log.info(f"Comparing k values: {input_path.name}")
    df = pd.read_parquet(input_path)
    X = df.drop(columns=[c for c in META_COLS if c in df.columns])
    log.info(f"  {X.shape[0]} segments × {X.shape[1]} features")

    X_z = (X - X.mean()) / X.std()
    X_z = X_z.dropna(axis=1, how="any")

    rows = []
    for k in k_values:
        log.info(f"  Fitting k = {k}")
        try:
            row = evaluate_k(X_z, k, rotation, loading_threshold, min_feats_per_factor)
            rows.append(row)
        except Exception as e:
            log.warning(f"    k={k} failed: {e}")

    out = pd.DataFrame(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(report_path, index=False)

    # Print table
    print(f"\n=== {input_path.stem} ===")
    print(
        f"(rotation={rotation}, loading threshold={loading_threshold}, "
        f"min_feats_per_factor={min_feats_per_factor})\n"
    )
    print(out.drop(columns=["feats_per_factor"]).to_string(index=False))
    print()
    print("Salient features per factor (one list per k):")
    for _, r in out.iterrows():
        print(f"  k={int(r['k']):2d}: {r['feats_per_factor']}")
    print(f"\nReport → {report_path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered-dir", default="features_filtered")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 10],
        help="Candidate k values to compare.",
    )
    ap.add_argument("--rotation", default="promax")
    ap.add_argument(
        "--loading-threshold",
        type=float,
        default=0.35,
        help="|loading| above this is 'salient' (Biber used .35).",
    )
    ap.add_argument(
        "--min-feats-per-factor",
        type=int,
        default=3,
        help="A factor is 'interpretable' if it has at least this many salient features.",
    )
    args = ap.parse_args()

    filtered_dir = Path(args.filtered_dir)
    if args.files:
        files = [filtered_dir / f for f in args.files]
    else:
        files = sorted(filtered_dir.glob("*_filtered.parquet"))

    for f in files:
        stem = f.stem.replace("_filtered", "")
        report_path = Path(args.reports_dir) / f"{stem}_k_comparison.csv"
        process_file(
            f,
            report_path,
            k_values=args.k_values,
            rotation=args.rotation,
            loading_threshold=args.loading_threshold,
            min_feats_per_factor=args.min_feats_per_factor,
        )


if __name__ == "__main__":
    main()
