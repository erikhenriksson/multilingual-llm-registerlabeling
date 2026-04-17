#!/usr/bin/env python3
"""
SMC (Squared Multiple Correlation) pre-filter for MDA features.

For each feature, compute its SMC: the R² of a linear regression predicting
that feature from all other features. Features with low SMC don't share
variance with the rest of the matrix and cannot meaningfully load on common
factors (Biber 1988, following standard FA practice).

Additionally:
  - Drops zero-variance and near-zero-variance features
  - Log-transforms rate features (log1p) to reduce skew before SMC/FA
    (this deviates from strict Biber 1988 but follows modern replications)
  - Iteratively drops extremely collinear features before computing SMC,
    since perfect collinearity makes the correlation matrix singular

Input:  features/{lang}_{script}_features.parquet
Output: features_filtered/{lang}_{script}_filtered.parquet
        reports/{lang}_{script}_smc.csv  (full SMC table for inspection)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = ["doc_id", "source", "segment_idx", "label", "n_tokens", "n_sents"]

# Features that are means/ratios rather than per-1000-token rates.
# Everything else from extract_features.py is a rate and gets log1p.
NON_RATE_FEATURES = {
    "mean_sent_len",
    "mean_dep_dist",
    "mean_word_len",
    "mattr50_words",
    "mattr50_lemmas",
}


def log_transform_rates(X: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to rate features (columns not in NON_RATE_FEATURES)."""
    X = X.copy()
    for col in X.columns:
        if col not in NON_RATE_FEATURES:
            X[col] = np.log1p(X[col].clip(lower=0))
    return X


def drop_low_variance(X: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    stds = X.std()
    drop = stds[stds < threshold].index.tolist()
    if drop:
        log.info(f"  Dropping {len(drop)} zero/near-zero variance features:")
        for c in drop:
            log.info(f"    {c} (std={stds[c]:.2e})")
    return X.drop(columns=drop)


def drop_collinear(X: pd.DataFrame, r_threshold: float = 0.98) -> pd.DataFrame:
    """Greedy removal of features with |r| > threshold against another kept feature.

    For each pair above threshold, drop the one with lower variance (less
    informative). Prevents singular correlation matrix before SMC.
    """
    corr = X.corr().abs()
    n = len(corr)
    cols = list(corr.columns)
    keep = set(cols)
    stds = X.std()

    # Upper triangle pairs above threshold, sorted by correlation descending
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = cols[i], cols[j]
            r = corr.iloc[i, j]
            if r > r_threshold:
                pairs.append((r, ci, cj))
    pairs.sort(reverse=True)

    for r, ci, cj in pairs:
        if ci not in keep or cj not in keep:
            continue
        # Drop the one with lower variance (less unique info)
        drop = ci if stds[ci] < stds[cj] else cj
        keep.discard(drop)
        log.info(f"  Collinear (r={r:.3f}): dropping {drop} (keeping other)")

    return X[[c for c in cols if c in keep]]


def compute_smc(X: pd.DataFrame) -> pd.Series:
    """SMC_i = 1 - 1/C⁻¹[i,i] where C is the correlation matrix.

    Uses pseudo-inverse for robustness, but by this point we should have
    already removed collinear features so the matrix is well-conditioned.
    """
    C = X.corr().values
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        log.warning("  Correlation matrix singular; falling back to pinv")
        C_inv = np.linalg.pinv(C)

    diag = np.diag(C_inv)
    # Guard: diag should be >= 1 in theory; if < 1 due to numerical issues, clip
    diag = np.where(diag < 1.0, 1.0, diag)
    smc = 1.0 - 1.0 / diag
    smc = np.clip(smc, 0.0, 1.0)
    return pd.Series(smc, index=X.columns, name="SMC")


def process_file(
    input_path: Path,
    filtered_out: Path,
    report_out: Path,
    smc_threshold: float,
    collinearity_threshold: float,
    log_transform: bool,
):
    log.info(f"Filtering: {input_path.name}")
    df = pd.read_parquet(input_path)
    log.info(f"  Input: {len(df)} rows, {len(df.columns)} cols")

    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    X = df.drop(columns=[c for c in META_COLS if c in df.columns])
    log.info(f"  {X.shape[1]} candidate features")

    # 1. Drop zero-variance
    X = drop_low_variance(X)
    log.info(f"  After variance filter: {X.shape[1]} features")

    # 2. Log-transform rate features
    if log_transform:
        X = log_transform_rates(X)
        log.info("  Applied log1p to rate features")

    # 3. Drop extremely collinear features
    X = drop_collinear(X, r_threshold=collinearity_threshold)
    log.info(f"  After collinearity filter: {X.shape[1]} features")

    # 4. Compute SMC and drop below threshold
    smc = compute_smc(X)
    smc_sorted = smc.sort_values(ascending=False)

    # Write full SMC report before filtering
    report_out.parent.mkdir(parents=True, exist_ok=True)
    smc_sorted.to_csv(report_out, header=True)
    log.info(f"  SMC report → {report_out}")

    keep = smc[smc >= smc_threshold].index.tolist()
    drop = smc[smc < smc_threshold].index.tolist()

    log.info(f"  SMC threshold: {smc_threshold}")
    log.info(f"  Keeping {len(keep)} features; dropping {len(drop)}")
    if drop:
        for c in sorted(drop, key=lambda c: smc[c]):
            log.info(f"    drop: {c} (SMC={smc[c]:.3f})")

    X_final = X[keep]

    # Recombine with metadata
    out = pd.concat(
        [meta.reset_index(drop=True), X_final.reset_index(drop=True)], axis=1
    )
    filtered_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(filtered_out, index=False)
    log.info(f"  Wrote {len(out)} rows × {X_final.shape[1]} features → {filtered_out}")

    # Summary stats
    log.info(
        f"  SMC summary: min={smc.min():.3f}, median={smc.median():.3f}, max={smc.max():.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--filtered-dir", default="features_filtered")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument(
        "--smc-threshold",
        type=float,
        default=0.20,
        help="Drop features below this SMC. Biber used .30; .20 is gentler.",
    )
    ap.add_argument(
        "--collinearity-threshold",
        type=float,
        default=0.99,
        help="Drop feature pairs correlated above this before SMC.",
    )
    ap.add_argument(
        "--no-log-transform",
        action="store_true",
        help="Skip log1p transform of rate features (stricter Biber 1988).",
    )
    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    if args.files:
        files = [features_dir / f for f in args.files]
    else:
        files = sorted(features_dir.glob("*_features.parquet"))

    if not files:
        log.error(f"No *_features.parquet files in {features_dir}")
        return

    for f in files:
        stem = f.stem.replace("_features", "")
        filtered_out = Path(args.filtered_dir) / f"{stem}_filtered.parquet"
        report_out = Path(args.reports_dir) / f"{stem}_smc.csv"
        process_file(
            f,
            filtered_out,
            report_out,
            smc_threshold=args.smc_threshold,
            collinearity_threshold=args.collinearity_threshold,
            log_transform=not args.no_log_transform,
        )


if __name__ == "__main__":
    main()
