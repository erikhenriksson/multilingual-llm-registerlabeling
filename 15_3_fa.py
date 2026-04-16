#!/usr/bin/env python3
"""
Biber-style factor analysis on filtered MDA features.

Pipeline:
  1. Standardize features (z-score)
  2. KMO and Bartlett sanity checks
  3. Determine number of factors via parallel analysis (Horn 1965)
     + scree eigenvalues for comparison
  4. Fit principal-axis factor analysis with promax rotation
  5. Save loadings, communalities, factor scores, factor correlations

Input:  features_filtered/{lang}_{script}_filtered.parquet
Output: fa_results/{lang}_{script}/
          loadings.csv
          communalities.csv
          scores.parquet       (one row per segment, factor scores + metadata)
          factor_correlations.csv
          scree_and_parallel.csv
          fa_summary.txt

Notes:
  - Uses factor_analyzer package (proper PAF), not sklearn's FactorAnalysis (ML).
  - If --n-factors is not given, picks k from parallel analysis.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = ["doc_id", "segment_idx", "label", "n_tokens", "n_sents"]


def parallel_analysis(
    X: np.ndarray,
    n_iter: int = 100,
    percentile: float = 95.0,
    seed: int = 0,
) -> np.ndarray:
    """Horn's parallel analysis.

    Generate n_iter random matrices of the same shape as X, compute their
    eigenvalues, return the percentile'th percentile for each position.
    Real eigenvalues above this threshold correspond to retainable factors.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    eigs_sim = np.zeros((n_iter, p))
    for i in range(n_iter):
        R = rng.standard_normal((n, p))
        # Use correlation matrix eigenvalues (matches our FA input)
        corr = np.corrcoef(R, rowvar=False)
        ev = np.linalg.eigvalsh(corr)[::-1]  # descending
        eigs_sim[i] = ev
    return np.percentile(eigs_sim, percentile, axis=0)


def pick_n_factors(real_ev: np.ndarray, pa_ev: np.ndarray) -> int:
    """Number of real eigenvalues exceeding the parallel-analysis threshold."""
    k = int(np.sum(real_ev > pa_ev))
    return max(k, 1)


def process_file(
    input_path: Path,
    output_dir: Path,
    n_factors: int | None,
    pa_iter: int,
    pa_percentile: float,
    rotation: str,
    seed: int,
):
    log.info(f"FA: {input_path.name}")
    df = pd.read_parquet(input_path)
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    X = df.drop(columns=[c for c in META_COLS if c in df.columns])
    log.info(f"  {X.shape[0]} segments × {X.shape[1]} features")

    if X.shape[0] < 10 * X.shape[1]:
        log.warning(
            f"  N/p ratio = {X.shape[0] / X.shape[1]:.1f}; "
            "FA prefers at least 5–10. Results may be unstable."
        )

    # Standardize (FA is scale-invariant under correlation, but z-scoring
    # makes factor scores comparable and guards against unit pathologies)
    X_z = (X - X.mean()) / X.std()
    # Drop any columns that went NaN (shouldn't happen after SMC, but guard)
    bad = X_z.columns[X_z.isna().any()].tolist()
    if bad:
        log.warning(f"  Dropping NaN cols after z-score: {bad}")
        X_z = X_z.drop(columns=bad)

    # Sanity checks
    try:
        chi2, p_val = calculate_bartlett_sphericity(X_z)
        log.info(f"  Bartlett: chi²={chi2:.1f}, p={p_val:.3g}")
        if p_val > 0.05:
            log.warning("  Bartlett p > .05: data may not be factorable")
    except Exception as e:
        log.warning(f"  Bartlett test failed: {e}")

    try:
        kmo_per, kmo_total = calculate_kmo(X_z)
        log.info(f"  KMO total: {kmo_total:.3f} (>.6 acceptable, >.8 good)")
        if kmo_total < 0.5:
            log.warning("  KMO < .5: FA not recommended")
    except Exception as e:
        log.warning(f"  KMO test failed: {e}")

    # Eigenvalues (unrotated, all factors) for scree + parallel analysis
    fa_full = FactorAnalyzer(n_factors=X_z.shape[1], rotation=None, method="principal")
    fa_full.fit(X_z)
    real_ev, _ = fa_full.get_eigenvalues()  # original eigenvalues of corr
    pa_ev = parallel_analysis(
        X_z.values, n_iter=pa_iter, percentile=pa_percentile, seed=seed
    )

    scree_df = pd.DataFrame(
        {
            "factor": np.arange(1, len(real_ev) + 1),
            "eigenvalue": real_ev,
            "parallel_threshold": pa_ev,
            "retain_parallel": real_ev > pa_ev,
            "retain_kaiser": real_ev > 1.0,
        }
    )

    k_parallel = pick_n_factors(real_ev, pa_ev)
    k_kaiser = int(np.sum(real_ev > 1.0))
    log.info(f"  Parallel analysis suggests k = {k_parallel}")
    log.info(f"  Kaiser criterion suggests k = {k_kaiser}")

    if n_factors is None:
        k = k_parallel
        log.info(f"  Using k = {k} (from parallel analysis)")
    else:
        k = n_factors
        log.info(f"  Using k = {k} (user-specified)")

    # Fit PAF with chosen k and rotation
    fa = FactorAnalyzer(n_factors=k, rotation=rotation, method="principal")
    fa.fit(X_z)

    # Loadings
    loadings = pd.DataFrame(
        fa.loadings_,
        index=X_z.columns,
        columns=[f"F{i + 1}" for i in range(k)],
    )

    # Communalities
    communalities = pd.Series(
        fa.get_communalities(), index=X_z.columns, name="communality"
    )

    # Factor scores (regression method, per factor_analyzer default)
    scores = fa.transform(X_z)
    scores_df = pd.DataFrame(scores, columns=[f"F{i + 1}" for i in range(k)])
    scores_out = pd.concat(
        [meta.reset_index(drop=True), scores_df.reset_index(drop=True)], axis=1
    )

    # Factor correlations (only meaningful for oblique rotation)
    if rotation == "promax" and fa.phi_ is not None:
        factor_corr = pd.DataFrame(
            fa.phi_,
            index=[f"F{i + 1}" for i in range(k)],
            columns=[f"F{i + 1}" for i in range(k)],
        )
    else:
        factor_corr = None

    # Variance explained
    var, prop_var, cum_var = fa.get_factor_variance()
    var_df = pd.DataFrame(
        {
            "factor": [f"F{i + 1}" for i in range(k)],
            "SS_loadings": var,
            "proportion_var": prop_var,
            "cumulative_var": cum_var,
        }
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    loadings.to_csv(output_dir / "loadings.csv")
    communalities.to_csv(output_dir / "communalities.csv", header=True)
    scores_out.to_parquet(output_dir / "scores.parquet", index=False)
    scree_df.to_csv(output_dir / "scree_and_parallel.csv", index=False)
    var_df.to_csv(output_dir / "variance_explained.csv", index=False)
    if factor_corr is not None:
        factor_corr.to_csv(output_dir / "factor_correlations.csv")

    # Text summary
    with open(output_dir / "fa_summary.txt", "w") as f:
        f.write(f"FA summary: {input_path.name}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"N segments: {X.shape[0]}\n")
        f.write(f"N features: {X.shape[1]}\n")
        f.write(f"Rotation: {rotation}\n")
        f.write(f"Method: principal axis factoring\n\n")
        f.write(f"k from parallel analysis: {k_parallel}\n")
        f.write(f"k from Kaiser:            {k_kaiser}\n")
        f.write(f"k used:                   {k}\n\n")
        f.write("Variance explained:\n")
        f.write(var_df.to_string(index=False))
        f.write("\n\n")
        if factor_corr is not None:
            f.write("Factor correlations:\n")
            f.write(factor_corr.round(3).to_string())
            f.write("\n")

    log.info(f"  Wrote results to {output_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered-dir", default="features_filtered")
    ap.add_argument("--output-dir", default="fa_results")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Number of factors. If omitted, picked by parallel analysis.",
    )
    ap.add_argument("--pa-iter", type=int, default=100)
    ap.add_argument("--pa-percentile", type=float, default=95.0)
    ap.add_argument(
        "--rotation",
        default="promax",
        choices=["promax", "varimax", "oblimin", "quartimin", None],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    filtered_dir = Path(args.filtered_dir)
    if args.files:
        files = [filtered_dir / f for f in args.files]
    else:
        files = sorted(filtered_dir.glob("*_filtered.parquet"))

    if not files:
        log.error(f"No *_filtered.parquet files in {filtered_dir}")
        return

    for f in files:
        stem = f.stem.replace("_filtered", "")
        out_dir = Path(args.output_dir) / stem
        process_file(
            f,
            out_dir,
            n_factors=args.n_factors,
            pa_iter=args.pa_iter,
            pa_percentile=args.pa_percentile,
            rotation=args.rotation,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
