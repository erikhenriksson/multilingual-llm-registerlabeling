#!/usr/bin/env python3
"""
Plot MDA factor scores in 2D, colored by register label.

Input:  fa_results/{lang}_{script}/scores.parquet
Output: plots/{lang}_{script}/F{x}_vs_F{y}.png  (one per factor pair)

By default plots F1 vs F2. Use --pairs to specify other combinations.
"""

import argparse
import logging
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = ["doc_id", "source", "segment_idx", "label", "n_tokens", "n_sents"]

# Qualitative palette that works for up to ~20 labels
PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
    "#000000",
]


def plot_factor_pair(
    scores: pd.DataFrame,
    fx: str,
    fy: str,
    output_path: Path,
    title_prefix: str,
    point_size: float,
    alpha: float,
    show_means: bool,
    show_ellipses: bool,
    figsize: tuple,
    dpi: int,
):
    labels = sorted(scores["label"].unique())
    n_labels = len(labels)
    colors = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for lbl in labels:
        sub = scores[scores["label"] == lbl]
        ax.scatter(
            sub[fx],
            sub[fy],
            c=colors[lbl],
            s=point_size,
            alpha=alpha,
            label=lbl,
            edgecolors="none",
            rasterized=True,  # keeps file size reasonable at high N
        )

    if show_means:
        means = scores.groupby("label")[[fx, fy]].mean()
        for lbl in labels:
            mx, my = means.loc[lbl, fx], means.loc[lbl, fy]
            ax.scatter(
                mx,
                my,
                c=colors[lbl],
                s=200,
                marker="X",
                edgecolors="black",
                linewidths=1.0,
                zorder=10,
            )
            ax.annotate(
                lbl,
                (mx, my),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                fontweight="bold",
                color=colors[lbl],
                zorder=11,
            )

    if show_ellipses:
        from matplotlib.patches import Ellipse

        for lbl in labels:
            sub = scores[scores["label"] == lbl]
            if len(sub) < 3:
                continue
            mx, my = sub[fx].mean(), sub[fy].mean()
            cov = np.cov(sub[fx].values, sub[fy].values)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort descending
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            # 95% confidence ellipse (chi2 with 2 df, p=0.05 → 5.991)
            width = 2 * np.sqrt(5.991 * eigenvalues[0])
            height = 2 * np.sqrt(5.991 * eigenvalues[1])
            ell = Ellipse(
                xy=(mx, my),
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor=colors[lbl],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )
            ax.add_patch(ell)

    ax.set_xlabel(fx, fontsize=12)
    ax.set_ylabel(fy, fontsize=12)
    ax.set_title(f"{title_prefix}: {fx} vs {fy}", fontsize=13)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")

    # Legend: outside plot if many labels, inside if few
    if n_labels > 8:
        ax.legend(
            fontsize=7,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            markerscale=3,
            frameon=False,
        )
        fig.subplots_adjust(right=0.78)
    else:
        ax.legend(fontsize=8, markerscale=3, frameon=True, fancybox=True)

    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  {fx} vs {fy} → {output_path}")


def process_lang(
    scores_path: Path,
    output_dir: Path,
    pairs: list[tuple[str, str]],
    point_size: float,
    alpha: float,
    show_means: bool,
    show_ellipses: bool,
    figsize: tuple,
    dpi: int,
):
    stem = scores_path.parent.name
    log.info(f"Plotting: {stem}")
    scores = pd.read_parquet(scores_path)
    factor_cols = sorted([c for c in scores.columns if c.startswith("F")])

    log.info(
        f"  {len(scores)} segments, {len(factor_cols)} factors, "
        f"{scores['label'].nunique()} labels"
    )

    for fx, fy in pairs:
        if fx not in factor_cols or fy not in factor_cols:
            log.warning(f"  Skipping {fx} vs {fy}: factor not found")
            continue
        out_path = output_dir / f"{fx}_vs_{fy}.png"
        plot_factor_pair(
            scores,
            fx,
            fy,
            out_path,
            title_prefix=stem,
            point_size=point_size,
            alpha=alpha,
            show_means=show_means,
            show_ellipses=show_ellipses,
            figsize=figsize,
            dpi=dpi,
        )


def parse_pairs(pair_strs: list[str]) -> list[tuple[str, str]]:
    """Parse '1,2' or 'F1,F2' into ('F1', 'F2')."""
    out = []
    for s in pair_strs:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"Bad pair: {s!r}. Expected 'X,Y' e.g. '1,2' or 'F1,F2'")
        a, b = parts
        if not a.startswith("F"):
            a = f"F{a}"
        if not b.startswith("F"):
            b = f"F{b}"
        out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa-dir", default="fa_results")
    ap.add_argument("--output-dir", default="plots")
    ap.add_argument("--langs", nargs="*", default=None)
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=["1,2"],
        help="Factor pairs to plot (e.g. '1,2' '1,3' '2,3'). "
        "Use 'all' for all pairwise combinations.",
    )
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Point transparency (lower for dense data).",
    )
    ap.add_argument(
        "--no-means", action="store_true", help="Don't plot register centroids."
    )
    ap.add_argument(
        "--ellipses",
        action="store_true",
        help="Draw 95%% confidence ellipses per label.",
    )
    ap.add_argument("--figsize", type=float, nargs=2, default=[10, 8])
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    fa_root = Path(args.fa_dir)
    if args.langs:
        subdirs = [fa_root / lang for lang in args.langs]
    else:
        subdirs = [d for d in sorted(fa_root.iterdir()) if d.is_dir()]

    for sub in subdirs:
        scores_path = sub / "scores.parquet"
        if not scores_path.exists():
            log.warning(f"  Skipping {sub.name}: no scores.parquet")
            continue

        # Determine factor pairs
        scores_peek = pd.read_parquet(scores_path, columns=["doc_id"])
        # Re-read to get column names (cheap)
        all_cols = pd.read_parquet(scores_path).columns
        factor_cols = sorted([c for c in all_cols if c.startswith("F")])

        if "all" in args.pairs:
            pairs = list(combinations(factor_cols, 2))
        else:
            pairs = parse_pairs(args.pairs)

        out_dir = Path(args.output_dir) / sub.name
        process_lang(
            scores_path,
            out_dir,
            pairs=pairs,
            point_size=args.point_size,
            alpha=args.alpha,
            show_means=not args.no_means,
            show_ellipses=args.ellipses,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
