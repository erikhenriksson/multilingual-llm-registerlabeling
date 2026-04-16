#!/usr/bin/env python3
"""
Plot MDA factor scores in 2D, colored by register label.

Three plot modes (--mode):
  means    : centroids + 95% CI ellipses of the mean (default; best for large N)
  density  : 2D KDE contours per register + centroids
  scatter  : raw scatter + centroids (only useful for small N)

The key insight for large N: you care about where each register's *mean*
sits and whether means are significantly separated — not where individual
segments fall. The mean-CI ellipse shrinks with sqrt(N), so at 60k segments
the ellipses are tiny and show real separation clearly.

Input:  fa_results/{lang}_{script}/scores.parquet
Output: plots/{lang}_{script}/F{x}_vs_F{y}.png
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
from matplotlib.patches import Ellipse
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

META_COLS = ["doc_id", "source", "segment_idx", "label", "n_tokens", "n_sents"]

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


def make_ellipse(mean_x, mean_y, cov, scale, **kwargs):
    """Create a matplotlib Ellipse from a 2x2 covariance matrix.

    scale: chi2 critical value (5.991 for 95% population, or
           5.991/N for 95% CI of the mean).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    # Clamp negative eigenvalues from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0)
    width = 2 * np.sqrt(scale * eigenvalues[0])
    height = 2 * np.sqrt(scale * eigenvalues[1])
    return Ellipse(
        xy=(mean_x, mean_y), width=width, height=height, angle=angle, **kwargs
    )


def plot_factor_pair(
    scores,
    fx,
    fy,
    output_path,
    title_prefix,
    mode,
    point_size,
    alpha,
    figsize,
    dpi,
    contour_levels,
    sd_rings,
):
    labels = sorted(scores["label"].unique())
    colors = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ── Scatter or density background ─────────────────────────────────────
    if mode == "scatter":
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
                rasterized=True,
            )
    elif mode == "density":
        for lbl in labels:
            sub = scores[scores["label"] == lbl]
            if len(sub) < 20:
                continue
            try:
                xy = np.vstack([sub[fx].values, sub[fy].values])
                kde = stats.gaussian_kde(xy)
                # Grid
                xmin, xmax = sub[fx].quantile(0.01), sub[fx].quantile(0.99)
                ymin, ymax = sub[fy].quantile(0.01), sub[fy].quantile(0.99)
                pad = 0.3
                xx, yy = np.mgrid[
                    xmin - pad : xmax + pad : 100j,
                    ymin - pad : ymax + pad : 100j,
                ]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(
                    xx,
                    yy,
                    zz,
                    levels=contour_levels,
                    colors=[colors[lbl]],
                    linewidths=1.2,
                    alpha=0.7,
                )
                # Filled lightest contour for background
                ax.contourf(
                    xx,
                    yy,
                    zz,
                    levels=[zz.max() * 0.1, zz.max()],
                    colors=[colors[lbl]],
                    alpha=0.06,
                )
            except Exception as e:
                log.warning(f"  KDE failed for {lbl}: {e}")

    # ── Centroids + mean-CI ellipses (always shown) ──────────────────────
    chi2_95 = 5.991  # chi2 with 2 df, p=0.05

    for lbl in labels:
        sub = scores[scores["label"] == lbl]
        n = len(sub)
        if n < 3:
            continue
        mx, my = sub[fx].mean(), sub[fy].mean()
        cov = np.cov(sub[fx].values, sub[fy].values)

        # 95% CI of the mean: covariance / N
        cov_mean = cov / n
        ell_mean = make_ellipse(
            mx,
            my,
            cov_mean,
            chi2_95,
            facecolor=colors[lbl],
            edgecolor=colors[lbl],
            alpha=0.35,
            linewidth=2,
            linestyle="-",
        )
        ax.add_patch(ell_mean)

        # Optional: SD rings (population spread, at 1 or 2 SD)
        for sd in sd_rings:
            ell_pop = make_ellipse(
                mx,
                my,
                cov,
                sd**2,
                facecolor="none",
                edgecolor=colors[lbl],
                linewidth=0.8,
                linestyle=":",
                alpha=0.4,
            )
            ax.add_patch(ell_pop)

        # Centroid marker
        ax.scatter(
            mx,
            my,
            c=colors[lbl],
            s=180,
            marker="X",
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )

    # ── Labels: use adjustText if available, else basic offset ───────────
    means = scores.groupby("label")[[fx, fy]].mean()
    texts = []
    try:
        from adjustText import adjust_text

        for lbl in labels:
            mx, my = means.loc[lbl, fx], means.loc[lbl, fy]
            t = ax.text(
                mx,
                my,
                lbl,
                fontsize=9,
                fontweight="bold",
                color=colors[lbl],
                zorder=12,
            )
            texts.append(t)
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    except ImportError:
        # Fallback: stagger offsets to reduce overlap
        offsets = [
            (10, 10),
            (-10, 10),
            (10, -10),
            (-10, -10),
            (15, 0),
            (-15, 0),
            (0, 15),
            (0, -15),
        ]
        for i, lbl in enumerate(labels):
            mx, my = means.loc[lbl, fx], means.loc[lbl, fy]
            ox, oy = offsets[i % len(offsets)]
            ax.annotate(
                lbl,
                (mx, my),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=9,
                fontweight="bold",
                color=colors[lbl],
                zorder=12,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
            )

    ax.set_xlabel(fx, fontsize=12)
    ax.set_ylabel(fy, fontsize=12)
    ax.set_title(f"{title_prefix}: {fx} vs {fy}", fontsize=13)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle=":")

    # Legend (for scatter/density modes; means mode uses inline labels)
    if mode in ("scatter", "density"):
        # For density mode, scatter didn't add labels, so add proxy patches
        if mode == "density":
            from matplotlib.patches import Patch

            handles = [
                Patch(facecolor=colors[lbl], alpha=0.5, label=lbl) for lbl in labels
            ]
        else:
            handles = None  # scatter already set labels
        n_labels = len(labels)
        if n_labels > 8:
            ax.legend(
                handles=handles,
                fontsize=7,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                markerscale=3,
                frameon=False,
            )
            fig.subplots_adjust(right=0.78)
        else:
            ax.legend(
                handles=handles, fontsize=8, markerscale=3, frameon=True, fancybox=True
            )

    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  {fx} vs {fy} → {output_path}")


def process_lang(scores_path, output_dir, pairs, **kwargs):
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
        plot_factor_pair(scores, fx, fy, out_path, title_prefix=stem, **kwargs)


def parse_pairs(pair_strs):
    out = []
    for s in pair_strs:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"Bad pair: {s!r}")
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
        help="Factor pairs (e.g. '1,2' '1,3'). Use 'all' for all combos.",
    )
    ap.add_argument(
        "--mode",
        default="means",
        choices=["means", "density", "scatter"],
        help="Plot mode. 'means' shows centroids + CI ellipses only (best "
        "for large N). 'density' adds KDE contours. 'scatter' shows "
        "all points (only for small N).",
    )
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument(
        "--sd-rings",
        type=float,
        nargs="*",
        default=[1.0],
        help="Draw population SD rings at these SD values (e.g. 1 2). "
        "Pass --sd-rings with no args to disable.",
    )
    ap.add_argument(
        "--contour-levels",
        type=int,
        default=4,
        help="Number of KDE contour levels (density mode).",
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
            mode=args.mode,
            point_size=args.point_size,
            alpha=args.alpha,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            contour_levels=args.contour_levels,
            sd_rings=args.sd_rings if args.sd_rings else [],
        )


if __name__ == "__main__":
    main()
