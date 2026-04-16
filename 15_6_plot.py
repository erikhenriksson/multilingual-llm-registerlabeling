#!/usr/bin/env python3
"""
Plot MDA factor scores in 2D: register centroids with 95% CI error bars.

Clean, presentation-ready plots. One point per register, crosshair error
bars showing 95% CI of the mean on each axis. No scatter, no rings, no
clutter.

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
    "#9A6324",
    "#469990",
    "#800000",
    "#808000",
    "#000075",
    "#a9a9a9",
    "#000000",
    "#dcbeff",
]


def bootstrap_ci(vals, n_boot=2000, ci=95, seed=0):
    """Bootstrap 95% CI of the mean."""
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    )
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi


def plot_factor_pair(
    scores,
    fx,
    fy,
    output_path,
    title_prefix,
    figsize,
    dpi,
    n_boot,
    use_bootstrap,
):
    labels = sorted(scores["label"].unique())
    colors = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Compute means and CIs
    plot_data = []
    for lbl in labels:
        sub = scores[scores["label"] == lbl]
        n = len(sub)
        mx = sub[fx].mean()
        my = sub[fy].mean()

        if use_bootstrap:
            x_lo, x_hi = bootstrap_ci(sub[fx].values, n_boot=n_boot)
            y_lo, y_hi = bootstrap_ci(sub[fy].values, n_boot=n_boot)
        else:
            # Parametric: mean ± 1.96 * SE
            x_se = sub[fx].std() / np.sqrt(n)
            y_se = sub[fy].std() / np.sqrt(n)
            x_lo, x_hi = mx - 1.96 * x_se, mx + 1.96 * x_se
            y_lo, y_hi = my - 1.96 * y_se, my + 1.96 * y_se

        plot_data.append(
            {
                "label": lbl,
                "n": n,
                "mx": mx,
                "my": my,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
            }
        )

    # Plot error bars + points
    for d in plot_data:
        ax.errorbar(
            d["mx"],
            d["my"],
            xerr=[[d["mx"] - d["x_lo"]], [d["x_hi"] - d["mx"]]],
            yerr=[[d["my"] - d["y_lo"]], [d["y_hi"] - d["my"]]],
            fmt="none",
            ecolor=colors[d["label"]],
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            zorder=5,
        )
        ax.scatter(
            d["mx"],
            d["my"],
            c=colors[d["label"]],
            s=120,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            zorder=10,
        )

    # Label placement using adjustText if available, else manual offsets
    try:
        from adjustText import adjust_text

        texts = []
        # Collect point positions so adjustText avoids them
        x_points = [d["mx"] for d in plot_data]
        y_points = [d["my"] for d in plot_data]
        for d in plot_data:
            t = ax.text(
                d["mx"],
                d["my"],
                f"  {d['label']}",
                fontsize=12,
                fontweight="bold",
                color=colors[d["label"]],
                va="center",
                zorder=12,
            )
            texts.append(t)
        adjust_text(
            texts,
            x=x_points,
            y=y_points,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.8),
            expand=(2.0, 2.0),
            force_text=(1.5, 1.5),
            force_points=(2.0, 2.0),
        )
    except ImportError:
        # Fallback: place labels above or below the point, offset enough
        # to clear the error bars
        sorted_data = sorted(plot_data, key=lambda d: d["my"], reverse=True)
        for i, d in enumerate(sorted_data):
            # Alternate above/below, offset vertically to avoid error bars
            above = i % 2 == 0
            oy = 18 if above else -18
            ax.annotate(
                d["label"],
                (d["mx"], d["my"]),
                textcoords="offset points",
                xytext=(0, oy),
                fontsize=12,
                fontweight="bold",
                color=colors[d["label"]],
                va="bottom" if above else "top",
                ha="center",
                zorder=12,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.8),
            )

    # Axis lines at zero
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="-", zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.8, linestyle="-", zorder=1)

    # Light grid
    ax.grid(True, alpha=0.15, linestyle="-")

    ax.set_xlabel(fx, fontsize=16, fontweight="bold")
    ax.set_ylabel(fy, fontsize=16, fontweight="bold")
    ax.set_title(f"{title_prefix}: {fx} vs {fy}", fontsize=18, pad=14)
    ax.tick_params(labelsize=12)

    # Pad axes slightly beyond data range
    all_x = [d["mx"] for d in plot_data]
    all_y = [d["my"] for d in plot_data]
    x_range = max(all_x) - min(all_x) or 1
    y_range = max(all_y) - min(all_y) or 1
    pad_x = x_range * 0.45
    pad_y = y_range * 0.45
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
        "--bootstrap",
        action="store_true",
        help="Use bootstrap CIs (slower, better for clustered data). "
        "Default is parametric SE.",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--figsize", type=float, nargs=2, default=[10, 7])
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
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            n_boot=args.n_boot,
            use_bootstrap=args.bootstrap,
        )


if __name__ == "__main__":
    main()
