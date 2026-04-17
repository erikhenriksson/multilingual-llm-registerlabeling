#!/usr/bin/env python3
"""
Plot MDA factor scores in 2D: register centroids with 95% CI error bars.

Clean, presentation-ready plots. One point per register, crosshair error
bars showing 95% CI of the mean on each axis. No scatter, no rings, no
clutter.

Input:  fa_results/{lang}_{script}/scores.parquet
Output: plots/{lang}_{script}/F{x}_vs_F{y}.png
        plots/combined__{lang1}__{lang2}/F{x}_vs_F{y}.png   (with --combine)
"""

import argparse
import logging
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
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

# Markers used to distinguish languages in combined plots.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]


def display_name(factor):
    """Convert 'F1' to 'Dimension 1' for plot labels."""
    if factor.startswith("F") and factor[1:].isdigit():
        return f"Dimension {factor[1:]}"
    return factor


def bootstrap_ci(vals, n_boot=2000, ci=95, seed=0):
    """Bootstrap 95% CI of the mean."""
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    )
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi


def compute_point(sub, fx, fy, use_bootstrap, n_boot):
    """Compute mean and 95% CI endpoints on each axis for a group of rows."""
    n = len(sub)
    mx = sub[fx].mean()
    my = sub[fy].mean()
    if use_bootstrap:
        x_lo, x_hi = bootstrap_ci(sub[fx].values, n_boot=n_boot)
        y_lo, y_hi = bootstrap_ci(sub[fy].values, n_boot=n_boot)
    else:
        x_se = sub[fx].std() / np.sqrt(n)
        y_se = sub[fy].std() / np.sqrt(n)
        x_lo, x_hi = mx - 1.96 * x_se, mx + 1.96 * x_se
        y_lo, y_hi = my - 1.96 * y_se, my + 1.96 * y_se
    return dict(n=n, mx=mx, my=my, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi)


def _draw_point(ax, d, color, marker, point_size=120):
    """Draw one errorbar crosshair + scatter marker at (mx, my)."""
    ax.errorbar(
        d["mx"],
        d["my"],
        xerr=[[d["mx"] - d["x_lo"]], [d["x_hi"] - d["mx"]]],
        yerr=[[d["my"] - d["y_lo"]], [d["y_hi"] - d["my"]]],
        fmt="none",
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=5,
    )
    ax.scatter(
        d["mx"],
        d["my"],
        c=color,
        s=point_size,
        marker=marker,
        edgecolors="white",
        linewidths=1.0,
        zorder=10,
    )


def _finalize_axes(ax, fx, fy, title, plot_data):
    """Zero lines, grid, labels, limits, spines."""
    ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="-", zorder=1)
    ax.axvline(0, color="#cccccc", linewidth=0.8, linestyle="-", zorder=1)
    ax.grid(True, alpha=0.15, linestyle="-")
    ax.set_xlabel(display_name(fx), fontsize=16, fontweight="bold")
    ax.set_ylabel(display_name(fy), fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=18, pad=14)
    ax.tick_params(labelsize=12)

    all_x = [d["mx"] for d in plot_data]
    all_y = [d["my"] for d in plot_data]
    x_range = (max(all_x) - min(all_x)) or 1
    y_range = (max(all_y) - min(all_y)) or 1
    pad_x = x_range * 0.45
    pad_y = y_range * 0.45
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y, max(all_y) + pad_y)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _place_labels(ax, plot_data, colors_by_label):
    """Place text labels near each point, avoiding overlap if adjustText is available."""
    try:
        from adjustText import adjust_text

        texts = []
        x_points = [d["mx"] for d in plot_data]
        y_points = [d["my"] for d in plot_data]
        for d in plot_data:
            t = ax.text(
                d["mx"],
                d["my"],
                f"  {d['label']}",
                fontsize=12,
                fontweight="bold",
                color=colors_by_label[d["label"]],
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
        sorted_data = sorted(plot_data, key=lambda d: d["my"], reverse=True)
        for i, d in enumerate(sorted_data):
            above = i % 2 == 0
            oy = 18 if above else -18
            ax.annotate(
                d["label"],
                (d["mx"], d["my"]),
                textcoords="offset points",
                xytext=(0, oy),
                fontsize=12,
                fontweight="bold",
                color=colors_by_label[d["label"]],
                va="bottom" if above else "top",
                ha="center",
                zorder=12,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.8),
            )


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
    """Single-language plot: one point per register."""
    labels = sorted(scores["label"].unique())
    colors = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    plot_data = []
    for lbl in labels:
        sub = scores[scores["label"] == lbl]
        d = compute_point(sub, fx, fy, use_bootstrap, n_boot)
        d["label"] = lbl
        plot_data.append(d)

    for d in plot_data:
        _draw_point(ax, d, colors[d["label"]], marker="o")

    _place_labels(ax, plot_data, colors)
    _finalize_axes(
        ax,
        fx,
        fy,
        f"{title_prefix}: {display_name(fx)} vs {display_name(fy)}",
        plot_data,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  {fx} vs {fy} → {output_path}")


def plot_factor_pair_combined(
    per_lang_scores,
    fx,
    fy,
    output_path,
    title_prefix,
    figsize,
    dpi,
    n_boot,
    use_bootstrap,
):
    """
    Combined plot across several languages.

    per_lang_scores: dict[lang_name -> DataFrame with factor columns and 'label']

    Color = register (shared across languages). Marker = language.
    Points with fewer than 2 rows in a given language are skipped (SE undefined).
    """
    langs = list(per_lang_scores.keys())

    # Union of registers across the selected languages. Colors shared.
    all_labels = sorted(
        {lbl for df in per_lang_scores.values() for lbl in df["label"].unique()}
    )
    colors = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(all_labels)}
    markers = {lang: MARKERS[i % len(MARKERS)] for i, lang in enumerate(langs)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    plot_data = []  # for axis limits + label placement
    # We want one text label per (register, language) pair only if the register
    # appears in multiple languages and they are far apart; simplest approach:
    # label every point, let adjustText sort it out.
    for lang, df in per_lang_scores.items():
        if fx not in df.columns or fy not in df.columns:
            log.warning(f"  {lang} missing {fx} or {fy}; skipping in combined plot")
            continue
        for lbl in sorted(df["label"].unique()):
            sub = df[df["label"] == lbl]
            if len(sub) < 2:
                log.warning(f"  {lang}/{lbl}: n={len(sub)}, skipping (CI undefined)")
                continue
            d = compute_point(sub, fx, fy, use_bootstrap, n_boot)
            d["label"] = lbl
            d["lang"] = lang
            plot_data.append(d)

    if not plot_data:
        log.warning(f"  No plottable points for {fx} vs {fy} in combined plot")
        plt.close(fig)
        return

    for d in plot_data:
        _draw_point(ax, d, colors[d["label"]], marker=markers[d["lang"]])

    _finalize_axes(
        ax,
        fx,
        fy,
        f"{title_prefix}: {display_name(fx)} vs {display_name(fy)}",
        plot_data,
    )

    # Two legends: languages (by marker) and registers (by color).
    lang_handles = [
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker=markers[lang],
            linestyle="None",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=lang,
        )
        for lang in langs
    ]
    leg_lang = ax.legend(
        handles=lang_handles,
        title="Language",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
    )
    ax.add_artist(leg_lang)

    # Only show register legend entries that actually appear in the plot
    present_labels = sorted({d["label"] for d in plot_data})
    reg_handles = [
        mlines.Line2D(
            [],
            [],
            color=colors[lbl],
            marker="o",
            linestyle="None",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=lbl,
        )
        for lbl in present_labels
    ]
    ax.legend(
        handles=reg_handles,
        title="Register",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.75),
        frameon=False,
        fontsize=10,
        title_fontsize=11,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  [combined] {fx} vs {fy} → {output_path}")


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


def process_combined(fa_root, combine_langs, output_root, pairs, **kwargs):
    """Produce one combined plot per factor pair across the given languages."""
    per_lang = {}
    for lang in combine_langs:
        p = fa_root / lang / "scores.parquet"
        if not p.exists():
            log.warning(f"  [combined] no scores.parquet for {lang}; skipping it")
            continue
        per_lang[lang] = pd.read_parquet(p)

    if len(per_lang) < 2:
        log.warning(
            "  [combined] need at least 2 languages with scores; got "
            f"{len(per_lang)}. Skipping combined plots."
        )
        return

    combo_name = "combined__" + "__".join(per_lang.keys())
    out_dir = output_root / combo_name
    log.info(f"Plotting combined: {list(per_lang.keys())}")
    log.warning(
        "  Note: factor scores from separate per-language FA runs are not "
        "guaranteed to be on a common scale or sign. Interpret cross-language "
        "positions with care."
    )

    # Pairs were already normalized to 'F<n>' form upstream; verify presence.
    for fx, fy in pairs:
        any_has = any(fx in df.columns and fy in df.columns for df in per_lang.values())
        if not any_has:
            log.warning(f"  [combined] skipping {fx} vs {fy}: not present in any lang")
            continue
        out_path = out_dir / f"{fx}_vs_{fy}.png"
        plot_factor_pair_combined(
            per_lang, fx, fy, out_path, title_prefix=combo_name, **kwargs
        )


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
        "--combine",
        "--cl",
        nargs="*",
        default=None,
        dest="combine",
        help="Language subdirs to combine into a single plot per factor pair, "
        "IN ADDITION to per-language plots. Registers are colored the "
        "same across languages; languages are distinguished by marker. "
        "Example: --combine en_Latn es_Latn fr_Latn",
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
    output_root = Path(args.output_dir)

    if args.langs:
        subdirs = [fa_root / lang for lang in args.langs]
    else:
        subdirs = [d for d in sorted(fa_root.iterdir()) if d.is_dir()]

    # --- Per-language plots (unchanged behavior) -------------------------
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
        out_dir = output_root / sub.name
        process_lang(
            scores_path,
            out_dir,
            pairs=pairs,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            n_boot=args.n_boot,
            use_bootstrap=args.bootstrap,
        )

    # --- Combined plot (new) --------------------------------------------
    if args.combine:
        # Determine pairs for combined plots. 'all' = intersection of factor
        # columns across requested languages, to avoid silently dropping pairs.
        if "all" in args.pairs:
            factor_sets = []
            for lang in args.combine:
                p = fa_root / lang / "scores.parquet"
                if p.exists():
                    cols = pd.read_parquet(p).columns
                    factor_sets.append(set(c for c in cols if c.startswith("F")))
            shared = sorted(set.intersection(*factor_sets)) if factor_sets else []
            combined_pairs = list(combinations(shared, 2))
        else:
            combined_pairs = parse_pairs(args.pairs)

        process_combined(
            fa_root,
            args.combine,
            output_root,
            pairs=combined_pairs,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            n_boot=args.n_boot,
            use_bootstrap=args.bootstrap,
        )


if __name__ == "__main__":
    main()
