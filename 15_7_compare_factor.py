#!/usr/bin/env python3
"""
Compare a single MDA dimension across languages.

Horizontal dot plot: each row is a register, each dot is a language.
Shows at a glance whether the register ordering on a given dimension
is consistent across languages.

Input:  fa_results/{lang}/scores.parquet  (one per language)
Output: plots/dimension_comparison/F{n}_comparison.png
"""

import argparse
import logging
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

# ── Visual encoding ──────────────────────────────────────────────────────────
LANG_STYLES = {
    "eng_Latn": {"color": "#4363d8", "name": "English"},
    "cmn": {"color": "#e6194b", "name": "Mandarin"},
    "fin_Latn": {"color": "#3cb44b", "name": "Finnish"},
    "swe_Latn": {"color": "#f58231", "name": "Swedish"},
    "pes_Arab": {"color": "#911eb4", "name": "Persian"},
}
FALLBACK_STYLES = [
    {"color": "#469990", "name": None},
    {"color": "#800000", "name": None},
    {"color": "#808000", "name": None},
]


def get_lang_style(lang, _idx=[0]):
    if lang in LANG_STYLES:
        return LANG_STYLES[lang]
    style = FALLBACK_STYLES[_idx[0] % len(FALLBACK_STYLES)].copy()
    if style["name"] is None:
        style["name"] = lang
    _idx[0] += 1
    return style


def plot_dimension(
    all_data,
    factor,
    output_path,
    figsize,
    dpi,
    sort_by,
):
    """
    all_data: list of (lang_key, scores_df) tuples
    """
    # Compute mean factor score per (language, register)
    # Z-score normalize within each language first, so all languages
    # are on the same scale (mean=0, sd=1 within each language).
    # This makes the plot show relative register positions, not absolute
    # factor scores which are incommensurable across languages.
    records = []
    for lang, scores in all_data:
        if factor not in scores.columns:
            log.warning(f"  {lang}: {factor} not found, skipping")
            continue
        # Normalize within this language
        lang_mean = scores[factor].mean()
        lang_sd = scores[factor].std()
        if lang_sd == 0:
            log.warning(f"  {lang}: zero variance on {factor}, skipping")
            continue
        scores = scores.copy()
        scores[factor] = (scores[factor] - lang_mean) / lang_sd

        for lbl, grp in scores.groupby("label"):
            n = len(grp)
            mean = grp[factor].mean()
            se = grp[factor].std() / np.sqrt(n)
            records.append(
                {
                    "lang": lang,
                    "label": lbl,
                    "mean": mean,
                    "ci_lo": mean - 1.96 * se,
                    "ci_hi": mean + 1.96 * se,
                    "n": n,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        log.error("No data")
        return

    # Sort registers by the chosen language's mean (or overall mean)
    if sort_by and sort_by in df["lang"].values:
        sort_vals = df[df["lang"] == sort_by].set_index("label")["mean"]
    else:
        sort_vals = df.groupby("label")["mean"].mean()
    register_order = sort_vals.sort_values().index.tolist()

    # Map registers to y positions
    n_regs = len(register_order)
    reg_to_y = {reg: i for i, reg in enumerate(register_order)}

    # Vertical jitter: spread languages within each register row
    langs = sorted(df["lang"].unique())
    n_langs = len(langs)
    jitter_width = 0.35
    if n_langs > 1:
        jitter_positions = np.linspace(-jitter_width, jitter_width, n_langs)
    else:
        jitter_positions = [0.0]
    lang_to_jitter = {lang: jitter_positions[i] for i, lang in enumerate(langs)}

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Light horizontal bands for readability
    for i in range(n_regs):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color="#f5f5f5", zorder=0)

    # Zero line
    ax.axvline(0, color="#cccccc", linewidth=1.0, zorder=1)

    # Plot each language's points
    for lang in langs:
        style = get_lang_style(lang)
        sub = df[df["lang"] == lang]
        jit = lang_to_jitter[lang]

        for _, row in sub.iterrows():
            if row["label"] not in reg_to_y:
                continue
            y = reg_to_y[row["label"]] + jit

            # Error bar
            ax.errorbar(
                row["mean"],
                y,
                xerr=[[row["mean"] - row["ci_lo"]], [row["ci_hi"] - row["mean"]]],
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.5,
                capsize=4,
                capthick=1.5,
                alpha=0.7,
                zorder=5,
            )

        # Plot all points for this language at once (for legend)
        ys = [
            reg_to_y[r["label"]] + jit
            for _, r in sub.iterrows()
            if r["label"] in reg_to_y
        ]
        xs = [r["mean"] for _, r in sub.iterrows() if r["label"] in reg_to_y]
        ax.scatter(
            xs,
            ys,
            marker="o",
            c=style["color"],
            s=100,
            edgecolors="white",
            linewidths=0.8,
            label=style["name"],
            zorder=10,
        )

    # Y axis: register names
    ax.set_yticks(range(n_regs))
    ax.set_yticklabels(register_order, fontsize=13)
    ax.set_ylim(-0.6, n_regs - 0.4)

    # X axis
    ax.set_xlabel(
        f"{factor} score (z-normalized within language)", fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=12)

    ax.set_title(f"Register means on {factor} across languages", fontsize=18, pad=14)

    ax.legend(
        title="Language",
        title_fontsize=12,
        fontsize=11,
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle="-")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  {factor} → {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa-dir", default="fa_results")
    ap.add_argument("--output-dir", default="plots/dimension_comparison")
    ap.add_argument(
        "--langs",
        nargs="*",
        default=None,
        help="Language subdirs to include (default: all)",
    )
    ap.add_argument(
        "--factors",
        nargs="*",
        default=["F1"],
        help="Which factors to plot (e.g. F1 F2). Default: F1 only.",
    )
    ap.add_argument(
        "--sort-by",
        default=None,
        help="Sort registers by this language's means (e.g. eng_Latn). "
        "Default: sort by cross-language average.",
    )
    ap.add_argument("--figsize", type=float, nargs=2, default=[14, 8])
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    fa_root = Path(args.fa_dir)
    if args.langs:
        subdirs = [fa_root / lang for lang in args.langs]
    else:
        subdirs = [d for d in sorted(fa_root.iterdir()) if d.is_dir()]

    all_data = []
    for sub in subdirs:
        scores_path = sub / "scores.parquet"
        if not scores_path.exists():
            log.warning(f"  Skipping {sub.name}: no scores.parquet")
            continue
        scores = pd.read_parquet(scores_path)
        all_data.append((sub.name, scores))
        log.info(f"  Loaded {sub.name}: {len(scores)} segments")

    if not all_data:
        log.error("No data loaded")
        return

    output_dir = Path(args.output_dir)
    for factor in args.factors:
        if not factor.startswith("F"):
            factor = f"F{factor}"
        out_path = output_dir / f"{factor}_comparison.png"
        plot_dimension(
            all_data,
            factor,
            out_path,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            sort_by=args.sort_by,
        )


if __name__ == "__main__":
    main()
