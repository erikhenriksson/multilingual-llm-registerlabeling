"""
Analyze LLM annotation evaluation files.
Computes per-language and averaged classification metrics + confusion matrices.
Also derives CORE register labels (Biber & Egbert) from facet labels and evaluates those.
Usage: python analyze_evals.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

EVAL_DIR = "evaluation"
OUT_DIR = "evaluation_results"
os.makedirs(OUT_DIR, exist_ok=True)

FACETS = ["mode_medium", "mode_turn", "field_activity", "tenor_formality"]

# Mapping from field_activity to CORE register (for monologic written texts)
ACTIVITY_TO_REGISTER = {
    "recounting": "Narrative",
    "explaining": "Info. description/explanation",
    "evaluating": "Opinion",
    "promoting": "Info. persuasion",
    "directing": "How-to/instruct.",
    "creating": "Lyrical",
}


def get_n_tasks(lang):
    """swe_Latn gets 100 tasks, everything else 250."""
    return 100 if lang == "swe_Latn" else 250


def load_eval_files():
    """Load all eval JSON files, return dict of lang -> data."""
    files = glob.glob(os.path.join(EVAL_DIR, "*_eval.json"))
    langs = {}
    for f in files:
        basename = os.path.basename(f)
        lang = basename.replace("_eval.json", "")
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        langs[lang] = data
    return langs


def extract_pairs(annotations, n_tasks, facet):
    """Extract (human_label, llm_label) pairs for a facet, skipping Nones."""
    pairs = []
    for ann in annotations[:n_tasks]:
        h = ann["human_labels"].get(facet)
        l = ann["llm_labels"].get(facet)
        if h is not None and l is not None:
            pairs.append((h, l))
    return pairs


def derive_core_label(labels):
    """
    Derive a CORE register label from a set of facet labels.
    Decision tree (Biber & Egbert):
      1. mode_medium == "cannot_rate" -> "Cannot rate"
      2. mode_medium == "transcribed" -> "Spoken"
      3. mode_turn == "dialogic" -> "Interactive Discussion"
      4. field_activity -> mapped via ACTIVITY_TO_REGISTER
    Returns None if insufficient info to derive.
    """
    medium = labels.get("mode_medium")
    turn = labels.get("mode_turn")
    activity = labels.get("field_activity")

    if medium == "cannot_rate":
        return "Cannot rate"
    if medium == "transcribed":
        return "Spoken"
    # medium == "written" from here
    if turn == "dialogic":
        return "Interactive Discussion"
    # monologic written -> use activity
    if activity in ACTIVITY_TO_REGISTER:
        return ACTIVITY_TO_REGISTER[activity]
    return None


def extract_core_pairs(annotations, n_tasks):
    """Extract (human_core, llm_core) pairs by deriving CORE labels from facets."""
    pairs = []
    for ann in annotations[:n_tasks]:
        h_core = derive_core_label(ann["human_labels"])
        l_core = derive_core_label(ann["llm_labels"])
        if h_core is not None and l_core is not None:
            pairs.append((h_core, l_core))
    return pairs


def compute_metrics(pairs):
    """Return classification report string + confusion matrix info."""
    if not pairs:
        return None, None, None, None
    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return acc, report, cm, labels


def format_cm(cm, labels):
    """Format confusion matrix as a readable string."""
    col_width = max(len(l) for l in labels) + 2
    col_width = max(col_width, 8)
    header = " " * col_width + "  " + "  ".join(f"{l:>{col_width}s}" for l in labels)
    lines = [header]
    for i, label in enumerate(labels):
        row = f"{label:>{col_width}s}  " + "  ".join(
            f"{cm[i][j]:>{col_width}d}" for j in range(len(labels))
        )
        lines.append(row)
    return "\n".join(lines)


def main():
    langs = load_eval_files()
    if not langs:
        print(
            f"No eval files found in {EVAL_DIR}/. Expected files like eng_Latn_eval.json"
        )
        return

    print(f"Found {len(langs)} language(s): {', '.join(sorted(langs.keys()))}")

    # Collect all pairs per facet across languages
    all_pairs_by_facet = defaultdict(list)
    all_core_pairs = []
    all_reports = []
    lang_overall_accs = []
    lang_core_accs = []

    for lang in sorted(langs.keys()):
        data = langs[lang]
        annotations = data["annotations"]
        n_tasks = get_n_tasks(lang)
        actual_n = min(n_tasks, len(annotations))

        report_lines = []
        report_lines.append(f"{'=' * 70}")
        report_lines.append(f"LANGUAGE: {lang}  (using first {actual_n} tasks)")
        report_lines.append(f"{'=' * 70}")

        # --- Per-facet metrics ---
        lang_total_correct = 0
        lang_total_n = 0

        for facet in FACETS:
            pairs = extract_pairs(annotations, n_tasks, facet)
            if not pairs:
                report_lines.append(f"\n--- {facet}: no valid pairs ---\n")
                continue

            acc, report, cm, labels = compute_metrics(pairs)
            all_pairs_by_facet[facet].extend(pairs)
            lang_total_correct += int(acc * len(pairs))
            lang_total_n += len(pairs)

            report_lines.append(
                f"\n--- {facet} (n={len(pairs)}, accuracy={acc:.3f}) ---"
            )
            report_lines.append(report)
            report_lines.append("Confusion Matrix (rows=human, cols=LLM):")
            report_lines.append(format_cm(cm, labels))
            report_lines.append("")

        # Overall accuracy across all facets
        if lang_total_n > 0:
            lang_overall_acc = lang_total_correct / lang_total_n
            lang_overall_accs.append(lang_overall_acc)
            report_lines.append(f"{'~' * 70}")
            report_lines.append(
                f"OVERALL ACCURACY (all facets pooled): {lang_overall_acc:.3f}  (n={lang_total_n})"
            )
            report_lines.append(f"{'~' * 70}")

        # --- Derived CORE register metrics ---
        core_pairs = extract_core_pairs(annotations, n_tasks)
        all_core_pairs.extend(core_pairs)

        if core_pairs:
            acc, report, cm, labels = compute_metrics(core_pairs)
            lang_core_accs.append(acc)
            report_lines.append("")
            report_lines.append(f"{'=' * 70}")
            report_lines.append(
                f"DERIVED CORE REGISTER (n={len(core_pairs)}, accuracy={acc:.3f})"
            )
            report_lines.append(f"{'=' * 70}")
            report_lines.append(report)
            report_lines.append("Confusion Matrix (rows=human, cols=LLM):")
            report_lines.append(format_cm(cm, labels))
            report_lines.append("")

        lang_report = "\n".join(report_lines)
        all_reports.append(lang_report)

        # Save per-language report
        with open(os.path.join(OUT_DIR, f"{lang}_report.txt"), "w") as f:
            f.write(lang_report)
        print(f"  Saved {lang}_report.txt")

    # ===== Pooled report =====
    avg_lines = []
    avg_lines.append(f"{'=' * 70}")
    avg_lines.append(f"POOLED ACROSS ALL LANGUAGES ({len(langs)} languages)")
    avg_lines.append(f"{'=' * 70}")

    if lang_overall_accs:
        mean_acc = np.mean(lang_overall_accs)
        std_acc = np.std(lang_overall_accs)
        avg_lines.append(
            f"\nMEAN OVERALL ACCURACY (avg of per-language facet accuracies): {mean_acc:.3f} (std={std_acc:.3f})"
        )

    if lang_core_accs:
        mean_core = np.mean(lang_core_accs)
        std_core = np.std(lang_core_accs)
        avg_lines.append(
            f"MEAN CORE REGISTER ACCURACY (avg of per-language): {mean_core:.3f} (std={std_core:.3f})"
        )

    avg_lines.append("")

    # Pooled per-facet metrics
    for facet in FACETS:
        pairs = all_pairs_by_facet[facet]
        if not pairs:
            avg_lines.append(f"\n--- {facet}: no valid pairs ---\n")
            continue
        acc, report, cm, labels = compute_metrics(pairs)
        avg_lines.append(f"\n--- {facet} (n={len(pairs)}, accuracy={acc:.3f}) ---")
        avg_lines.append(report)
        avg_lines.append("Confusion Matrix (rows=human, cols=LLM):")
        avg_lines.append(format_cm(cm, labels))
        avg_lines.append("")

    # Pooled CORE register metrics
    if all_core_pairs:
        acc, report, cm, labels = compute_metrics(all_core_pairs)
        avg_lines.append(f"\n{'=' * 70}")
        avg_lines.append(
            f"DERIVED CORE REGISTER - POOLED (n={len(all_core_pairs)}, accuracy={acc:.3f})"
        )
        avg_lines.append(f"{'=' * 70}")
        avg_lines.append(report)
        avg_lines.append("Confusion Matrix (rows=human, cols=LLM):")
        avg_lines.append(format_cm(cm, labels))
        avg_lines.append("")

    avg_report = "\n".join(avg_lines)
    all_reports.append(avg_report)

    with open(os.path.join(OUT_DIR, "pooled_report.txt"), "w") as f:
        f.write(avg_report)
    print(f"  Saved pooled_report.txt")

    # Save combined report
    with open(os.path.join(OUT_DIR, "full_report.txt"), "w") as f:
        f.write("\n\n".join(all_reports))
    print(f"  Saved full_report.txt")


if __name__ == "__main__":
    main()
