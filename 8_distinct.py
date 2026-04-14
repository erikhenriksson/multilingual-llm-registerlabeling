import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = "core_mapped"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

FILE_LANG_MAP = {
    "cmn_Hans_annotated.jsonl": "Mandarin",
    "cmn_Hant_annotated.jsonl": "Mandarin",
    "eng_Latn_annotated.jsonl": "English",
    "fin_Latn_annotated.jsonl": "Finnish",
    "pes_Arab_annotated.jsonl": "Persian",
    "swe_Latn_annotated.jsonl": "Swedish",
}

# Each JSONL line = one document = JSON array of {"label": ..., "text": ...}
lang_doc_counts = defaultdict(list)

for fname, lang in FILE_LANG_MAP.items():
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments = json.loads(line)
            distinct = len(
                set(seg["label"] for seg in segments if seg["label"] != "Cannot rate")
            )
            # Cap at 6+
            lang_doc_counts[lang].append(min(distinct, 6))

# Build DataFrame for seaborn
rows = []
for lang, counts in lang_doc_counts.items():
    counter = Counter(counts)
    for n_reg, freq in counter.items():
        rows.append(
            {"Language": lang, "Number of Distinct Registers": n_reg, "Frequency": freq}
        )

df = pd.DataFrame(rows)

# Normalize to proportions per language for comparability
totals = df.groupby("Language")["Frequency"].transform("sum")
df["Proportion"] = df["Frequency"] / totals

# Compute mean and std of proportions across languages for each register count
labels = sorted(df["Number of Distinct Registers"].unique())
label_names = [str(x) if x < 6 else "6+" for x in labels]

languages = sorted(df["Language"].unique())
# Build matrix: rows=languages, cols=register counts
matrix = np.zeros((len(languages), len(labels)))
for i, lang in enumerate(languages):
    sub = df[df["Language"] == lang]
    for j, lab in enumerate(labels):
        val = sub[sub["Number of Distinct Registers"] == lab]["Proportion"].values
        matrix[i, j] = val[0] if len(val) > 0 else 0.0

means = matrix.mean(axis=0)
stds = matrix.std(axis=0)

# Plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(labels))
ax.bar(x, means, yerr=stds, capsize=5, color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels(label_names)
ax.set_xlabel("Number of Distinct Registers")
ax.set_ylabel("Proportion of Documents")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "register_dist_combined.png"), dpi=300)
plt.close()
print("Saved to plots/register_dist_combined.png")
