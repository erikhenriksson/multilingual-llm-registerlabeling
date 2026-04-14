import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
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

lengths = []
n_registers = []

for fname, lang in FILE_LANG_MAP.items():
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments = json.loads(line)
            total_len = sum(len(seg["text"]) for seg in segments)
            distinct = len(set(seg["label"] for seg in segments))
            lengths.append(total_len)
            n_registers.append(distinct)

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(lengths, n_registers, alpha=0.3, s=15, color="orange", edgecolors="none")
ax.set_xscale("log")
ax.set_xlabel("Text length (characters, log scale)")
ax.set_ylabel("Number of distinct registers")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "length_vs_registers.png"), dpi=300)
plt.close()
print("Saved to plots/length_vs_registers.png")
