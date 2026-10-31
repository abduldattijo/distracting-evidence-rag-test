#!/usr/bin/env python3
"""Plot recall at 5 for the smallest and largest collection in the arXiv test (screened run)."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / "full_gold_screened.jsonl").open()]
cell = defaultdict(lambda: defaultdict(list))
for r in rows:
    cell[(r["condition"], r["method"], r["representation"], r["corpus_size"])][r["qid"]].append(float(r["evidence_recall_at_5"]))

def recall(c, m, rep, n):
    per_q = [sum(v) / len(v) for v in cell[(c, m, rep, n)].values()]
    return 100 * sum(per_q) / len(per_q)

names = {("bm25", "flat"): "BM25, flat chunks", ("bm25", "parent"): "BM25, parent documents",
         ("embeddings", "flat"): "Embeddings, flat chunks", ("embeddings", "parent"): "Embeddings, parent documents"}
groups = [("ordinary", "Ordinary distractors"), ("look_alike", "Look alike distractors")]

fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True, constrained_layout=True)
for ax, (cond, title) in zip(axes, groups):
    keys = list(names)[::-1]
    y = np.arange(len(keys))
    for yi, (m, rep) in zip(y, keys):
        a, b = recall(cond, m, rep, 100), recall(cond, m, rep, 1801)
        ax.plot([b, a], [yi, yi], color="#AAB4C0", linewidth=2, zorder=1)
        ax.scatter(a, yi, s=70, color="#276FBF", edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(b, yi, s=70, color="#D88719", edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(a + 0.15, yi + 0.22, f"{a:.1f}", ha="left", va="center", fontsize=9, color="#184A7A")
        ax.text(b - 0.15, yi + 0.22, f"{b:.1f}", ha="right", va="center", fontsize=9, color="#8D5200")
    ax.set_yticks(y, [names[k] for k in keys])
    ax.set_xlim(92, 100.5)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.grid(axis="x", color="#E5E9EF", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[1].set_xlabel("Questions with a supporting paper in the top 5 (percent)")
axes[0].scatter([], [], s=65, color="#276FBF", label="100 papers")
axes[0].scatter([], [], s=65, color="#D88719", label="1,801 papers")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("Recall at 5 lost 3.4 points at most as the arXiv collection grew", x=0.01, ha="left", weight="bold")
out = ROOT / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "arxiv_recall_by_size.png", dpi=200, bbox_inches="tight", facecolor="white")
print(out / "arxiv_recall_by_size.png")
