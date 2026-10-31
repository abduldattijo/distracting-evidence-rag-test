#!/usr/bin/env python3
"""Featured chart for the article, drawn from the saved Wikipedia size results (BM25, success at 10)."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / "followup_nq" / "confirmatory_results_ranked.jsonl").open()]
hits = defaultdict(list)
for r in rows:
    hits[(r["size"], r["seed"])].append(r["success_at_10"])
by_size = defaultdict(list)
for (size, _), v in hits.items():
    by_size[size].append(100 * sum(v) / len(v))
sizes = sorted(by_size)
scores = [sum(by_size[s]) / len(by_size[s]) for s in sizes]
labels = ["10 thousand", "100 thousand", "500 thousand", "1 million", "2.7 million"]

fig, ax = plt.subplots(figsize=(8.4, 4.72), constrained_layout=True)
colors = ["#276FBF", "#3F83C9", "#6F9FCB", "#C79A5B", "#D88719"]
bars = ax.bar(range(len(sizes)), scores, color=colors, width=0.72)
for i, s in enumerate(scores):
    ax.text(i, s + 1.5, f"{s:.1f}", ha="center", va="bottom", fontsize=16, weight="bold", color="#1F2933")
ax.set_xticks(range(len(sizes)), labels, fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", length=0, pad=8)
ax.set_xlabel("Passages in the search collection", fontsize=11, labelpad=10)
out = ROOT / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "featured_chart.png", dpi=200, facecolor="white")
print(out / "featured_chart.png")
