#!/usr/bin/env python3
"""Plot paired Success@10 for the three same-page distractor corpus draws."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "hard_arm_results.jsonl"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

rows = [json.loads(line) for line in RESULTS.open()]
qids = sorted({r["qid"] for r in rows})
draws = sorted({r["seed"] for r in rows})
grouped = {(r["qid"], r["seed"], r["condition"]): r["success_at_10"] for r in rows}
ordinary = [100 * np.mean([grouped[q, seed, "ordinary"] for q in qids]) for seed in draws]
hard = [100 * np.mean([grouped[q, seed, "hard"] for q in qids]) for seed in draws]

fig, ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
y = np.arange(len(draws))[::-1]
for yi, o, h, seed in zip(y, ordinary, hard, draws):
    ax.plot([h, o], [yi, yi], color="#AAB4C0", linewidth=2, zorder=1)
    ax.scatter(o, yi, s=70, color="#276FBF", edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter(h, yi, s=70, color="#D88719", edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(o + 0.5, yi, f"{o:.0f}%", ha="left", va="center", fontsize=10, color="#184A7A")
    ax.text(h - 0.5, yi, f"{h:.0f}%", ha="right", va="center", fontsize=10, color="#8D5200")

ax.set_yticks(y, [f"Draw {s}" for s in draws])
ax.set_ylim(-0.55, 2.7)
ax.set_xlim(75, 100)
ax.set_xticks(np.arange(75, 101, 5))
ax.set_xlabel("Questions with a qrel-positive passage in the top 10 (%)")
ax.set_title("Twenty same-page distractors reduced retrieval success in every draw", loc="left", weight="bold", pad=14)
ax.grid(axis="x", color="#E5E9EF", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0, pad=8)
ax.tick_params(axis="x", length=0, pad=6)

ax.scatter([], [], s=65, color="#276FBF", label="Ordinary distractors")
ax.scatter([], [], s=65, color="#D88719", label="20 reviewed same-page distractors")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.98), ncol=1)
fig.text(0.01, -0.01,
         "Each point is Success@10 over the same 100 questions in one paired 10,000-passage draw. "
         "Across draws: 90.7% ordinary vs 82.3% hard; paired question-bootstrap difference −8.3 pp "
         "(95% CI −13.7 to −3.7). Retrieval only; candidate labels were AI-assisted.",
         ha="left", va="top", fontsize=9, color="#475467", wrap=True)

fig.savefig(OUT / "hard_arm_success_at_10.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "hard_arm_success_at_10.svg", bbox_inches="tight", facecolor="white")
print(OUT / "hard_arm_success_at_10.png")
