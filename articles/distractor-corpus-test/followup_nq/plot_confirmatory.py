#!/usr/bin/env python3
"""Create the primary publication figure from the frozen NQ results."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "confirmatory_results_ranked.jsonl"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
QIDS = sorted((ROOT / "confirmatory_qids.txt").read_text().split())
SIZES = (10_000, 100_000, 500_000, 1_000_000, 2_681_468)
SEEDS = (11, 23, 37)
FULL = 2_681_468


def main():
    grouped = defaultdict(dict)
    for line in RESULTS.open():
        row = json.loads(line)
        grouped[(row["size"], row["seed"])][row["qid"]] = row["success_at_10"]
    rng = np.random.default_rng(20261030)
    indices = rng.integers(0, len(QIDS), size=(10_000, len(QIDS)), dtype=np.int32)

    mean_values, lower, upper, draw_values = [], [], [], defaultdict(list)
    for size in SIZES:
        if size == FULL:
            per_q = np.array([grouped[(FULL, None)][q] for q in QIDS], dtype=float)
            for_seed = [per_q]
        else:
            for_seed = [np.array([grouped[(size, seed)][q] for q in QIDS], dtype=float)
                        for seed in SEEDS]
            per_q = np.mean(np.stack(for_seed), axis=0)
            for draw in for_seed:
                draw_values[size].append(float(draw.mean()))
        mean_values.append(float(per_q.mean()))
        boots = per_q[indices].mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        lower.append(float(per_q.mean() - lo))
        upper.append(float(hi - per_q.mean()))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.labelcolor": "#303943",
                         "text.color": "#202830", "axes.edgecolor": "#88919a",
                         "xtick.color": "#46515c", "ytick.color": "#46515c"})
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    x = np.array(SIZES, dtype=float)
    blue = "#276A9B"
    gray = "#A6B2BC"
    offsets = (-0.08, 0.0, 0.08)
    for size in SIZES[:-1]:
        vals = draw_values[size]
        ax.scatter([size * np.exp(o) for o in offsets], vals, s=42, facecolors="white", edgecolors=gray,
                   linewidths=1.1, zorder=3)
    ax.errorbar(x, mean_values, yerr=np.array([lower, upper]), color=blue, marker="o",
                markersize=6, linewidth=2.2, capsize=3, ecolor=blue, elinewidth=1.1,
                label="Mean across random draws; 95% question-bootstrap CI", zorder=4)
    baseline = mean_values[0]
    threshold_y = baseline - 0.05
    ax.axhline(threshold_y, color="#737D86", linestyle=(0, (4, 3)), linewidth=1.1,
               label="5 percentage points below 10k")
    ax.axvline(100_000, color="#C8802D", linestyle=(0, (3, 3)), linewidth=1.2)
    ax.annotate("First tested size meeting\nthe pre-set degradation rule",
                xy=(100_000, mean_values[1]), xytext=(155_000, 0.83),
                arrowprops={"arrowstyle": "->", "color": "#8B5A24", "lw": 1.0},
                color="#704515", fontsize=9, ha="left", va="center")
    ax.set_xscale("log")
    ax.set_xticks(x, ["10k", "100k", "500k", "1m", "2.68m"])
    ax.set_ylim(0.35, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1), [f"{v:.0%}" for v in np.arange(0.4, 1.01, 0.1)])
    ax.set_xlabel("Wikipedia passages in searchable corpus (log scale)")
    ax.set_ylabel("BEIR-qrel Success@10")
    ax.set_title("BM25 retrieves fewer judged relevant passages as the corpus grows", loc="left", pad=13)
    ax.grid(axis="y", color="#DCE2E7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.text(0.01, -0.035,
             "3,412 BEIR Natural Questions test queries; Tantivy BM25, no generator. "
             "Question-bootstrap intervals do not include random-draw variation; the three draw results are shown as open circles.",
             fontsize=8, color="#4B5660")
    fig.savefig(OUT / "success_at_10_by_corpus_size.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "success_at_10_by_corpus_size.svg", bbox_inches="tight")
    print(json.dumps({"sizes": list(SIZES), "mean_success": mean_values,
                      "draws": {str(k): v for k, v in draw_values.items()},
                      "five_point_threshold": threshold_y}, indent=2))


if __name__ == "__main__":
    main()
