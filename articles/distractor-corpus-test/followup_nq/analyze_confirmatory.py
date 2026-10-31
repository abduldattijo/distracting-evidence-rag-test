#!/usr/bin/env python3
"""Validate rankings, apply the frozen tie-break, and bootstrap primary outcomes."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "confirmatory_results.jsonl"
RANKED = ROOT / "confirmatory_results_ranked.jsonl"
SUMMARY = ROOT / "confirmatory_summary.json"
QIDS = sorted((ROOT / "confirmatory_qids.txt").read_text().split())
SIZES = (10_000, 100_000, 500_000, 1_000_000)
SEEDS = (11, 23, 37)
FULL = 2_681_468
B = 10_000
BOOT_SEED = 20261030


def ndcg_at_10(hits, gold_count):
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(hits[:10]) if hit["relevant"])
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(10, gold_count)))
    return dcg / ideal if ideal else 0.0


def main():
    grouped = defaultdict(dict)
    for line in RAW.open():
        row = json.loads(line)
        key = (row["size"], row["seed"])
        if row["qid"] in grouped[key]:
            raise ValueError(f"duplicate row: {key}, {row['qid']}")
        grouped[key][row["qid"]] = row
    expected_conditions = {(s, seed) for s in SIZES for seed in SEEDS} | {(FULL, None)}
    if set(grouped) != expected_conditions:
        raise ValueError("condition set differs from frozen design")
    for key, rows in grouped.items():
        if set(rows) != set(QIDS):
            raise ValueError(f"question IDs incomplete for {key}")
        for qid, row in rows.items():
            if len(row["top100"]) != 100:
                raise ValueError(f"top 100 missing for {key}, {qid}")
            hits = row["top100"]
            if len({h["docid"] for h in hits}) != 100:
                raise ValueError(f"duplicate returned doc IDs for {key}, {qid}")
            if hits[9]["score"] == hits[-1]["score"]:
                raise ValueError(f"top-10 tie extends through result cutoff for {key}, {qid}")
            hits.sort(key=lambda h: (-h["score"], h["docid"]))
            pos = [i + 1 for i, h in enumerate(hits[:10]) if h["relevant"]]
            row["success_at_10"] = int(bool(pos))
            row["relevant_in_top10"] = len(pos)
            row["recall_at_10"] = len(pos) / row["gold_count"]
            row["mrr_at_10"] = 1 / pos[0] if pos else 0.0
            row["ndcg_at_10"] = ndcg_at_10(hits, row["gold_count"])
            row["tie_break"] = "score descending, then passage ID ascending within retrieved top 100"

    with RANKED.open("x") as out:
        for key in sorted(grouped, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
            for qid in QIDS:
                out.write(json.dumps(grouped[key][qid]) + "\n")

    def summarize(rows):
        return {
            "success_at_10": float(np.mean([r["success_at_10"] for r in rows])),
            "macro_qrel_recall_at_10": float(np.mean([r["recall_at_10"] for r in rows])),
            "mrr_at_10": float(np.mean([r["mrr_at_10"] for r in rows])),
            "ndcg_at_10": float(np.mean([r["ndcg_at_10"] for r in rows])),
            "median_query_latency_ms": float(np.median([r["latency_seconds"] for r in rows]) * 1000),
        }

    per_condition = {}
    for key, qrows in grouped.items():
        label = f"{key[0]}_full" if key[1] is None else f"{key[0]}_seed_{key[1]}"
        per_condition[label] = summarize([qrows[q] for q in QIDS])

    baseline = np.mean(np.array([[grouped[(10_000, seed)][q]["success_at_10"]
                                  for q in QIDS] for seed in SEEDS]), axis=0)
    rng = np.random.default_rng(BOOT_SEED)
    sampled = rng.integers(0, len(QIDS), size=(B, len(QIDS)), dtype=np.int32)
    growth = {}
    for size in (100_000, 500_000, 1_000_000, FULL):
        if size == FULL:
            target = np.array([grouped[(FULL, None)][q]["success_at_10"] for q in QIDS])
            target_point = float(np.mean(target))
        else:
            target = np.mean(np.array([[grouped[(size, seed)][q]["success_at_10"]
                                        for q in QIDS] for seed in SEEDS]), axis=0)
            target_point = float(np.mean(target))
        diff = target - baseline
        boot = diff[sampled].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        growth[str(size)] = {
            "success_at_10": target_point,
            "difference_from_10k": float(np.mean(diff)),
            "difference_ci95": [float(lo), float(hi)],
            "drop_points": float(-np.mean(diff) * 100),
            "operational_threshold_met": bool(np.mean(diff) <= -0.05 and hi < 0),
        }

    summary = {
        "n_confirmatory_questions": len(QIDS),
        "n_rows": sum(len(x) for x in grouped.values()),
        "bootstrap_resamples": B,
        "bootstrap_seed": BOOT_SEED,
        "tie_audit": "all conditions had 100 unique results; no tie at rank 10 extended through rank 100",
        "per_condition": per_condition,
        "growth_vs_10k": growth,
        "first_tested_degradation_size": next((int(size) for size, x in growth.items()
                                                if x["operational_threshold_met"]), None),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
