"""Summarize the full pre-screen retrieval run. Local only.

Recall@5 and mean reciprocal rank per cell, by condition and corpus size, with a paired
bootstrap (10,000 resamples of the 298 questions, seed 0) for the change from N = 100.
Ordinary rows are averaged over the three draws for each question first, and each draw is
also shown. The breaking point rule is the one in PROTOCOL.md section 8.
"""
import json, sys, collections
import numpy as np

mode = sys.argv[1]
drop = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
rows = [json.loads(l) for l in open(f"full_{mode}.jsonl") if json.loads(l)["qid"] not in drop]
# keep only sizes that every remaining question has, so an incomplete anchor (N=2000) is never averaged
_n = collections.Counter((r["condition"], r["corpus_size"]) for r in rows)
_full = {k: v for k, v in _n.items()}
_q = len({r["qid"] for r in rows})
rows = [r for r in rows if r["condition"] != "look_alike" or _full[(r["condition"], r["corpus_size"])] >= _q * 4 * 1 or r["corpus_size"] != 2000]
if drop: print("dropped questions:", sorted(drop))
rng = np.random.default_rng(0)
qids = sorted({r["qid"] for r in rows})
qi = {q: i for i, q in enumerate(qids)}
cells = [(m, rep) for m in ("bm25", "embeddings") for rep in ("flat", "parent")]
sizes = sorted({r["corpus_size"] for r in rows})

def table(cond, m, rep, metric):
    """array [size, draw, question] of the metric (draw 0 only for look_alike)"""
    out = {}
    for r in rows:
        if r["condition"] == cond and r["method"] == m and r["representation"] == rep:
            d = r["seed"] or 0
            out.setdefault((r["corpus_size"], d), np.full(len(qids), np.nan))[qi[r["qid"]]] = float(r[metric])
    return out

B = rng.integers(0, len(qids), size=(10000, len(qids)))
print(f"== selection by {mode}: {len(rows)} rows, {len(qids)} questions\n")
for cond in ("look_alike", "ordinary"):
    print(f"-- {cond}")
    for m, rep in cells:
        t = table(cond, m, rep, "evidence_recall_at_5")
        mrr = table(cond, m, rep, "mrr_at_5")
        base_sz = 100
        line = []
        def per_q(tab, sz):
            arrs = [v for (s, d), v in tab.items() if s == sz]
            return np.mean(arrs, axis=0)
        base = per_q(t, base_sz)
        for sz in sizes:
            if not any(s == sz for s, _ in t):
                continue
            cur = per_q(t, sz)
            diff = cur - base
            bs = diff[B].mean(axis=1) * 100
            lo, hi = np.percentile(bs, [2.5, 97.5])
            draws = ""
            if cond == "ordinary":
                draws = " draws " + "/".join(f"{np.mean(v)*100:.1f}" for (s, d), v in sorted(t.items()) if s == sz)
            line.append(f"N={sz}: recall {cur.mean()*100:.1f} mrr {per_q(mrr, sz).mean():.3f} change {diff.mean()*100:+.1f} [{lo:+.1f},{hi:+.1f}]{draws}")
        print(f"{m:10s}{rep:7s}")
        for l in line: print("   ", l)
    print()
