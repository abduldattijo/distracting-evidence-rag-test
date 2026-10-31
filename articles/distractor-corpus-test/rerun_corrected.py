"""Rerun judging for questions whose requirements were corrected after reading the gold abstracts.

Paid, strict budget (total 0.03 US dollars across the whole gate and rerun exercise, of which
0.0118 was already spent by gate_sonnet.py, so the default here is the 0.0182 that is left).
Uses the same candidates, shuffle seeds and prompts as screen_v3.py. Haiku judges, then Sonnet
reviews flagged papers, the control and 2 random none papers, if the budget still allows.
Output goes to screen_v3b_* so the v3 results stay as they were.
"""
import argparse, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "agentic-rag-eval"))
import llm  # noqa: E402
import run_pilot as rp  # noqa: E402
import screen_v3 as sv  # noqa: E402

# Corrected requirement sets: the Haiku-kept ones plus the ones Sonnet kept and the gold abstract supports.
R = json.load((HERE / "screen_v3_requirements.json").open())
ORIG = {q: R["requirements"][q] + R["dropped_by_gold_gate"].get(q, []) for q in R["requirements"]}
CORRECTED = {
    "q000": [ORIG["q000"][0], ORIG["q000"][1]],   # drops only the anneal requirement
    "q009": list(ORIG["q009"]),                    # both kept
}

ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=float, default=0.0182)
args = ap.parse_args()
budget = sv.Budget(args.budget)
papers, questions = rp.load_data()
paper_idx = {p["id"]: i for i, p in enumerate(papers)}
qs = [q for q in questions if q["qid"] in CORRECTED]
model = rp.load_sentence_transformer(rp.DEFAULT_SELECTION_MODEL, rp.DEFAULT_SELECTION_REVISION)
orders = rp.selection_orders(papers, [q for q in questions[:10]], paper_idx, model, "gold")
del model
raw = (HERE / "screen_v3b_raw.jsonl").open("w")
labels, layouts = {}, {}
for q in qs:
    cand = orders[q["qid"]][:50]
    gold = paper_idx[q["gold_id"]]
    entries = [(pi, rank + 1) for rank, pi in enumerate(cand)] + [(gold, 0)]
    random.Random(q["qid"]).shuffle(entries)
    items = [(f"c{i + 1:02d}", papers[pi]) for i, (pi, _) in enumerate(entries)]
    layouts[q["qid"]] = [(cid, p["id"], rank) for (cid, p), (_, rank) in zip(items, entries)]
    ids = [cid for cid, _ in items]
    labels[q["qid"]] = sv.call(budget, raw, f"judge:{q['qid']}", sv.HAIKU, sv.JUDGE_SYSTEM,
                               sv.prompt(q["question"], CORRECTED[q["qid"]], items), 1500, sv.make_parse_labels(ids))
    print(q["qid"], "judged", labels[q["qid"]] is not None, f"spent={llm.LEDGER.usd:.4f}", flush=True)
review = {}
for q in sorted(qs, key=lambda q: -sum(1 for t in layouts[q["qid"]] if (labels[q["qid"]] or {}).get(t[0], "none") != "none" and t[2] != 0)):
    lab = labels[q["qid"]]
    if lab is None:
        continue
    lay = layouts[q["qid"]]
    nones = [t for t in lay if lab[t[0]] == "none" and t[2] != 0]
    chosen = {t[0] for t in lay if lab[t[0]] != "none" or t[2] == 0}
    chosen |= {t[0] for t in random.Random(q["qid"]).sample(nones, min(2, len(nones)))}
    ordered = sorted((t for t in lay if t[0] in chosen), key=lambda t: t[0])
    sub = [(f"r{n + 1:02d}", papers[paper_idx[t[1]]]) for n, t in enumerate(ordered)]
    mapping = {f"r{n + 1:02d}": t for n, t in enumerate(ordered)}
    out = sv.call(budget, raw, f"review:{q['qid']}", sv.SONNET, sv.JUDGE_SYSTEM,
                  sv.prompt(q["question"], CORRECTED[q["qid"]], sub), 300, sv.make_parse_labels([c for c, _ in sub]), retries=0)
    if out is not None:
        review[q["qid"]] = {mapping[k][1]: out[k] for k in out}
    print(q["qid"], "reviewed", out is not None, f"spent={llm.LEDGER.usd:.4f}", flush=True)
raw.close()
with (HERE / "screen_v3b_labels.jsonl").open("w") as f:
    for q in qs:
        for cid, pid, rank in layouts[q["qid"]]:
            f.write(json.dumps({"qid": q["qid"], "paper_id": pid, "gold_rank": rank, "is_control": rank == 0,
                                "haiku": (labels[q["qid"]] or {}).get(cid, "missing"),
                                "sonnet": review.get(q["qid"], {}).get(pid)}) + "\n")
print(f"calls={llm.LEDGER.calls} spent={llm.LEDGER.usd:.4f} budget={args.budget} skipped_for_budget={budget.skipped}")
