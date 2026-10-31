#!/usr/bin/env python3
"""Run the frozen 40-question NQ engineering/headroom pilot."""
import json
import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
INDEXES = ROOT / "pilot_indexes"
PILOT_QIDS = set((ROOT / "pilot_qids.txt").read_text().split())
SIZES = (10_000, 100_000, 2_681_468)
TOKENS = re.compile(r"[A-Za-z0-9]+")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="10000,100000,2681468",
                        help="comma-separated corpus sizes")
    parser.add_argument("--output", default="pilot_results.jsonl")
    args = parser.parse_args()
    sizes = tuple(int(n) for n in args.sizes.split(","))
    queries = {}
    with (DATA / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in PILOT_QIDS:
                queries[row["_id"]] = row["text"]
    if set(queries) != PILOT_QIDS:
        raise ValueError("pilot query IDs do not match frozen query file")

    qrels = defaultdict(set)
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels[qid].add(docid)

    results = []
    out = ROOT / args.output
    for size in sizes:
        path = INDEXES / (f"bm25_{size}_seed_11.sqlite" if size != 2_681_468 else "bm25_2681468_full.sqlite")
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for qid in sorted(PILOT_QIDS):
            words = list(dict.fromkeys(t.lower() for t in TOKENS.findall(queries[qid])))
            match = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
            hits = db.execute(
                "SELECT docid, bm25(passages, 2.0, 1.0) AS score "
                "FROM passages WHERE passages MATCH ? "
                "ORDER BY score ASC, docid ASC LIMIT 10",
                (match,),
            ).fetchall()
            hit_ids = [r[0] for r in hits]
            gold = qrels[qid]
            rel_positions = [i + 1 for i, docid in enumerate(hit_ids) if docid in gold]
            results.append({
                "qid": qid,
                "size": size,
                "gold_count": len(gold),
                "relevant_in_top10": len(rel_positions),
                "success_at_10": int(bool(rel_positions)),
                "recall_at_10": len(rel_positions) / len(gold),
                "mrr_at_10": 1 / rel_positions[0] if rel_positions else 0.0,
                "top10": [{"docid": docid, "score": score, "relevant": docid in gold}
                          for docid, score in hits],
            })
            if len(results) % 10 == 0:
                out.write_text("".join(json.dumps(row) + "\n" for row in results))
        db.close()
        sample = [r for r in results if r["size"] == size]
        print(f"size={size:,} success@10={sum(r['success_at_10'] for r in sample)}/{len(sample)} "
              f"macro_recall@10={sum(r['recall_at_10'] for r in sample)/len(sample):.3f} "
              f"mrr@10={sum(r['mrr_at_10'] for r in sample)/len(sample):.3f}", flush=True)
        out.write_text("".join(json.dumps(row) + "\n" for row in results))

    print(f"saved {len(results)} rows to {out}")


if __name__ == "__main__":
    main()
