#!/usr/bin/env python3
"""Parallel full-index pass for the frozen 40-query NQ engineering pilot."""
import concurrent.futures
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
INDEX = ROOT / "pilot_indexes" / "bm25_2681468_full.sqlite"
OUT = ROOT / "pilot_full_parallel_results.jsonl"
TOKENS = re.compile(r"[A-Za-z0-9]+")
DB_URL = f"file:{INDEX}?mode=ro"


def get_qrels():
    qrels = defaultdict(set)
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels[qid].add(docid)
    return qrels


def run_one(task):
    qid, query, gold = task
    words = list(dict.fromkeys(t.lower() for t in TOKENS.findall(query)))
    match = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
    with sqlite3.connect(DB_URL, uri=True) as db:
        hits = db.execute(
            "SELECT docid, bm25(passages, 2.0, 1.0) AS score "
            "FROM passages WHERE passages MATCH ? "
            "ORDER BY score ASC, docid ASC LIMIT 10",
            (match,),
        ).fetchall()
    hit_ids = [r[0] for r in hits]
    ranks = [i + 1 for i, docid in enumerate(hit_ids) if docid in gold]
    return {
        "qid": qid,
        "size": 2_681_468,
        "gold_count": len(gold),
        "relevant_in_top10": len(ranks),
        "success_at_10": int(bool(ranks)),
        "recall_at_10": len(ranks) / len(gold),
        "mrr_at_10": 1 / ranks[0] if ranks else 0.0,
        "top10": [{"docid": did, "score": score, "relevant": did in gold}
                  for did, score in hits],
    }


def main():
    queries = {}
    for line in (DATA / "queries.jsonl").open():
        row = json.loads(line)
        queries[row["_id"]] = row["text"]
    qids = sorted((ROOT / "pilot_qids.txt").read_text().split())
    qrels = get_qrels()
    tasks = [(qid, queries[qid], qrels[qid]) for qid in qids]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 5 == 0:
                ordered = sorted(results, key=lambda x: x["qid"])
                OUT.write_text("".join(json.dumps(x) + "\n" for x in ordered))
                print(f"completed {len(results)}/{len(tasks)}", flush=True)
    results.sort(key=lambda x: x["qid"])
    OUT.write_text("".join(json.dumps(x) + "\n" for x in results))
    n = len(results)
    print(f"size=2,681,468 success@10={sum(x['success_at_10'] for x in results)}/{n} "
          f"macro_recall@10={sum(x['recall_at_10'] for x in results)/n:.3f} "
          f"mrr@10={sum(x['mrr_at_10'] for x in results)/n:.3f}")
    print(f"saved {n} rows to {OUT}")


if __name__ == "__main__":
    main()
