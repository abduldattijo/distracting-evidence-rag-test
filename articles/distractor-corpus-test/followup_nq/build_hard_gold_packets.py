#!/usr/bin/env python3
"""Prepare answer-key packets to manually check gold relevance for hard-arm queries."""
import json
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    suffix = f"_{args.suffix}" if args.suffix else ""
    QIDS = set(json.loads((ROOT / f"hard_arm_manifest{suffix}.json").read_text())["selected_qids"])
    queries = {}
    for row in map(json.loads, (DATA / "queries.jsonl").open()):
        if row["_id"] in QIDS:
            queries[row["_id"]] = row["text"]
    qrels = defaultdict(set)
    for line in (DATA / "qrels" / "test.tsv").open():
        qid, docid, score = line.rstrip("\n").split("\t")
        if qid in QIDS and int(score) > 0:
            qrels[qid].add(docid)
    gold = defaultdict(list)
    all_gold = set().union(*(qrels[qid] for qid in QIDS))
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in all_gold:
                gold[row["_id"]] = {"docid": row["_id"], "title": row.get("title", "") or "",
                                    "text": row.get("text", "") or ""}
    out = ROOT / f"hard_gold_review_packets{suffix}.jsonl"
    with out.open("w") as f:
        for qid in sorted(QIDS):
            f.write(json.dumps({"qid": qid, "query": queries[qid],
                                "gold": [gold[did] for did in sorted(qrels[qid]) if did in gold]},
                               ensure_ascii=False) + "\n")
    print(f"saved {len(QIDS)} gold packets; {len(all_gold)} unique qrel passages to {out}")


if __name__ == "__main__":
    main()
