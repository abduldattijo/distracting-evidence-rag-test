#!/usr/bin/env python3
"""Print one blinded hard-negative review packet with its qrel answer references."""
import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"


def main(qid, suffix=""):
    suffix = f"_{suffix}" if suffix else ""
    packets = {r["qid"]: r for r in map(json.loads, (ROOT / f"hard_review_packets{suffix}.jsonl").open())}
    if qid not in packets:
        raise SystemExit(f"unknown packet qid: {qid}")
    packet = packets[qid]
    positive = set()
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            q, did, score = line.rstrip("\n").split("\t")
            if q == qid and int(score) > 0:
                positive.add(did)
    gold = []
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in positive:
                gold.append(row)
    print(f"QUESTION {qid}: {packet['query']}\n")
    print("REFERENCE ANSWER PASSAGES (use to check factual support):")
    for i, row in enumerate(gold, 1):
        print(f"G{i} | {row.get('title','')} | {row.get('text','')}\n")
    print("BLINDED CANDIDATES (random order; candidate IDs are anonymized):")
    for i, row in enumerate(packet["candidates"], 1):
        print(f"C{i:02d} | {row['title']} | {row['text']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("qid")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    main(args.qid, args.suffix)
