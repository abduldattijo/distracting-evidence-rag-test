#!/usr/bin/env python3
"""Validate and freeze BEIR NQ data before retrieval outcomes are inspected."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
ZIP = ROOT / "data" / "nq.zip"
PILOT_N = 40
PILOT_SEED = 20261029


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    queries = {}
    with (DATA / "queries.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            if x["_id"] in queries:
                raise ValueError("duplicate query ID")
            queries[x["_id"]] = x["text"]

    qrels = defaultdict(set)
    with (DATA / "qrels" / "test.tsv").open() as f:
        header = next(f).rstrip("\n").split("\t")
        if header != ["query-id", "corpus-id", "score"]:
            raise ValueError(f"unexpected qrels header: {header}")
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels[qid].add(did)

    if set(qrels) != set(queries):
        raise ValueError("query/qrels ID mismatch")
    gold_docs = set().union(*qrels.values())

    corpus_ids = set()
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            if x["_id"] in corpus_ids:
                raise ValueError("duplicate corpus ID")
            corpus_ids.add(x["_id"])
    missing = gold_docs - corpus_ids
    if missing:
        raise ValueError(f"{len(missing)} qrel-positive documents missing from corpus")

    all_qids = sorted(queries)
    rng = random.Random(PILOT_SEED)
    pilot_ids = sorted(rng.sample(all_qids, PILOT_N))
    pilot_set = set(pilot_ids)
    confirmatory_ids = [qid for qid in all_qids if qid not in pilot_set]

    manifest = {
        "dataset": "BEIR Natural Questions test",
        "archive_md5": "d4d3d2e48787a744b6f6e691ff534307",
        "archive_bytes": ZIP.stat().st_size,
        "corpus_rows": len(corpus_ids),
        "query_rows": len(queries),
        "positive_qrel_rows": sum(map(len, qrels.values())),
        "queries_with_positive_qrels": len(qrels),
        "unique_positive_document_ids": len(gold_docs),
        "missing_positive_document_ids": 0,
        "pilot_seed": PILOT_SEED,
        "pilot_query_count": len(pilot_ids),
        "confirmatory_query_count": len(confirmatory_ids),
        "sha256": {
            "archive": sha256(ZIP),
            "corpus": sha256(DATA / "corpus.jsonl"),
            "queries": sha256(DATA / "queries.jsonl"),
            "qrels": sha256(DATA / "qrels" / "test.tsv"),
        },
    }
    (ROOT / "nq_frozen.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "pilot_qids.txt").write_text("\n".join(pilot_ids) + "\n")
    (ROOT / "confirmatory_qids.txt").write_text("\n".join(confirmatory_ids) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
