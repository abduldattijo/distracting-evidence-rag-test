#!/usr/bin/env python3
"""Run the frozen 40-question pilot with the Tantivy BM25 candidate system."""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import tantivy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
PILOT = sorted((ROOT / "pilot_qids.txt").read_text().split())
SIZES = (10_000, 100_000, 2_681_468)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
INDEX_PATHS = {
    10_000: ROOT / "tantivy_pilot_indexes" / "tantivy_10000_seed_11",
    100_000: ROOT / "tantivy_trial_100k",
    2_681_468: ROOT / "tantivy_pilot_indexes" / "tantivy_2681468_full",
}


def load_schema():
    b = tantivy.SchemaBuilder()
    b.add_text_field("docid", stored=True, tokenizer_name="raw")
    b.add_text_field("title", stored=False, index_option="freq")
    b.add_text_field("body", stored=False, index_option="freq")
    return b.build()


def main():
    queries = {}
    with (DATA / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in PILOT:
                queries[row["_id"]] = row["text"]
    qrels = defaultdict(set)
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels[qid].add(docid)

    output = ROOT / "pilot_tantivy_results.jsonl"
    rows = []
    for size in SIZES:
        index = tantivy.Index(load_schema(), path=str(INDEX_PATHS[size]))
        searcher = index.searcher()
        for qid in PILOT:
            start = time.perf_counter()
            words = list(dict.fromkeys(w.lower() for w in TOKEN_RE.findall(queries[qid])))
            parsed = " OR ".join('"' + w.replace('"', '\\"') + '"' for w in words)
            query = index.parse_query(parsed, ["title", "body"],
                                      field_boosts={"title": 2.0, "body": 1.0})
            result = searcher.search(query, limit=10, count=False)
            hits = []
            for score, address in result.hits:
                docid = searcher.doc(address)["docid"][0]
                hits.append({"docid": docid, "score": score, "relevant": docid in qrels[qid]})
            positions = [i + 1 for i, hit in enumerate(hits) if hit["relevant"]]
            rows.append({
                "qid": qid, "size": size, "gold_count": len(qrels[qid]),
                "relevant_in_top10": len(positions), "success_at_10": int(bool(positions)),
                "recall_at_10": len(positions) / len(qrels[qid]),
                "mrr_at_10": 1 / positions[0] if positions else 0.0,
                "latency_seconds": time.perf_counter() - start, "top10": hits,
            })
            if len(rows) % 10 == 0:
                output.write_text("".join(json.dumps(x) + "\n" for x in rows))
        subset = [r for r in rows if r["size"] == size]
        n = len(subset)
        print(f"size={size:,} success@10={sum(x['success_at_10'] for x in subset)}/{n} "
              f"macro_recall@10={sum(x['recall_at_10'] for x in subset)/n:.3f} "
              f"mrr@10={sum(x['mrr_at_10'] for x in subset)/n:.3f} "
              f"median_ms={sorted(x['latency_seconds'] for x in subset)[n//2]*1000:.2f}", flush=True)
        output.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
