#!/usr/bin/env python3
"""Run confirmatory BEIR NQ retrieval across the frozen nested Tantivy indexes."""
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import tantivy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
INDEX_ROOT = ROOT / "confirmatory_indexes"
QIDS = sorted((ROOT / "confirmatory_qids.txt").read_text().split())
SIZES = (10_000, 100_000, 500_000, 1_000_000)
SEEDS = (11, 23, 37)
FULL = 2_681_468
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
OUT = ROOT / "confirmatory_results.jsonl"


def load_schema():
    b = tantivy.SchemaBuilder()
    b.add_text_field("docid", stored=True, tokenizer_name="raw")
    b.add_text_field("title", stored=False, index_option="freq")
    b.add_text_field("body", stored=False, index_option="freq")
    return b.build()


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing result file: {OUT}")
    queries = {}
    with (DATA / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in set(QIDS):
                queries[row["_id"]] = row["text"]
    if len(queries) != len(QIDS):
        raise ValueError("confirmatory query count mismatch")
    qrels = defaultdict(set)
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if qid in queries and int(score) > 0:
                qrels[qid].add(docid)
    if any(not qrels[qid] for qid in QIDS):
        raise ValueError("a confirmatory question has no positive qrels")

    conditions = [(size, seed) for seed in SEEDS for size in SIZES]
    conditions.append((FULL, None))
    started = time.time()
    total_done = 0
    with OUT.open("x") as output:
        for size, seed in conditions:
            name = f"{size}_full" if seed is None else f"{size}_seed_{seed}"
            index_path = INDEX_ROOT / f"tantivy_{name}"
            index = tantivy.Index(load_schema(), path=str(index_path))
            searcher = index.searcher()
            cond_rows = 0
            cond_success = 0
            cond_recall = 0.0
            cond_mrr = 0.0
            latencies = []
            for qid in QIDS:
                words = list(dict.fromkeys(w.lower() for w in TOKEN_RE.findall(queries[qid])))
                parsed = " OR ".join('"' + w.replace('"', '\\"') + '"' for w in words)
                start = time.perf_counter()
                query = index.parse_query(parsed, ["title", "body"],
                                          field_boosts={"title": 2.0, "body": 1.0})
                result = searcher.search(query, limit=100, count=False)
                latency = time.perf_counter() - start
                hits = []
                for score, address in result.hits:
                    docid = searcher.doc(address)["docid"][0]
                    hits.append({"docid": docid, "score": score, "relevant": docid in qrels[qid]})
                positions = [i + 1 for i, hit in enumerate(hits[:10]) if hit["relevant"]]
                row = {
                    "qid": qid, "size": size, "seed": seed,
                    "gold_count": len(qrels[qid]),
                    "relevant_in_top10": len(positions),
                    "success_at_10": int(bool(positions)),
                    "recall_at_10": len(positions) / len(qrels[qid]),
                    "mrr_at_10": 1 / positions[0] if positions else 0.0,
                    "latency_seconds": latency,
                    "top100": hits,
                }
                output.write(json.dumps(row) + "\n")
                cond_rows += 1
                cond_success += row["success_at_10"]
                cond_recall += row["recall_at_10"]
                cond_mrr += row["mrr_at_10"]
                latencies.append(latency)
                total_done += 1
                if total_done % 500 == 0:
                    output.flush()
                    print(f"completed {total_done}/{len(QIDS)*len(conditions)}; "
                          f"elapsed={time.time()-started:.1f}s", flush=True)
            output.flush()
            latencies.sort()
            print(f"condition={name} success@10={cond_success}/{cond_rows} "
                  f"({cond_success/cond_rows:.4f}) macro_recall@10={cond_recall/cond_rows:.4f} "
                  f"mrr@10={cond_mrr/cond_rows:.4f} median_latency_ms={latencies[len(latencies)//2]*1000:.2f} "
                  f"elapsed={time.time()-started:.1f}s", flush=True)
    print(f"saved {total_done} question-condition rows to {OUT}")


if __name__ == "__main__":
    main()
