#!/usr/bin/env python3
"""Build a throwaway 100k Tantivy BM25 index to measure search throughput."""
import json
import sqlite3
import time
from pathlib import Path

import tantivy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
OUT = ROOT / "tantivy_trial_100k"
MANIFEST = ROOT / "pilot_indexes" / "manifest_seed_11.json"
SIZE = 100_000


def main():
    ids = set(json.loads(MANIFEST.read_text())[str(SIZE)])
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing trial index: {OUT}")
    OUT.mkdir()
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("docid", stored=True, tokenizer_name="raw")
    schema_builder.add_text_field("title", stored=False, index_option="freq")
    schema_builder.add_text_field("body", stored=False, index_option="freq")
    schema = schema_builder.build()
    index = tantivy.Index(schema, path=str(OUT), reuse=False)
    writer = index.writer(heap_size=512_000_000, num_threads=4)
    count = 0
    started = time.time()
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] not in ids:
                continue
            doc = tantivy.Document()
            doc.add_text("docid", row["_id"])
            doc.add_text("title", row.get("title", "") or "")
            doc.add_text("body", row.get("text", "") or "")
            writer.add_document(doc)
            count += 1
            if count % 25_000 == 0:
                print(f"indexed {count:,}/{SIZE:,} in {time.time()-started:.1f}s", flush=True)
    writer.commit()
    index.reload()
    searcher = index.searcher()
    queries = {}
    for line in (DATA / "queries.jsonl").open():
        row = json.loads(line)
        queries[row["_id"]] = row["text"]
    pilot_ids = sorted((ROOT / "pilot_qids.txt").read_text().split())
    for qid in pilot_ids[:5]:
        start = time.time()
        query = index.parse_query(queries[qid], ["title", "body"],
                                  field_boosts={"title": 2.0, "body": 1.0})
        result = searcher.search(query, limit=10, count=False)
        elapsed = time.time() - start
        print(f"query={qid} seconds={elapsed:.4f} hits={len(result.hits)}", flush=True)
    print(f"ready docs={count} index_bytes={sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}")


if __name__ == "__main__":
    main()
