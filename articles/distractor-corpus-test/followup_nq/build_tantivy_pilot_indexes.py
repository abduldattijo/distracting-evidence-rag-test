#!/usr/bin/env python3
"""Build Tantivy BM25 pilot indexes for frozen seed 11 and the full corpus."""
import hashlib
import json
import time
from pathlib import Path

import tantivy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
OUT = ROOT / "tantivy_pilot_indexes"
FULL_SIZE = 2_681_468
SIZES = (10_000, FULL_SIZE)
MANIFEST = ROOT / "pilot_indexes" / "manifest_seed_11.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schema():
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("docid", stored=True, tokenizer_name="raw")
    builder.add_text_field("title", stored=False, index_option="freq")
    builder.add_text_field("body", stored=False, index_option="freq")
    return builder.build()


def build(size, ids=None):
    path = OUT / f"tantivy_{size}_seed_11" if size != FULL_SIZE else OUT / f"tantivy_{size}_full"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing index: {path}")
    path.mkdir(parents=True)
    index = tantivy.Index(schema(), path=str(path), reuse=False)
    writer = index.writer(heap_size=512_000_000, num_threads=4)
    count = 0
    start = time.time()
    with (DATA / "corpus.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            docid = row["_id"]
            if ids is not None and docid not in ids:
                continue
            doc = tantivy.Document()
            doc.add_text("docid", docid)
            doc.add_text("title", row.get("title", "") or "")
            doc.add_text("body", row.get("text", "") or "")
            writer.add_document(doc)
            count += 1
            if count % 100_000 == 0:
                print(f"{size:,}: indexed {count:,} in {time.time()-start:.1f}s", flush=True)
    if count != size:
        raise ValueError(f"Expected {size:,} passages, indexed {count:,}")
    writer.commit()
    index.reload()
    files = [p for p in path.rglob("*") if p.is_file()]
    print(json.dumps({"size": size, "path": str(path),
                      "bytes": sum(p.stat().st_size for p in files),
                      "files_sha256": {p.name: sha256(p) for p in files}}, indent=2), flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    manifests = json.loads(MANIFEST.read_text())
    ids = set(manifests[str(SIZES[0])])
    build(SIZES[0], ids)
    build(FULL_SIZE)


if __name__ == "__main__":
    main()
