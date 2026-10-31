#!/usr/bin/env python3
"""Build the frozen nested Tantivy BM25 indexes for the confirmatory NQ run."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import tantivy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
OUT = ROOT / "confirmatory_indexes"
SIZES = (10_000, 100_000, 500_000, 1_000_000)
SEEDS = (11, 23, 37)
FULL = 2_681_468
CONFIRMATORY = set((ROOT / "confirmatory_qids.txt").read_text().split())


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_qrels_and_ids():
    qrels = defaultdict(set)
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels[qid].add(docid)
    all_positive = set().union(*qrels.values())
    confirmatory_positive = set().union(*(qrels[qid] for qid in CONFIRMATORY))
    corpus_ids = []
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            corpus_ids.append(json.loads(line)["_id"])
    if len(corpus_ids) != FULL or len(set(corpus_ids)) != FULL:
        raise ValueError("corpus row count/uniqueness check failed")
    return qrels, all_positive, confirmatory_positive, corpus_ids


def make_schema():
    b = tantivy.SchemaBuilder()
    b.add_text_field("docid", stored=True, tokenizer_name="raw")
    b.add_text_field("title", stored=False, index_option="freq")
    b.add_text_field("body", stored=False, index_option="freq")
    return b.build()


def main():
    OUT.mkdir(exist_ok=True)
    qrels, all_positive, confirmatory_positive, corpus_ids = load_qrels_and_ids()
    non_positive = [docid for docid in corpus_ids if docid not in all_positive]
    manifests = {}
    for seed in SEEDS:
        sampled = random.Random(seed).sample(non_positive, max(SIZES) - len(all_positive))
        for size in SIZES:
            manifests[f"{size}_seed_{seed}"] = all_positive | set(sampled[:size-len(all_positive)])
    manifests[f"{FULL}_full"] = set(corpus_ids)
    for name, ids in manifests.items():
        expected = FULL if name.endswith("_full") else int(name.split("_seed_")[0])
        if len(ids) != expected:
            raise ValueError(f"{name}: expected {expected}, got {len(ids)}")
        if not confirmatory_positive <= ids:
            raise ValueError(f"{name} misses confirmatory qrel-positive documents")
    for seed in SEEDS:
        for smaller, larger in zip(SIZES, SIZES[1:]):
            if not manifests[f"{smaller}_seed_{seed}"] <= manifests[f"{larger}_seed_{seed}"]:
                raise ValueError(f"non-nested manifests for seed {seed}")

    manifest_file = OUT / "id_manifests.json"
    manifest_file.write_text(json.dumps({name: sorted(ids) for name, ids in manifests.items()},
                                        separators=(",", ":")) + "\n")
    manifest_hash = sha256(manifest_file)
    print(f"manifest sha256={manifest_hash}; qrel-positive total={len(all_positive)}; "
          f"confirmatory-positive unique={len(confirmatory_positive)}", flush=True)

    schema = make_schema()
    indexes = {}
    writers = {}
    counts = defaultdict(int)
    for name in manifests:
        path = OUT / f"tantivy_{name}"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite existing index {path}")
        path.mkdir()
        index = tantivy.Index(schema, path=str(path), reuse=False)
        indexes[name] = index
        writers[name] = index.writer(heap_size=32_000_000, num_threads=1)

    with (DATA / "corpus.jsonl").open() as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            docid = row["_id"]
            for name, ids in manifests.items():
                if docid not in ids:
                    continue
                doc = tantivy.Document()
                doc.add_text("docid", docid)
                doc.add_text("title", row.get("title", "") or "")
                doc.add_text("body", row.get("text", "") or "")
                writers[name].add_document(doc)
                counts[name] += 1
            if line_no % 250_000 == 0:
                summary = f"scanned {line_no:,}/{FULL:,}; full={counts[f'{FULL}_full']:,}"
                print(summary, flush=True)

    metadata = {"tantivy_version": "0.25.1", "sizes": SIZES, "seeds": SEEDS,
                "full_size": FULL, "manifest_sha256": manifest_hash, "indexes": {}}
    for name, writer in writers.items():
        writer.commit()
        indexes[name].reload()
        expected = FULL if name.endswith("_full") else int(name.split("_seed_")[0])
        if counts[name] != expected:
            raise ValueError(f"{name}: inserted {counts[name]}, expected {expected}")
        path = OUT / f"tantivy_{name}"
        files = [p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".tmp")]
        metadata["indexes"][name] = {
            "doc_count": counts[name],
            "bytes": sum(p.stat().st_size for p in files),
            "file_sha256": {str(p.relative_to(path)): sha256(p) for p in files},
        }
        print(f"committed {name}: {counts[name]:,} docs; {metadata['indexes'][name]['bytes']:,} bytes", flush=True)
    (OUT / "index_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved metadata: {OUT / 'index_metadata.json'}")


if __name__ == "__main__":
    main()
