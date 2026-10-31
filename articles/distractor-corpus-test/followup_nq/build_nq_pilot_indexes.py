#!/usr/bin/env python3
"""Build nested SQLite FTS5 BM25 pilot indexes from frozen BEIR NQ data."""
import hashlib
import json
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
OUT = ROOT / "pilot_indexes"
OUT.mkdir(exist_ok=True)
SEED = 11
SIZES = (10_000, 100_000)
FULL_SIZE = 2_681_468


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_ids_and_qrels():
    qrels = set()
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            _, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.add(docid)
    ids = []
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            ids.append(json.loads(line)["_id"])
    if len(ids) != FULL_SIZE or len(set(ids)) != FULL_SIZE:
        raise ValueError("corpus count or uniqueness check failed")
    if not qrels <= set(ids):
        raise ValueError("a qrel-positive passage is absent from the corpus")
    return ids, qrels


def create_db(path):
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("PRAGMA cache_size=-262144")
    db.execute("CREATE VIRTUAL TABLE passages USING fts5(docid UNINDEXED, title, body, tokenize='unicode61')")
    return db


def main():
    ids, qrels = load_ids_and_qrels()
    non_gold = [docid for docid in ids if docid not in qrels]
    rng = random.Random(SEED)
    sampled = rng.sample(non_gold, max(SIZES) - len(qrels))
    ordered_non_gold = sampled
    manifests = {n: qrels | set(ordered_non_gold[:n-len(qrels)]) for n in SIZES}
    manifest_path = OUT / f"manifest_seed_{SEED}.json"
    manifest_path.write_text(json.dumps({str(n): sorted(manifests[n]) for n in SIZES}, separators=(",", ":")))

    dbs = {n: create_db(OUT / f"bm25_{n}_seed_{SEED}.sqlite") for n in SIZES}
    dbs[FULL_SIZE] = create_db(OUT / f"bm25_{FULL_SIZE}_full.sqlite")
    sets = {n: manifests[n] for n in SIZES}
    pending = {n: [] for n in dbs}
    inserted = {n: 0 for n in dbs}

    with (DATA / "corpus.jsonl").open() as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            docid = row["_id"]
            title = row.get("title", "") or ""
            body = row.get("text", "") or ""
            for n, db in dbs.items():
                if n == FULL_SIZE or docid in sets[n]:
                    pending[n].append((docid, title, body))
                    if len(pending[n]) >= 1000:
                        db.executemany("INSERT INTO passages(docid,title,body) VALUES(?,?,?)", pending[n])
                        db.commit()
                        inserted[n] += len(pending[n])
                        pending[n].clear()
            if line_no % 250_000 == 0:
                print(f"scanned {line_no:,}/{FULL_SIZE:,}; inserted " +
                      ", ".join(f"{n:,}:{inserted[n]+len(pending[n]):,}" for n in dbs), flush=True)

    for n, db in dbs.items():
        if pending[n]:
            db.executemany("INSERT INTO passages(docid,title,body) VALUES(?,?,?)", pending[n])
            db.commit()
            inserted[n] += len(pending[n])
        db.execute("INSERT INTO passages(passages) VALUES('optimize')")
        db.commit()
        db.close()
        if inserted[n] != n:
            raise ValueError(f"index size {n} contains {inserted[n]} rows")
        path = OUT / (f"bm25_{n}_seed_{SEED}.sqlite" if n != FULL_SIZE else f"bm25_{n}_full.sqlite")
        print(f"ready {n:,} rows: {path.name} ({path.stat().st_size:,} bytes; sha256 {sha256(path)})", flush=True)


if __name__ == "__main__":
    main()
