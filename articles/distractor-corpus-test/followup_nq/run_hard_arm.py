#!/usr/bin/env python3
"""Build and run the paired 10k same-page hard-distractor arm.

The ordinary and hard-mixed corpora are question-specific. For each question
and each of three fixed draws, the ordinary corpus contains all qrel positives
for that question plus random non-qrel passages, excluding the reviewed
same-page candidate pool. The paired hard corpus replaces 20 ordinary slots
with 20 reviewed `none` same-page passages. No generator is used.
"""
import hashlib
import gc
import json
import random
import re
import shutil
import sqlite3
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tantivy

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
OUT = ROOT / "hard_arm_results.jsonl"
SUMMARY = ROOT / "hard_arm_summary.json"
MANIFEST = ROOT / "hard_arm_run_manifest.json"
ID_MANIFEST = ROOT / "hard_arm_id_manifests.jsonl"
LABELS = ROOT / "hard_review_labels_confirm50_ai.jsonl"
SUPPLEMENTAL = ROOT / "hard_review_supplemental_ai.jsonl"
PACKETS = ROOT / "review_packets_confirm50_final100.jsonl"
RANKED_PRIVATE = ROOT / "candidates_ranked_PRIVATE_confirm50_final100.jsonl"
QREL_FILE = DATA / "qrels" / "test.tsv"
CORPUS_FILE = DATA / "corpus.jsonl"
QUERY_FILE = DATA / "queries.jsonl"
QIDS = sorted(r["qid"] for r in map(json.loads, PACKETS.open()))
SEEDS = (11, 23, 37)
N_DOCS = 10_000
N_HARD = 20
B = 10_000
BOOT_SEED = 20261030
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs():
    packet_rows = {r["qid"]: r for r in map(json.loads, PACKETS.open())}
    labels = {r["qid"]: r for r in map(json.loads, LABELS.open())}
    if set(labels) != set(QIDS):
        raise ValueError("candidate label qids do not match frozen sample")
    qrels = defaultdict(set)
    qrel_grades = defaultdict(dict)
    all_positive = set()
    with QREL_FILE.open() as qrel_in:
        next(qrel_in)
        qrel_lines = qrel_in
        for line in qrel_lines:
            qid, docid, grade = line.rstrip("\n").split("\t")
            if int(grade) > 0:
                qrels[qid].add(docid)
                qrel_grades[qid][docid] = int(grade)
                all_positive.add(docid)
    query_texts = {r["_id"]: r["text"] for r in map(json.loads, QUERY_FILE.open())}
    candidates_by_qid = {}
    for qid in QIDS:
        pos = qrels[qid]
        if not pos:
            raise ValueError(f"{qid} has no positive qrel")
        packet = packet_rows[qid]
        id_by_label = {}
        for label in ("full", "partial", "none"):
            for cid in labels[qid][label]:
                idx = int(cid[1:]) - 1
                if idx < 0 or idx >= len(packet["candidates"]):
                    raise ValueError(f"invalid candidate position {qid} {cid}")
                id_by_label[cid] = packet["candidates"][idx]["docid"]
        none_ids = [id_by_label[x] for x in labels[qid]["none"]]
        if len(set(none_ids)) != len(none_ids):
            raise ValueError(f"duplicate candidate IDs for {qid}")
        if set(none_ids) & all_positive:
            raise ValueError(f"qrel-positive passage labelled none for {qid}")
        extra_ids = []
        if qid == "test1591":
            for row in map(json.loads, SUPPLEMENTAL.open()):
                if row["qid"] == qid and row["label"] == "none":
                    if row["docid"] in all_positive:
                        raise ValueError("supplemental qrel-positive document")
                    extra_ids.append(row["docid"])
        none_ids += extra_ids
        if len(set(none_ids)) < N_HARD:
            raise ValueError(f"{qid} has only {len(set(none_ids))} reviewed none passages")
        candidates_by_qid[qid] = sorted(set(none_ids))
    # All question qrel positives are included; random filler is non-qrel.
    nonpositive_ids = []
    with CORPUS_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            did = row["_id"]
            if did not in all_positive:
                nonpositive_ids.append(did)
    manifests = {}
    hard_ids = {}
    for qid in QIDS:
        candidate_ids = set(candidates_by_qid[qid])
        for seed in SEEDS:
            rng = random.Random(seed * 1_000_003 + int(qid.replace("test", "")))
            ordinary_fill = rng.sample(nonpositive_ids, N_DOCS - len(qrels[qid]))
            ordinary = set(ordinary_fill) | qrels[qid]
            if len(ordinary) != N_DOCS:
                raise ValueError(f"ordinary size mismatch for {qid} seed {seed}")
            manifests[(qid, seed, "ordinary")] = ordinary
        union_ordinary = set().union(*(manifests[(qid, s, "ordinary")] for s in SEEDS))
        # The hard set is fixed across the three paired draws and absent from
        # all ordinary draws, ensuring a true 20-document treatment change.
        eligible = sorted(candidate_ids - union_ordinary)
        if len(eligible) < N_HARD:
            raise ValueError(f"{qid}: only {len(eligible)} reviewed none IDs outside ordinary draws")
        hard_rng = random.Random(20261030 + int(qid.replace("test", "")))
        picked = hard_rng.sample(eligible, N_HARD)
        hard_ids[qid] = picked
        for seed in SEEDS:
            ordinary = manifests[(qid, seed, "ordinary")]
            drop_pool = sorted(ordinary - qrels[qid])
            drop_rng = random.Random(20262000 + seed * 1_000_003 + int(qid.replace("test", "")))
            removed = set(drop_rng.sample(drop_pool, N_HARD))
            hard = (ordinary - removed) | set(picked)
            if len(hard) != N_DOCS or not qrels[qid] <= hard:
                raise ValueError(f"hard corpus size/qrel error for {qid} seed {seed}")
            if len(hard & set(picked)) != N_HARD:
                raise ValueError(f"hard passage overlap error for {qid} seed {seed}")
            manifests[(qid, seed, "hard")] = hard
    return packet_rows, labels, qrels, qrel_grades, query_texts, manifests, hard_ids, all_positive


def store_required_docs(manifests):
    needed = set().union(*manifests.values())
    db_path = ROOT / "hard_arm_docs.sqlite"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE docs (docid TEXT PRIMARY KEY, title TEXT, body TEXT)")
    found = 0
    with CORPUS_FILE.open() as f:
        batch = []
        for line in f:
            row = json.loads(line)
            if row["_id"] in needed:
                batch.append((row["_id"], row.get("title", "") or "", row.get("text", "") or ""))
                found += 1
                if len(batch) >= 2000:
                    con.executemany("INSERT INTO docs VALUES (?,?,?)", batch)
                    con.commit()
                    batch.clear()
        if batch:
            con.executemany("INSERT INTO docs VALUES (?,?,?)", batch)
            con.commit()
    db_count = con.execute("SELECT count(*) FROM docs").fetchone()[0]
    con.close()
    if db_count != len(needed) or found != len(needed):
        raise ValueError(f"document cache coverage mismatch: {db_count}/{len(needed)}")
    return db_path, len(needed)


def save_id_manifests(manifests, hard_ids):
    with ID_MANIFEST.open("w") as f:
        for (qid, seed, condition), ids in sorted(manifests.items()):
            row = {"qid": qid, "seed": seed, "condition": condition,
                   "docids": sorted(ids),
                   "hard_docids": hard_ids[qid] if condition == "hard" else []}
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def make_schema():
    b = tantivy.SchemaBuilder()
    b.add_text_field("docid", stored=True, tokenizer_name="raw")
    b.add_text_field("title", stored=False, index_option="freq")
    b.add_text_field("body", stored=False, index_option="freq")
    return b.build()


def retrieve(query, qrel_ids, qrel_grades, ids, con, temp_root):
    build_started = time.perf_counter()
    schema = make_schema()
    index_path = Path(temp_root) / "index"
    index_path.mkdir(parents=True)
    index = tantivy.Index(schema, path=str(index_path), reuse=False)
    writer = index.writer(heap_size=256_000_000)
    cur = con.cursor()
    for did in ids:
        row = cur.execute("SELECT title,body FROM docs WHERE docid=?", (did,)).fetchone()
        if row is None:
            raise ValueError(f"document missing from cache: {did}")
        writer.add_document(tantivy.Document(docid=did, title=row[0], body=row[1]))
    writer.commit()
    index.reload()
    searcher = index.searcher()
    build_seconds = time.perf_counter() - build_started
    words = list(dict.fromkeys(w.lower() for w in TOKEN_RE.findall(query)))
    parsed = " OR ".join('"' + w.replace('"', '\\"') + '"' for w in words)
    started = time.perf_counter()
    parsed_query = index.parse_query(parsed, ["title", "body"], field_boosts={"title": 2.0, "body": 1.0})
    result = searcher.search(parsed_query, limit=100, count=False)
    elapsed = time.perf_counter() - started
    hits = []
    for score, address in result.hits:
        did = searcher.doc(address)["docid"][0]
        hits.append({"docid": did, "score": score, "relevant": did in qrel_ids})
    hits.sort(key=lambda h: (-h["score"], h["docid"]))
    ranks = [i + 1 for i, h in enumerate(hits) if h["relevant"]]
    dcg = sum((2 ** qrel_grades[h["docid"]] - 1) / np.log2(i + 2)
              for i, h in enumerate(hits[:10]) if h["relevant"])
    ideal_grades = sorted(qrel_grades.values(), reverse=True)[:10]
    idcg = sum((2 ** grade - 1) / np.log2(i + 2) for i, grade in enumerate(ideal_grades))
    ndcg = dcg / idcg if idcg else 0.0
    index_bytes = sum(x.stat().st_size for x in index_path.rglob("*") if x.is_file())
    del searcher, parsed_query, index, writer
    gc.collect()
    return hits, ranks, elapsed, build_seconds, index_bytes, ndcg


def main():
    if OUT.exists() or SUMMARY.exists():
        raise RuntimeError("Hard-arm results already exist; refusing to overwrite")
    packet_rows, labels, qrels, qrel_grades, query_texts, manifests, hard_ids, all_positive = load_inputs()
    if not ID_MANIFEST.exists():
        save_id_manifests(manifests, hard_ids)
    manifest = {
        "sample_qids": QIDS, "sample_n": len(QIDS), "corpus_size": N_DOCS,
        "hard_passages_per_question": N_HARD, "seeds": SEEDS,
        "retriever": "Tantivy 0.25.1 default BM25; title boost 2, body boost 1; OR unique lowercase alphanumeric terms",
        "tie_break": "score descending, passage ID ascending", "bootstrap_resamples": B,
        "bootstrap_seed": BOOT_SEED, "reviewer": "AI-assisted adjudication by ChatGPT; not independent human annotation",
        "hard_ids": hard_ids,
        "review_counts": {q: {k: len(labels[q][k]) for k in ("full", "partial", "none")} for q in QIDS},
        "supplemental_none": 6, "build_time_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    db_path = ROOT / "hard_arm_docs.sqlite"
    if db_path.exists():
        with sqlite3.connect(db_path) as cached:
            n_docs = cached.execute("SELECT count(*) FROM docs").fetchone()[0]
    else:
        db_path, n_docs = store_required_docs(manifests)
    manifest["cached_document_count"] = n_docs
    manifest["document_cache_sha256"] = sha256(db_path)
    manifest["id_manifests_sha256"] = sha256(ID_MANIFEST)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    con = sqlite3.connect(db_path)
    raw = []
    build_started = time.time()
    for qindex, qid in enumerate(QIDS, 1):
        for seed in SEEDS:
            for condition in ("ordinary", "hard"):
                ids = sorted(manifests[(qid, seed, condition)])
                with tempfile.TemporaryDirectory(prefix="hard-tantivy-") as tmp:
                    hits, ranks, latency, build_seconds, index_bytes, ndcg = retrieve(
                        query_texts[qid], qrels[qid], qrel_grades[qid], ids, con, tmp)
                row = {
                    "qid": qid, "seed": seed, "condition": condition,
                    "size": len(ids), "gold_count": len(qrels[qid]),
                    "success_at_10": int(any(h["relevant"] for h in hits[:10])),
                    "recall_at_10": sum(h["relevant"] for h in hits[:10]) / len(qrels[qid]),
                    "mrr_at_10": 1 / min(ranks) if ranks else 0.0,
                    "ndcg_at_10": ndcg,
                    "first_relevant_rank": min(ranks) if ranks else None,
                    "latency_seconds": latency, "build_seconds": build_seconds,
                    "index_bytes": index_bytes, "top100": hits,
                    "hard_docids": hard_ids[qid] if condition == "hard" else [],
                }
                raw.append(row)
        if qindex % 10 == 0:
            print(f"completed {qindex}/{len(QIDS)} questions; elapsed={time.time()-build_started:.1f}s", flush=True)
            with OUT.open("w") as f:
                for row in raw:
                    f.write(json.dumps(row) + "\n")
    con.close()
    with OUT.open("w") as f:
        for row in raw:
            f.write(json.dumps(row) + "\n")
    # Analyze paired question-draw differences, averaging draw differences per question.
    grouped = {(r["qid"], r["seed"], r["condition"]): r for r in raw}
    if len(grouped) != len(QIDS) * len(SEEDS) * 2:
        raise ValueError("unexpected duplicate/missing hard-arm result rows")
    question_diffs = []
    draw_summaries = {}
    for metric in ("success_at_10", "recall_at_10", "mrr_at_10", "ndcg_at_10"):
        diffs_by_q = []
        for qid in QIDS:
            diffs_by_q.append(np.mean([grouped[(qid,s,"hard")][metric]-grouped[(qid,s,"ordinary")][metric] for s in SEEDS]))
        rng = np.random.default_rng(BOOT_SEED + (0 if metric == "success_at_10" else 1 if metric == "recall_at_10" else 2))
        vals = np.asarray(diffs_by_q)
        boots = vals[rng.integers(0, len(vals), size=(B, len(vals)))].mean(axis=1)
        lo, hi = np.quantile(boots, [0.025,0.975])
        draw_summaries[metric] = {
            "ordinary_mean": float(np.mean([r[metric] for r in raw if r["condition"]=="ordinary"])),
            "hard_mean": float(np.mean([r[metric] for r in raw if r["condition"]=="hard"])),
            "difference_hard_minus_ordinary": float(vals.mean()),
            "paired_question_bootstrap_95_ci": [float(lo),float(hi)],
            "draws": {str(s): float(np.mean([grouped[(q,s,"hard")][metric]-grouped[(q,s,"ordinary")][metric] for q in QIDS])) for s in SEEDS},
        }
    for seed in SEEDS:
        draw_summaries.setdefault("success_at_10_by_draw", {})[str(seed)] = {
            "ordinary": float(np.mean([grouped[(q,seed,"ordinary")]["success_at_10"] for q in QIDS])),
            "hard": float(np.mean([grouped[(q,seed,"hard")]["success_at_10"] for q in QIDS])),
            "difference": float(np.mean([grouped[(q,seed,"hard")]["success_at_10"]-grouped[(q,seed,"ordinary")]["success_at_10"] for q in QIDS])),
        }
    summary = {
        "n_questions": len(QIDS), "n_question_draw_condition_rows": len(raw),
        "n_questions_success_retrieved_ordinary": sum(any(grouped[(q,s,"ordinary")]["success_at_10"] for s in SEEDS) for q in QIDS),
        "paired_metrics": draw_summaries,
        "median_query_latency_ms": {cond: float(np.median([r["latency_seconds"] for r in raw if r["condition"]==cond])*1000) for cond in ("ordinary","hard")},
        "median_index_build_seconds": {cond: float(np.median([r["build_seconds"] for r in raw if r["condition"]==cond])) for cond in ("ordinary","hard")},
        "median_index_build_minutes": {cond: float(np.median([r["build_seconds"] for r in raw if r["condition"]==cond])/60) for cond in ("ordinary","hard")},
        "median_index_bytes": {cond: float(np.median([r["index_bytes"] for r in raw if r["condition"]==cond])) for cond in ("ordinary","hard")},
        "limitations": ["AI-assisted candidate screening, not independent human annotation", "BEIR qrels are positive-only", "100-question secondary subset", "retrieval-only; no generator evaluation"],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    manifest["run_elapsed_seconds"] = time.time()-build_started
    manifest["results_sha256"] = sha256(OUT)
    manifest["summary_sha256"] = sha256(SUMMARY)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    db_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
