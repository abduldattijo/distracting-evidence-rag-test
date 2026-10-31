#!/usr/bin/env python3
"""Freeze eligible same-page hard-negative query IDs and extract candidate passages."""
import hashlib
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "nq"
CONFIRMATORY = set((ROOT / "confirmatory_qids.txt").read_text().split())
SEED = 20261031
N_QUERIES = 100
MIN_CANDIDATES = 20


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-questions", type=int, default=N_QUERIES)
    parser.add_argument("--suffix", default="", help="output filename suffix")
    parser.add_argument("--exclude-qids", type=Path, help="newline-delimited qids excluded before sampling")
    parser.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES,
                        help="minimum distinct same-page passages required before sampling")
    args = parser.parse_args()
    qrels = defaultdict(set)
    all_positive = set()
    with (DATA / "qrels" / "test.tsv").open() as f:
        next(f)
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels[qid].add(docid)
                all_positive.add(docid)
    queries = {}
    with (DATA / "queries.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in CONFIRMATORY:
                queries[row["_id"]] = row["text"]

    gold_ids = set().union(*(qrels[qid] for qid in CONFIRMATORY))
    title_for_gold = {}
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in gold_ids:
                title_for_gold[row["_id"]] = row.get("title", "") or ""
    if set(title_for_gold) != gold_ids:
        raise ValueError(f"missing title for {len(gold_ids-set(title_for_gold))} gold documents")
    qtitles = {qid: {title_for_gold[did] for did in qrels[qid]} for qid in CONFIRMATORY}
    title_to_qids = defaultdict(set)
    for qid, titles in qtitles.items():
        for title in titles:
            title_to_qids[title].add(qid)

    pools = defaultdict(list)
    with (DATA / "corpus.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            docid = row["_id"]
            title = row.get("title", "") or ""
            if docid in all_positive or title not in title_to_qids:
                continue
            candidate = {"docid": docid, "title": title, "text": row.get("text", "") or ""}
            for qid in title_to_qids[title]:
                pools[qid].append(candidate)

    excluded_qids = set(args.exclude_qids.read_text().split()) if args.exclude_qids else set()
    eligible = sorted(
        qid for qid in CONFIRMATORY
        if qid not in excluded_qids
        if len({row["text"].strip() for row in pools[qid] if row["text"].strip()}) >= args.min_candidates
    )
    if len(eligible) < args.n_questions:
        selected = eligible
    else:
        selected = sorted(random.Random(args.seed).sample(eligible, args.n_questions))
    suffix = f"_{args.suffix}" if args.suffix else ""
    out = ROOT / f"hard_candidate_pools{suffix}.jsonl"
    with out.open("w") as f:
        for qid in selected:
            dedup = {}
            for row in pools[qid]:
                key = row["text"].strip()
                if key and key not in dedup:
                    dedup[key] = row
            candidates = list(dedup.values())
            f.write(json.dumps({"qid": qid, "query": queries[qid],
                                "gold_titles": sorted(qtitles[qid]),
                                "gold_docids": sorted(qrels[qid]),
                                "candidate_count": len(candidates),
                                "candidates": candidates}, ensure_ascii=False) + "\n")
    manifest = {
        "seed": args.seed, "requested_questions": args.n_questions,
        "minimum_candidates_before_dedup": args.min_candidates,
        "eligible_questions": len(eligible), "eligible_qids_sha256": hashlib.sha256(
            ("\n".join(eligible)+"\n").encode()).hexdigest(),
        "selected_questions": len(selected), "selected_qids": selected,
        "candidate_file_sha256": sha256(out),
        "candidate_count_selected": {qid: len({r['text'].strip() for r in pools[qid] if r['text'].strip()})
                                     for qid in selected},
        "selection_rule": "uniform random sample without replacement from sorted eligible confirmatory QIDs",
        "excluded_qids_before_sampling": sorted(excluded_qids),
        "all_positive_qrel_ids_excluded_from_candidates": True,
    }
    manifest_path = ROOT / f"hard_arm_manifest{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"eligible": len(eligible), "selected": len(selected),
                      "candidate_count_min": min(manifest["candidate_count_selected"].values()),
                      "candidate_count_median": sorted(manifest["candidate_count_selected"].values())[len(selected)//2],
                      "candidate_count_max": max(manifest["candidate_count_selected"].values()),
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
