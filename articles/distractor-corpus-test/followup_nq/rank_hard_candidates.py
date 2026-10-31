#!/usr/bin/env python3
"""Rank same-page candidates with frozen BGE and create shuffled blind packets."""
import hashlib
import argparse
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
POOLS = ROOT / "hard_candidate_pools.jsonl"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots" / MODEL_REVISION
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 50
BLIND_SEED = 20261032


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    suffix = f"_{args.suffix}" if args.suffix else ""
    pools_path = ROOT / f"hard_candidate_pools{suffix}.jsonl"
    pools = [json.loads(line) for line in pools_path.open()]
    model = SentenceTransformer(str(MODEL_PATH), device="cpu", local_files_only=True)
    model.max_seq_length = 512
    output_ranked = ROOT / f"hard_candidates_ranked_PRIVATE{suffix}.jsonl"
    output_packets = ROOT / f"hard_review_packets{suffix}.jsonl"
    if output_ranked.exists() or output_packets.exists():
        raise RuntimeError("Refusing to overwrite an existing candidate ranking or packet file")
    with output_ranked.open("w") as rank_out, output_packets.open("w") as packet_out:
        for pool in pools:
            candidates = pool["candidates"]
            docs = [((x["title"] + ". ") if x["title"] else "") + x["text"] for x in candidates]
            qvec = model.encode([QUERY_PREFIX + pool["query"]], normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)[0]
            dvecs = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True,
                                 batch_size=64, show_progress_bar=False)
            # Avoid Accelerate's warning-prone float32 matmul path on this macOS host.
            scores = np.sum(dvecs.astype(np.float64) * qvec.astype(np.float64), axis=1)
            if not np.isfinite(scores).all():
                raise ValueError(f"non-finite embedding score for {pool['qid']}")
            order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]["docid"]))
            ranked = [{**candidates[i], "similarity": float(scores[i]), "similarity_rank": rank + 1}
                      for rank, i in enumerate(order)]
            rank_out.write(json.dumps({"qid": pool["qid"], "query": pool["query"],
                                       "gold_titles": pool["gold_titles"],
                                       "gold_docids": pool["gold_docids"],
                                       "ranked_candidates": ranked}, ensure_ascii=False) + "\n")
            blind = ranked[:TOP_K]
            random.Random(BLIND_SEED + int(pool["qid"].replace("test", ""))).shuffle(blind)
            packet_out.write(json.dumps({"qid": pool["qid"], "query": pool["query"],
                                         "candidates": [{"docid": x["docid"], "title": x["title"],
                                                         "text": x["text"]} for x in blind]},
                                        ensure_ascii=False) + "\n")
            print(f"ranked {pool['qid']}: {len(candidates)} candidates; reviewed packet {len(blind)}", flush=True)
    metadata = {
        "model": "BAAI/bge-small-en-v1.5", "revision": MODEL_REVISION,
        "query_prefix": QUERY_PREFIX, "passage_text": "title + '. ' + passage text",
        "normalized_embeddings": True, "score": "float64 elementwise dot product of normalized vectors",
        "ranking": "cosine similarity descending; docid ascending for exact ties",
        "top_k_for_initial_review": TOP_K, "blind_shuffle_seed": BLIND_SEED,
        "candidate_pools_sha256": sha256(pools_path),
        "ranked_private_sha256": sha256(output_ranked),
        "blind_packets_sha256": sha256(output_packets),
    }
    manifest = ROOT / f"hard_embedding_manifest{suffix}.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
