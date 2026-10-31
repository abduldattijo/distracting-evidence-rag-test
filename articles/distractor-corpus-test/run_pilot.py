"""Run the local retrieval-only distractor corpus pilot.

This script makes no API calls. It checks the frozen source data, drops the two
questions marked not ok in the reviewed file, builds paired nested corpora, and
evaluates BM25 and dense retrieval over flat chunks and parent papers.

Results are before screening: a distractor that also answers a question is not
yet removed, so it counts as a miss. Rows are marked stage = pilot_pre_screen.

Scope notes for reading the output:
  * latency_s is scoring and sorting only. Index building and query encoding are
    not included (query encoding is saved as query_encode_s).
  * index_payload_bytes is the size of this script's own index (a pickled
    postings table for BM25, the float32 vector matrix for embeddings). The two
    are not comparable storage measures.
  * "parent" and a flat run that keeps collecting chunks until it has 5 distinct
    papers give the same ranked papers by construction. They differ only in the
    text passed to a generator. So no separate cell is run for it, and the flat
    rows record how many distinct papers its 5 chunk slots covered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agentic-rag-eval" / "data_full"
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / "cache"
FROZEN_HASHES = {
    "questions.jsonl": "48752c4b3a76151f8420ac7b3c6527e07f0fbf41996b2adc2cd07de2df0d2cd1",
    "papers.jsonl": "1e4024754884782ce67c4e99b50e751b16fc848d116a29adc405e960cb2bf410",
    "questions_review.csv": "ecf29f70bd7d9145b405f33c143d221848d54004bae1b4a544559e2647cd7703",
}
DEFAULT_SELECTION_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SELECTION_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_RETRIEVAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RETRIEVAL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
TOKEN = re.compile(r"(?u)\b\w\w+\b")
TOP_K = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_chunks(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else [""]
    if len(sentences) == 1:
        return sentences
    # Two-sentence windows with one sentence of overlap.
    return [" ".join(sentences[i : i + 2]) for i in range(len(sentences) - 1)]


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) over pre-tokenized chunks, with an inverted index."""

    def __init__(self, chunk_tf: list[Counter], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(chunk_tf)
        self.lengths = [sum(tf.values()) for tf in chunk_tf]
        self.avgdl = sum(self.lengths) / max(1, self.n)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tf in enumerate(chunk_tf):
            for term, freq in tf.items():
                self.postings[term].append((i, freq))

    def payload_bytes(self) -> int:
        return len(pickle.dumps((dict(self.postings), self.lengths), protocol=pickle.HIGHEST_PROTOCOL))

    def rank(self, query_terms: set[str]) -> list[int]:
        scores = np.zeros(self.n, dtype=np.float64)
        for term in query_terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, freq in plist:
                denom = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avgdl, 1e-12))
                scores[i] += idf * freq * (self.k1 + 1) / denom
        return np.argsort(-scores, kind="stable").tolist()


def load_sentence_transformer(name: str, revision: str | None):
    from sentence_transformers import SentenceTransformer
    kwargs = {"device": "cpu"}
    if revision:
        kwargs["revision"] = revision
    return SentenceTransformer(name, **kwargs)


def encode(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                                   show_progress_bar=False), dtype=np.float32)


def load_data() -> tuple[list[dict], list[dict]]:
    for name, expected in FROZEN_HASHES.items():
        actual = sha256(SOURCE / name)
        if actual != expected:
            raise SystemExit(f"Frozen input hash mismatch for {name}: {actual}")
    papers = read_jsonl(SOURCE / "papers.jsonl")
    with (SOURCE / "questions_review.csv").open(encoding="utf-8", newline="") as f:
        ok = {row["qid"]: row["ok (1/0)"].strip() == "1" for row in csv.DictReader(f)}
    questions = sorted((q for q in read_jsonl(SOURCE / "questions.jsonl") if ok.get(q["qid"])),
                       key=lambda q: q["qid"])
    if len(papers) != 2000 or len(questions) != 298:
        raise SystemExit(f"Expected 2,000 papers and 298 reviewed questions; "
                         f"found {len(papers)} and {len(questions)}")
    return papers, questions


def selection_orders(papers, questions, paper_idx, model, mode: str) -> dict[str, list[int]]:
    paper_vecs = encode(model, [p["abstract"] for p in papers])
    if mode == "gold":
        texts = [papers[paper_idx[q["gold_id"]]]["abstract"] for q in questions]
    else:
        texts = [q["question"] for q in questions]
    q_vecs = encode(model, texts)
    orders = {}
    for qi, q in enumerate(questions):
        scores = paper_vecs @ q_vecs[qi]
        scores[paper_idx[q["gold_id"]]] = -np.inf  # the gold paper is never a distractor
        orders[q["qid"]] = np.argsort(-scores, kind="stable").tolist()[:-1]
    return orders


def make_corpus(order: list[int], gold_idx: int, size: int, condition: str, seed: int,
                skip: frozenset = frozenset()) -> list[int]:
    """`order` is the frozen original ranking. Skipped (excluded) papers are removed and the next ranked
    papers backfill, so a look-alike corpus keeps its exact size; the ordinary pool keeps original ranks."""
    if size < 2:
        raise ValueError("Corpus size must be at least 2")
    if condition == "look_alike":
        selected = [p for p in order if p not in skip][: size - 1]
        if len(selected) < size - 1:
            raise ValueError(f"Not enough papers left after exclusions for N={size}")
    else:
        pool = [p for p in order[199:] if p not in skip]
        if len(pool) < size - 1:
            raise ValueError(f"Not enough ordinary distractors for N={size}")
        # Nested draws: each larger size extends the same seeded permutation.
        selected = random.Random(seed).sample(pool, len(pool))[: size - 1]
    return [gold_idx, *selected]


def top_papers(unit_order: list[int], unit_paper: list[int], distinct: bool) -> list[int]:
    result, seen = [], set()
    for u in unit_order:
        p = unit_paper[u]
        if distinct and p in seen:
            continue
        result.append(p)
        seen.add(p)
        if len(result) == TOP_K:
            break
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1], help="ordinary condition draws; look-alike is deterministic and runs once")
    ap.add_argument("--selection-model", default=DEFAULT_SELECTION_MODEL)
    ap.add_argument("--selection-revision", default=DEFAULT_SELECTION_REVISION)
    ap.add_argument("--retrieval-model", default=DEFAULT_RETRIEVAL_MODEL)
    ap.add_argument("--retrieval-revision", default=DEFAULT_RETRIEVAL_REVISION)
    ap.add_argument("--selection-mode", choices=["gold", "question"], default="gold")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None, help="frozen ranked non-gold lists (float64); replaces recomputing the selection order")
    ap.add_argument("--exclusions", type=Path, default=None, help="json {qid: [paper ids]} removed from distractor pools, then backfilled")
    args = ap.parse_args()
    if args.output is None:
        args.output = OUT_DIR / f"pilot_{args.selection_mode}.jsonl"

    papers, all_questions = load_data()
    questions = all_questions[: args.questions]
    paper_idx = {p["id"]: i for i, p in enumerate(papers)}

    # Chunk every paper once. A corpus is then just a subset of chunk numbers.
    chunks, chunk_paper, paper_chunks = [], [], []
    for pi, p in enumerate(papers):
        ids = []
        for c in split_chunks(p["abstract"]):
            ids.append(len(chunks))
            chunks.append(c)
            chunk_paper.append(pi)
        paper_chunks.append(ids)
    chunk_tf = [Counter(tokens(c)) for c in chunks]

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"chunks_{args.retrieval_model.replace('/', '_')}_{args.retrieval_revision[:8]}.npy"
    retrieval_model = load_sentence_transformer(args.retrieval_model, args.retrieval_revision)
    if cache_file.exists():
        chunk_vecs = np.load(cache_file)
        if chunk_vecs.shape[0] != len(chunks):
            raise SystemExit("Cached chunk vectors do not match the chunk list; delete the cache file.")
    else:
        t0 = time.perf_counter()
        chunk_vecs = encode(retrieval_model, chunks, batch_size=128)
        np.save(cache_file, chunk_vecs)
        print(f"Embedded {len(chunks)} chunks in {time.perf_counter() - t0:.0f}s")
    query_vecs, query_times = {}, {}
    for q in questions:
        t0 = time.perf_counter()
        query_vecs[q["qid"]] = encode(retrieval_model, [q["question"]], batch_size=1)[0]
        query_times[q["qid"]] = time.perf_counter() - t0
    del retrieval_model

    if args.manifest:
        man = {r["qid"]: r for r in read_jsonl(args.manifest)}
        orders = {}
        for q in questions:
            if man[q["qid"]]["gold_id"] != q["gold_id"] or man[q["qid"]]["mode"] != args.selection_mode:
                raise SystemExit(f"Manifest does not match {q['qid']} or the selection mode")
            orders[q["qid"]] = [paper_idx[i] for i in man[q["qid"]]["ranked_ids"]]
    else:
        selection_model = load_sentence_transformer(args.selection_model, args.selection_revision)
        orders = selection_orders(papers, questions, paper_idx, selection_model, args.selection_mode)
        del selection_model
    excl = {}
    if args.exclusions:
        excl = {qid: frozenset(paper_idx[i] for i in ids) for qid, ids in json.load(args.exclusions.open()).items()}
    stage_label = "screened" if args.exclusions else "pilot_pre_screen"
    unavailable = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with args.output.open("w", encoding="utf-8") as out:
        for q in questions:
            gold_idx = paper_idx[q["gold_id"]]
            query_terms = set(tokens(q["question"]))
            for condition, seed in [("look_alike", None), *[("ordinary", sd) for sd in args.seeds]]:
                for size in args.sizes:
                    # Ordinary distractors come from papers ranked 200 or lower, so N tops out at 1,801.
                    if condition == "ordinary" and size - 1 > len(orders[q["qid"]]) - 199:
                        continue
                    try:
                        corpus = make_corpus(orders[q["qid"]], gold_idx, size, condition, seed or 0, excl.get(q["qid"], frozenset()))
                    except ValueError:
                        unavailable[(condition, size)] += 1   # reported, never padded with excluded papers
                        continue
                    ids = [c for pi in corpus for c in paper_chunks[pi]]
                    unit_paper = [chunk_paper[c] for c in ids]
                    bm25 = BM25([chunk_tf[c] for c in ids])
                    matrix = chunk_vecs[ids]
                    payloads = {"bm25": bm25.payload_bytes(), "embeddings": int(matrix.nbytes)}
                    for method in ("bm25", "embeddings"):
                        start = time.perf_counter()
                        if method == "bm25":
                            unit_order = bm25.rank(query_terms)
                        else:
                            unit_order = np.argsort(-(matrix @ query_vecs[q["qid"]]), kind="stable").tolist()
                        elapsed = time.perf_counter() - start
                        for representation in ("flat", "parent"):
                            ranked = top_papers(unit_order, unit_paper, distinct=(representation == "parent"))
                            positions = [i + 1 for i, pi in enumerate(ranked) if pi == gold_idx]
                            first = min(positions) if positions else None
                            slots = top_papers(unit_order, unit_paper, distinct=False)
                            row = {
                                "stage": stage_label, "n_excluded_for_question": len(excl.get(q["qid"], ())),
                                "qid": q["qid"], "gold_id": q["gold_id"],
                                "condition": condition, "selection_mode": args.selection_mode,
                                "corpus_size": size, "seed": seed,
                                "method": method, "representation": representation,
                                "evidence_recall_at_5": first is not None,
                                "first_supporting_rank": first,
                                "mrr_at_5": 1 / first if first else 0.0,
                                "distinct_papers_in_top5_chunk_slots": len(set(slots)),
                                "ranked_paper_ids": [papers[i]["id"] for i in ranked],
                                "latency_s": elapsed,
                                "query_encode_s": query_times[q["qid"]] if method == "embeddings" else None,
                                "n_indexed_chunks": len(ids),
                                "index_payload_bytes": payloads[method],
                                "selection_model": args.selection_model,
                                "selection_revision": args.selection_revision,
                                "retrieval_model": args.retrieval_model,
                                "retrieval_revision": args.retrieval_revision,
                            }
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n_rows += 1
                out.flush()
    print(f"Wrote {n_rows} retrieval-only pilot rows to {args.output}")
    if unavailable:
        print("Corpora that could not be built after exclusions (questions per condition and size):", dict(unavailable))


if __name__ == "__main__":
    main()
