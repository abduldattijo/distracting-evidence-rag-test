"""Screening pilot: does any of the 50 closest non-gold papers answer the question?

Paid step (Claude Haiku 5.5, thinking off). One call per question, 50 candidate
abstracts per call, plus the gold abstract hidden among them as a positive control.
The spend cap comes from MAX_USD (the harness in agentic-rag-eval/llm.py stops at it).
Candidates are the top 50 by similarity to the gold abstract, using the pinned MiniLM
model, the same order the retrieval runs use.

Output: screen_pilot.jsonl (one row per candidate) and screen_pilot_calls.jsonl (one row per call).
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "agentic-rag-eval"))
import llm  # noqa: E402
import run_pilot as rp  # noqa: E402

MODEL = "claude-haiku-5-5"
SYSTEM = (
    "You check whether scientific abstracts answer a question. An abstract SUPPORTS the "
    "question only if a reader could answer the question from that abstract alone, using "
    "evidence it states. Being on the same topic is not enough. Reply with JSON only: an "
    "object mapping each candidate id to \"yes\", \"no\" or \"unsure\". Use \"unsure\" when "
    "the abstract might partly answer it."
)


def build_prompt(question: str, items: list[tuple[str, dict]]) -> str:
    parts = [f"Question: {question}", "", "Candidates:"]
    for cid, p in items:
        parts.append(f"[{cid}] {p['title']}. {p['abstract']}")
    parts.append("")
    parts.append("Return the JSON object now.")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    papers, questions = rp.load_data()
    questions = questions[: args.questions]
    paper_idx = {p["id"]: i for i, p in enumerate(papers)}
    model = rp.load_sentence_transformer(rp.DEFAULT_SELECTION_MODEL, rp.DEFAULT_SELECTION_REVISION)
    orders = rp.selection_orders(papers, questions, paper_idx, model, "gold")
    del model

    rows, calls = [], []
    for q in questions:
        cand = orders[q["qid"]][: args.top]
        gold = paper_idx[q["gold_id"]]
        entries = [(pi, rank + 1) for rank, pi in enumerate(cand)] + [(gold, 0)]  # rank 0 = control
        random.Random(q["qid"]).shuffle(entries)
        labelled = [(f"c{i + 1:02d}", papers[pi]) for i, (pi, _) in enumerate(entries)]
        text, usage = llm.complete(MODEL, SYSTEM, build_prompt(q["question"], labelled), max_tokens=1500)
        verdicts = llm.extract_json(text) or {}
        calls.append({"qid": q["qid"], "n_candidates": len(entries), "parsed": bool(verdicts), **usage})
        for (cid, p), (pi, rank) in zip(labelled, entries):
            rows.append({"qid": q["qid"], "paper_id": p["id"], "gold_rank": rank,
                         "is_control": rank == 0, "verdict": verdicts.get(cid, "missing")})
        print(f"{q['qid']}: parsed={bool(verdicts)} usd={usage['usd']:.4f} total={llm.LEDGER.usd:.4f}", flush=True)

    with (HERE / "screen_pilot.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with (HERE / "screen_pilot_calls.jsonl").open("w") as f:
        f.writelines(json.dumps(c) + "\n" for c in calls)
    print(f"calls={llm.LEDGER.calls} total_usd={llm.LEDGER.usd:.4f}")


if __name__ == "__main__":
    main()
