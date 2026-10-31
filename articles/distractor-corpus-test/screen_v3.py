"""Screen v3 re-pilot: concise requirements, a gold-abstract gate, full / partial / none labels.

Paid, strict total budget (default 0.06 US dollars). One similarity order only (gold).
Steps, in order:
  1. Requirements. Haiku sees only the question text and lists its distinct answer
     requirements. Saved to screen_v2_requirements.json BEFORE any judging.
  2. Judge. One Haiku call per question: the 50 closest non-gold papers plus the gold
     abstract as an anonymized, shuffled control, labelled against the fixed requirements.
     Empty or unparsed replies are retried (up to 2 retries).
  3. Review. With whatever budget is left, one Sonnet call per question on the flagged
     papers, the control, and 2 randomly drawn "none" papers (seed = question id).
Before EVERY call the script estimates a conservative maximum cost (input at 3 characters
per token, output at max_tokens) and skips the call if it would exceed the remaining
budget. Every prompt and raw reply is saved to screen_v3_raw.jsonl.
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "agentic-rag-eval"))
import config  # noqa: E402
import llm  # noqa: E402
import run_pilot as rp  # noqa: E402

HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
LABELS = ("none", "partial", "full")

REQ_SYSTEM = (
    "You break a question into the distinct things a complete answer must contain. "
    "Reply with JSON only: a list of 2 or 3 strings, one per requirement. Each string is "
    "at most 15 words. Together they must cover every essential part of the question."
)
GATE_SYSTEM = (
    "You check which requirements an abstract states evidence for. Reply with JSON only: "
    "a list of the numbers of the requirements the abstract states evidence for."
)
JUDGE_SYSTEM = (
    "You check whether scientific abstracts answer a question, using fixed requirements. "
    "Label each candidate: \"full\" if the abstract states evidence for every requirement, "
    "\"partial\" if it states evidence for at least one requirement but not all, \"none\" if "
    "it states evidence for no requirement. Being on the same topic is not enough. Reply with "
    "JSON only: an object mapping each candidate id to \"full\", \"partial\" or \"none\"."
)


class Budget:
    def __init__(self, total: float):
        self.total, self.skipped = total, 0

    def remaining(self) -> float:
        return self.total - llm.LEDGER.usd

    def max_cost(self, model: str, system: str, user: str, max_tokens: int) -> float:
        price = config.PRICES[model]
        tin = (len(system) + len(user)) / 3.0
        return (tin * price["in"] + max_tokens * price["out"]) / 1e6

    def allows(self, model, system, user, max_tokens) -> bool:
        ok = self.max_cost(model, system, user, max_tokens) <= self.remaining()
        if not ok:
            self.skipped += 1
        return ok


def call(budget, raw, tag, model, system, user, max_tokens, parse, retries=2):
    """Returns parsed output or None. Saves every attempt."""
    for attempt in range(retries + 1):
        if not budget.allows(model, system, user, max_tokens):
            raw.write(json.dumps({"tag": tag, "attempt": attempt, "skipped_budget": True}) + "\n")
            return None
        text, usage = llm.complete(model, system, user, max_tokens=max_tokens)
        parsed = parse(text)
        raw.write(json.dumps({"tag": tag, "attempt": attempt, "model": model, "system": system,
                              "user": user, "reply": text, "usage": usage, "parsed_ok": parsed is not None},
                             ensure_ascii=False) + "\n")
        raw.flush()
        if parsed is not None:
            return parsed
    return None


def parse_list(text):
    out = llm.extract_json(text)
    return out if isinstance(out, list) and out and all(isinstance(x, str) for x in out) else None


def make_parse_labels(ids):
    def parse(text):
        out = llm.extract_json(text)
        if not isinstance(out, dict) or not all(i in out and out[i] in LABELS for i in ids):
            return None
        return out
    return parse


def prompt(question, requirements, items):
    lines = [f"Question: {question}", "", "Requirements:"]
    lines += [f"{n + 1}. {r}" for n, r in enumerate(requirements)]
    lines += ["", "Candidates:"]
    lines += [f"[{cid}] {p['title']}. {p['abstract']}" for cid, p in items]
    lines += ["", "Return the JSON object now."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--budget", type=float, default=0.06)
    args = ap.parse_args()
    budget = Budget(args.budget)

    papers, questions = rp.load_data()
    questions = questions[: args.questions]
    paper_idx = {p["id"]: i for i, p in enumerate(papers)}
    model = rp.load_sentence_transformer(rp.DEFAULT_SELECTION_MODEL, rp.DEFAULT_SELECTION_REVISION)
    orders = rp.selection_orders(papers, questions, paper_idx, model, "gold")
    del model

    raw = (HERE / "screen_v3_raw.jsonl").open("w", encoding="utf-8")

    # 1. Freeze requirements from the question text alone.
    reqs, unscreened = {}, {}
    for q in questions:
        r = call(budget, raw, f"req:{q['qid']}", HAIKU, REQ_SYSTEM, q["question"], 600, parse_list, retries=0)
        if r is None:
            # A model refusal (stop = refusal) or budget stop: record it, do not retry, screen it by hand later.
            unscreened[q["qid"]] = "no requirements returned (see screen_v3_raw.jsonl stop reason)"
        else:
            reqs[q["qid"]] = r
    # Gate: check the gold abstract alone. Requirements it does not support are dropped and recorded.
    dropped = {}
    for q in list(reqs):
        gold_p = papers[paper_idx[next(x for x in questions if x["qid"] == q)["gold_id"]]]
        user = "Requirements:\n" + "\n".join(f"{n + 1}. {r}" for n, r in enumerate(reqs[q])) + \
               f"\n\nAbstract: {gold_p['title']}. {gold_p['abstract']}\n\nReturn the JSON list now."
        def parse_nums(text, n=len(reqs[q])):
            out = llm.extract_json(text)
            return out if isinstance(out, list) and all(isinstance(i, int) and 1 <= i <= n for i in out) else None
        kept = call(budget, raw, f"gate:{q}", HAIKU, GATE_SYSTEM, user, 100, parse_nums, retries=1)
        if not kept:
            unscreened[q] = "gold abstract supports none of the requirements"
            del reqs[q]
            continue
        dropped[q] = [r for n, r in enumerate(reqs[q], 1) if n not in kept]
        reqs[q] = [r for n, r in enumerate(reqs[q], 1) if n in kept]
    (HERE / "screen_v3_requirements.json").write_text(json.dumps({"requirements": reqs, "dropped_by_gold_gate": dropped, "unscreened": unscreened}, indent=1, ensure_ascii=False))
    print("unscreened:", sorted(unscreened), flush=True)
    print(f"requirements frozen, spent {llm.LEDGER.usd:.4f}", flush=True)

    # 2. Judge with Haiku.
    labels, layouts = {}, {}
    for q in questions:
        cand = orders[q["qid"]][: args.top]
        gold = paper_idx[q["gold_id"]]
        entries = [(pi, rank + 1) for rank, pi in enumerate(cand)] + [(gold, 0)]  # rank 0 = control
        random.Random(q["qid"]).shuffle(entries)
        items = [(f"c{i + 1:02d}", papers[pi]) for i, (pi, _) in enumerate(entries)]
        layouts[q["qid"]] = [(cid, p["id"], rank) for (cid, p), (_, rank) in zip(items, entries)]
        ids = [cid for cid, _ in items]
        if q["qid"] in unscreened:
            labels[q["qid"]] = None
            continue
        out = call(budget, raw, f"judge:{q['qid']}", HAIKU, JUDGE_SYSTEM,
                   prompt(q["question"], reqs[q["qid"]], items), 1500, make_parse_labels(ids))
        labels[q["qid"]] = out
        print(f"{q['qid']}: judged={out is not None} spent={llm.LEDGER.usd:.4f}", flush=True)

    # 3. Sonnet review with the budget that is left (flags, control, 2 random "none").
    review = {}
    def n_flags(q):
        lab = labels[q["qid"]] or {}
        return sum(1 for t in layouts[q["qid"]] if lab.get(t[0], "none") != "none" and t[2] != 0)
    # Most flagged papers first, so the budget goes where review matters most.
    for q in sorted(questions, key=lambda q: -n_flags(q)):
        if labels[q["qid"]] is None:
            continue
        lay = layouts[q["qid"]]
        flagged = [t for t in lay if labels[q["qid"]][t[0]] != "none"]
        nones = [t for t in lay if labels[q["qid"]][t[0]] == "none" and t[2] != 0]
        chosen = {t[0] for t in flagged} | {t[0] for t in random.Random(q["qid"]).sample(nones, min(2, len(nones)))}
        chosen |= {t[0] for t in lay if t[2] == 0}
        pid = {t[1]: t for t in lay}
        sub = [(f"r{n + 1:02d}", papers[paper_idx[t[1]]]) for n, t in enumerate(sorted((t for t in lay if t[0] in chosen), key=lambda t: t[0]))]
        mapping = {f"r{n + 1:02d}": t for n, t in enumerate(sorted((t for t in lay if t[0] in chosen), key=lambda t: t[0]))}
        ids = [cid for cid, _ in sub]
        out = call(budget, raw, f"review:{q['qid']}", SONNET, JUDGE_SYSTEM,
                   prompt(q["question"], reqs[q["qid"]], sub), 300, make_parse_labels(ids), retries=0)
        if out is not None:
            review[q["qid"]] = {mapping[k][1]: out[k] for k in out}
        print(f"{q['qid']}: reviewed={out is not None} spent={llm.LEDGER.usd:.4f}", flush=True)
    raw.close()

    with (HERE / "screen_v3_labels.jsonl").open("w") as f:
        for q in questions:
            for cid, pid_, rank in layouts[q["qid"]]:
                f.write(json.dumps({"qid": q["qid"], "paper_id": pid_, "gold_rank": rank,
                                    "is_control": rank == 0,
                                    "haiku": (labels[q["qid"]] or {}).get(cid, "missing"),
                                    "sonnet": review.get(q["qid"], {}).get(pid_)}) + "\n")
    print(f"calls={llm.LEDGER.calls} spent={llm.LEDGER.usd:.4f} budget={args.budget} skipped_for_budget={budget.skipped}")


if __name__ == "__main__":
    main()
