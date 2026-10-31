"""Full screen of distractor candidates, in separate stages. Nothing here runs by default.

  preview   free. Counts calls and prices each stage from the pilot's measured token use.
  req       paid. Haiku writes 2 or 3 concise requirements per question from the question text alone.
  gate      paid. Haiku and Sonnet each say which requirements the gold abstract supports. DIAGNOSTIC ONLY:
            it writes requirements_gate.json and never edits requirements. The author reads every
            disagreement and every drop, then saves requirements_frozen.json by hand.
  judge     paid. Needs requirements_frozen.json. Haiku labels the 50 closest non-gold papers from the
            frozen manifest plus an anonymized, shuffled gold control as full / partial / none.
  review    paid. Sonnet re-labels the flagged papers and the control (not a none sample: the none audit
            is done by hand with a seeded sample).

Every paid stage takes --budget (US dollars). Before each call the script estimates a conservative maximum
cost and skips the call if it would pass the remaining budget. Every prompt and raw reply is saved, with
stop reasons. Empty or unparsed replies are retried; a refusal is recorded and the question is left for
manual screening. Candidates come from manifest_<order>.jsonl (float64 scores). Outputs go to full_screen/.
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "full_screen"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "agentic-rag-eval"))

PILOT_RAW = HERE / "screen_v3_raw.jsonl"
GATE_RAW = HERE / "screen_v3_gate_sonnet_raw.jsonl"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
REFUSED_BY_HAND = {"q001"}   # model refuses it; screened by hand (MANUAL_ADJUDICATION_2026-10-29.md)


def load_inputs():
    import run_pilot as rp
    papers, questions = rp.load_data()
    by_id = {p["id"]: p for p in papers}
    manifests = {}
    for mode in ("gold", "question"):
        manifests[mode] = {json.loads(l)["qid"]: json.loads(l)["ranked_ids"]
                           for l in (HERE / f"manifest_{mode}.jsonl").open()}
    return papers, by_id, questions, manifests


def pilot_means():
    """Mean input tokens, output tokens and cost per call type, from the saved pilot logs."""
    tot = {}
    for path in (PILOT_RAW, GATE_RAW):
        for line in path.open():
            r = json.loads(line)
            if "usage" not in r:
                continue
            kind = r["tag"].split(":")[0]
            t = tot.setdefault(kind, [0, 0, 0, 0.0])
            t[0] += 1; t[1] += r["usage"]["in"]; t[2] += r["usage"]["out"]; t[3] += r["usage"]["usd"]
    return {k: {"calls": v[0], "in": v[1] / v[0], "out": v[2] / v[0], "usd": v[3] / v[0]} for k, v in tot.items()}


def review_cost_model():
    """Fit input tokens of a pilot review call as base + per_item * items, using the saved prompts."""
    xs, ys = [], []
    for line in PILOT_RAW.open():
        r = json.loads(line)
        if r["tag"].startswith("review:") and "usage" in r:
            xs.append(sum(1 for ln in r["user"].splitlines() if ln.startswith("[r")))
            ys.append(r["usage"]["in"])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope, n


def stage_costs(n_q, orders, flag_rate, m, base, per_item, len_margin=1.0):
    """Expected cost of every stage. Review input grows with the number of flagged papers per call."""
    calls = (n_q - 1) * len(orders)
    flagged_per_call = 50 * flag_rate
    items = flagged_per_call + 1                           # flagged papers + the control
    sonnet_in = (base + per_item * items) * len_margin
    sonnet_out = (10 + 9 * items) * len_margin             # mean pilot reply: 33 tokens for about 4 items
    review = calls * (sonnet_in * 2.00 + sonnet_out * 10.00) / 1e6
    judge = calls * m["judge"]["usd"] * len_margin
    haiku_retry = 0.10 * judge
    fixed = (n_q - 1) * (m["req"]["usd"] + m["gate"]["usd"] + m["gate_sonnet"]["usd"]) * len_margin
    return {"req + gates": fixed, "judge (Haiku)": judge, "review (Sonnet)": review,
            "retries": haiku_retry}


def preview(args):
    papers, by_id, questions, manifests = load_inputs()
    n = len(questions)
    m = pilot_means()
    orders = args.orders
    base, per_item, n_fit = review_cost_model()
    print(f"questions: {n}   orders: {orders}")
    print(f"review call input tokens fitted on {n_fit} pilot calls: {base:.0f} + {per_item:.0f} per abstract\n")
    caps = [5.0, 6.0, 7.0]
    scenarios = [("low", 0.015, 1.0), ("expected", 0.024, 1.0), ("high", 0.05, 1.25), ("stress", 0.10, 1.25)]
    print("scenario    flag rate  length margin   req+gates   judge   review   retries    TOTAL   fits cap of 5 / 6 / 7")
    for name, rate, lm in scenarios:
        c = stage_costs(n, orders, rate, m, base, per_item, lm)
        tot = sum(c.values())
        fits = " / ".join("yes" if tot <= cap else "NO " for cap in caps)
        print(f"{name:10s}  {rate*100:5.1f}%     x{lm:<4}        {c['req + gates']:6.2f}   {c['judge (Haiku)']:6.2f}  {c['review (Sonnet)']:6.2f}   {c['retries']:6.2f}   {tot:6.2f}    {fits}")
    print("\nThe length margin multiplies every token estimate; 'high' and 'stress' use 1.25.")
    print("Sonnet review is priced per call as base + per-abstract tokens times flagged papers plus the control.\n")
    both = len(orders) == 2
    g, qm = manifests["gold"]["q001"][:50], manifests["question"]["q001"][:50]
    new_for_q = [i for i in qm if i not in set(g)]
    print(f"q001 by hand: gold list 50 papers (done, 4 partial); question list 50 papers, {len(set(g) & set(qm))} shared,"
          f" so {len(new_for_q)} still to read" if both else "q001 by hand: one list")
    print("Other hand work, unpriced: reading gate disagreements and drops, adjudicating every flag, seeded none audit.\n")
    print("Budget rule for the paid stages (to be built): before a stage starts, compute its conservative maximum;"
          " if it exceeds the remaining budget the stage does not start (--allow-partial is off by default).")
    print("If a call cannot be afforded mid-stage the run stops, writes full_screen/INCOMPLETE.json listing the"
          " unfinished questions and the spend, and no result file is treated as complete.")


# ---------------------------------------------------------------------------------------------
# Paid stages. Each stage is resumable (finished questions are skipped, so a rerun never pays twice)
# and stops with full_screen/INCOMPLETE.json if the next call cannot be afforded.
# ---------------------------------------------------------------------------------------------
class StageIncomplete(Exception):
    pass


class Guard:
    def __init__(self, stage, budget, llm_mod, config_mod):
        self.stage, self.budget, self.llm, self.config = stage, budget, llm_mod, config_mod
        self.done, self.pending = [], []
        # Spend from earlier runs of this stage (every attempt is in the raw files), so the cap is per stage, not per run.
        self.prior = sum(json.loads(l).get("usage", {}).get("usd", 0.0)
                         for f in OUT.glob(f"{stage}_*raw.jsonl") for l in f.open() if l.strip())

    def remaining(self):
        return self.budget - self.prior - self.llm.LEDGER.usd

    def max_cost(self, model, system, user, max_tokens):
        price = self.config.PRICES[model]
        return ((len(system) + len(user)) / 2.0 * price["in"] + max_tokens * price["out"]) / 1e6

    def call(self, raw, tag, model, system, user, max_tokens, parse, retries=2):
        """Parsed output, or None after retries. Raises StageIncomplete rather than skipping a call."""
        for attempt in range(retries + 1):
            if self.max_cost(model, system, user, max_tokens) > self.remaining():
                raise StageIncomplete(tag)
            text, usage = self.llm.complete(model, system, user, max_tokens=max_tokens)
            parsed = parse(text)
            raw.write(json.dumps({"tag": tag, "attempt": attempt, "model": model, "system": system, "user": user,
                                  "reply": text, "usage": usage, "parsed_ok": parsed is not None},
                                 ensure_ascii=False) + "\n")
            raw.flush()
            if parsed is not None:
                return parsed
            if usage.get("stop") == "refusal":
                return None            # a refusal is final: screened by hand, never retried
        return None

    def finish_incomplete(self, work_left):
        OUT.mkdir(exist_ok=True)
        status = {"stage": self.stage, "budget": self.budget, "spent": round(self.prior + self.llm.LEDGER.usd, 5),
                  "completed_this_run": self.done, "unfinished": work_left,
                  "note": "Stage did not finish within the budget. Results of this stage are INCOMPLETE."}
        (OUT / "INCOMPLETE.json").write_text(json.dumps(status, indent=1))
        return status


def done_ids(path):
    return {json.loads(l)["qid"] for l in path.open()} if path.exists() else set()


def stage_req(args, g, ctx):
    papers, by_id, questions, manifests = ctx
    import screen_v3 as sv
    path = OUT / "requirements_model.jsonl"
    pilot = json.load((OUT / "requirements_frozen_pilot.json").open()) if (OUT / "requirements_frozen_pilot.json").exists() else {}
    todo = [q for q in questions if q["qid"] not in done_ids(path) and q["qid"] not in REFUSED_BY_HAND and q["qid"] not in pilot]
    with path.open("a") as out, (OUT / "req_raw.jsonl").open("a") as raw:
        for i, q in enumerate(todo):
            try:
                reqs = g.call(raw, f"req:{q['qid']}", HAIKU, sv.REQ_SYSTEM, q["question"], 600, sv.parse_list, retries=0)
            except StageIncomplete:
                return [x["qid"] for x in todo[i:]]
            out.write(json.dumps({"qid": q["qid"], "requirements": reqs,
                                  "unscreened": None if reqs else "no requirements (see req_raw.jsonl)"}) + "\n")
            out.flush(); g.done.append(q["qid"])
    return []


def stage_gate(args, g, ctx):
    papers, by_id, questions, manifests = ctx
    import screen_v3 as sv
    import llm
    reqs = {json.loads(l)["qid"]: json.loads(l)["requirements"] for l in (OUT / "requirements_model.jsonl").open()}
    path = OUT / "requirements_gate.jsonl"
    gold = {q["qid"]: by_id[q["gold_id"]] for q in questions}
    complete = {json.loads(l)["qid"] for l in path.open()
                if json.loads(l)["haiku_supported"] is not None and json.loads(l)["sonnet_supported"] is not None} if path.exists() else set()
    todo = [q for q in questions if reqs.get(q["qid"]) and q["qid"] not in complete]
    with path.open("a") as out, (OUT / "gate_raw.jsonl").open("a") as raw:
        for i, q in enumerate(todo):
            r = reqs[q["qid"]]
            user = "Requirements:\n" + "\n".join(f"{n + 1}. {x}" for n, x in enumerate(r)) + \
                   f"\n\nAbstract: {gold[q['qid']]['title']}. {gold[q['qid']]['abstract']}\n\nReturn the JSON list now."

            def parse(text, n=len(r)):
                res = llm.extract_json(text)
                return res if isinstance(res, list) and all(isinstance(k, int) and 1 <= k <= n for k in res) else None
            try:
                h = g.call(raw, f"gate_haiku:{q['qid']}", HAIKU, sv.GATE_SYSTEM, user, 400, parse, retries=1)
                s = g.call(raw, f"gate_sonnet:{q['qid']}", SONNET, sv.GATE_SYSTEM, user, 400, parse, retries=1)
            except StageIncomplete:
                return [x["qid"] for x in todo[i:]]
            out.write(json.dumps({"qid": q["qid"], "original": r, "haiku_supported": h, "sonnet_supported": s}) + "\n")
            out.flush(); g.done.append(q["qid"])
    # Diagnostic list for the author: any requirement either model did not support, or a missing gate.
    latest = {}
    for l in path.open():
        row = json.loads(l); latest[row["qid"]] = row       # the latest row per question wins
    review = []
    for row in latest.values():
        n = len(row["original"]); allk = list(range(1, n + 1))
        if row["haiku_supported"] != allk or row["sonnet_supported"] != allk:
            review.append(row)
    (OUT / "gate_review_list.json").write_text(json.dumps(review, indent=1, ensure_ascii=False))
    return []


def layout_for(q, by_id, manifests, order, exclusions):
    cand = [i for i in manifests[order][q["qid"]] if i not in exclusions.get(q["qid"], set())][:50]
    entries = [(pid, rank + 1) for rank, pid in enumerate(cand)] + [(q["gold_id"], 0)]    # rank 0 = control
    random.Random(f"{q['qid']}:{order}").shuffle(entries)
    return [(f"c{i + 1:02d}", pid, rank) for i, (pid, rank) in enumerate(entries)]


def stage_judge(args, g, ctx):
    papers, by_id, questions, manifests = ctx
    import screen_v3 as sv
    frozen = json.load((OUT / "requirements_frozen.json").open())
    excl_path = OUT / "exclusions.json"
    exclusions = {k: set(v) for k, v in json.load(excl_path.open()).items()} if excl_path.exists() else {}
    for order in args.orders:
        path = OUT / f"judge_{order}.jsonl"
        todo = [q for q in questions if q["qid"] in frozen and q["qid"] not in done_ids(path)]
        with path.open("a") as out, (OUT / f"judge_{order}_raw.jsonl").open("a") as raw:
            for i, q in enumerate(todo):
                lay = layout_for(q, by_id, manifests, order, exclusions)
                items = [(cid, by_id[pid]) for cid, pid, _ in lay]
                try:
                    labels = g.call(raw, f"judge:{order}:{q['qid']}", HAIKU, sv.JUDGE_SYSTEM,
                                    sv.prompt(q["question"], frozen[q["qid"]], items), 1500,
                                    sv.make_parse_labels([c for c, _ in items]))
                except StageIncomplete:
                    return [f"{order}:{x['qid']}" for x in todo[i:]] + [f"{o}:*" for o in args.orders[args.orders.index(order) + 1:]]
                out.write(json.dumps({"qid": q["qid"], "order": order, "layout": lay, "labels": labels,
                                      "n_requirements": len(frozen[q["qid"]]),   # 1 means partial is impossible: report as supported / not
                                      "binary_only": len(frozen[q["qid"]]) == 1,
                                      "unscreened": None if labels else "no labels (see raw)"}) + "\n")
                out.flush(); g.done.append(f"{order}:{q['qid']}")
    return []


def stage_review(args, g, ctx):
    papers, by_id, questions, manifests = ctx
    import screen_v3 as sv
    frozen = json.load((OUT / "requirements_frozen.json").open())
    qmap = {q["qid"]: q for q in questions}
    for order in args.orders:
        rows = [json.loads(l) for l in (OUT / f"judge_{order}.jsonl").open()]
        path = OUT / f"review_{order}.jsonl"
        have = done_ids(path)

        def flagged(r):
            return [c for c, pid, rank in r["layout"] if rank != 0 and (r["labels"] or {}).get(c, "none") != "none"]
        todo = sorted((r for r in rows if r["labels"] and r["qid"] not in have),
                      key=lambda r: -len(flagged(r)))        # most flagged first
        with path.open("a") as out, (OUT / f"review_{order}_raw.jsonl").open("a") as raw:
            for i, r in enumerate(todo):
                chosen = set(flagged(r)) | {c for c, pid, rank in r["layout"] if rank == 0}
                ordered = sorted((t for t in r["layout"] if t[0] in chosen), key=lambda t: t[0])
                sub = [(f"r{n + 1:02d}", by_id[t[1]]) for n, t in enumerate(ordered)]
                mapping = {f"r{n + 1:02d}": t[1] for n, t in enumerate(ordered)}
                try:
                    res = g.call(raw, f"review:{order}:{r['qid']}", SONNET, sv.JUDGE_SYSTEM,
                                 sv.prompt(qmap[r["qid"]]["question"], frozen[r["qid"]], sub), 400,
                                 sv.make_parse_labels([c for c, _ in sub]), retries=1)
                except StageIncomplete:
                    return [f"{order}:{x['qid']}" for x in todo[i:]]
                out.write(json.dumps({"qid": r["qid"], "order": order,
                                      "sonnet": None if res is None else {mapping[k]: v for k, v in res.items()}}) + "\n")
                out.flush(); g.done.append(f"{order}:{r['qid']}")
    return []


STAGES = {"req": stage_req, "gate": stage_gate, "judge": stage_judge, "review": stage_review}


def run_stage(args, llm_mod=None, config_mod=None):
    if llm_mod is None:
        import llm as llm_mod
        import config as config_mod
    if args.budget is None:
        raise SystemExit("Paid stages need --budget (US dollars).")
    OUT.mkdir(exist_ok=True)
    ctx = load_inputs()
    g = Guard(args.stage, args.budget, llm_mod, config_mod)
    left = STAGES[args.stage](args, g, ctx)
    if left:
        status = g.finish_incomplete(left)
        print(f"INCOMPLETE: stage {args.stage} stopped before {left[0]} ({len(left)} unfinished), spent {status['spent']}")
        return 2
    marker = OUT / "INCOMPLETE.json"
    if marker.exists() and json.load(marker.open())["stage"] == args.stage:
        marker.unlink()
    print(f"stage {args.stage} finished, spent {llm_mod.LEDGER.usd:.4f} of {args.budget}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["preview", "req", "gate", "judge", "review"])
    ap.add_argument("--orders", nargs="+", default=["gold", "question"], choices=["gold", "question"])
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--flag-rate", type=float, default=0.024)
    args = ap.parse_args()
    if args.stage == "preview":
        return preview(args)
    sys.exit(run_stage(args))


if __name__ == "__main__":
    main()
