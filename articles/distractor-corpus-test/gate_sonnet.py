"""Second opinion on the gold gate. Paid, strict budget (default 0.03 US dollars).

For each screened question, Sonnet sees the ORIGINAL requirement list (the ones Haiku kept plus
the ones it dropped, from screen_v3_requirements.json) and the gold abstract alone, and lists
which requirements the abstract states evidence for. Nothing is deleted or changed: the output
(screen_v3_gate_sonnet.json) is compared with Haiku's gate and read by the author.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "agentic-rag-eval"))
import llm  # noqa: E402
import run_pilot as rp  # noqa: E402
import screen_v3 as sv  # noqa: E402

SONNET = "claude-sonnet-5-5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.03)
    args = ap.parse_args()
    budget = sv.Budget(args.budget)
    papers, questions = rp.load_data()
    by_id = {p["id"]: p for p in papers}
    gold = {q["qid"]: by_id[q["gold_id"]] for q in questions}
    R = json.load((HERE / "screen_v3_requirements.json").open())
    out, raw = {}, (HERE / "screen_v3_gate_sonnet_raw.jsonl").open("w")
    for qid, kept in R["requirements"].items():
        original = kept + R["dropped_by_gold_gate"].get(qid, [])
        user = "Requirements:\n" + "\n".join(f"{n + 1}. {r}" for n, r in enumerate(original)) + \
               f"\n\nAbstract: {gold[qid]['title']}. {gold[qid]['abstract']}\n\nReturn the JSON list now."

        def parse(text, n=len(original)):
            res = llm.extract_json(text)
            return res if isinstance(res, list) and all(isinstance(i, int) and 1 <= i <= n for i in res) else None

        res = sv.call(budget, raw, f"gate_sonnet:{qid}", SONNET, sv.GATE_SYSTEM, user, 100, parse, retries=1)
        out[qid] = {"original": original, "haiku_kept": kept,
                    "sonnet_supported": None if res is None else [original[i - 1] for i in res]}
        print(qid, "done", f"spent={llm.LEDGER.usd:.4f}", flush=True)
    raw.close()
    (HERE / "screen_v3_gate_sonnet.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"calls={llm.LEDGER.calls} spent={llm.LEDGER.usd:.4f} budget={args.budget} skipped={budget.skipped}")


if __name__ == "__main__":
    main()
