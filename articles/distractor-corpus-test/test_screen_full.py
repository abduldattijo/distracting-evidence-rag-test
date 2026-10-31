"""Free test of the budget guard and resume logic with a fake model. Makes no API calls."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "agentic-rag-eval"))
import screen_full as sf


class Ledger: usd = 0.0
class FakeLLM:
    LEDGER = Ledger()
    calls = 0
    @staticmethod
    def complete(model, system, user, max_tokens=400):
        FakeLLM.calls += 1; FakeLLM.LEDGER.usd += 0.001
        return '["first requirement", "second requirement"]', {"in": 100, "out": 20, "usd": 0.001, "s": 0.0, "stop": "end_turn"}
    @staticmethod
    def extract_json(t): return json.loads(t)
class FakeConfig:
    PRICES = {"claude-haiku-5-5": {"in": 0.10, "out": 0.50}, "claude-sonnet-5-5": {"in": 2.0, "out": 10.0}}

class A: stage = "req"; orders = ["gold"]; budget = 0.0025
sf.OUT = Path(tempfile.mkdtemp())
(sf.OUT / "requirements_frozen_pilot.json").write_text("{}")
import screen_v3  # parse_list lives here
rc = sf.run_stage(A, FakeLLM, FakeConfig)
inc = json.load((sf.OUT / "INCOMPLETE.json").open())
rows = [json.loads(l) for l in (sf.OUT / "requirements_model.jsonl").open()]
print("first run: exit", rc, "| calls", FakeLLM.calls, "| rows written", len(rows), "| unfinished", len(inc["unfinished"]), "| INCOMPLETE stage", inc["stage"])
assert rc == 2 and len(rows) == FakeLLM.calls and len(inc["unfinished"]) > 0
FakeLLM.LEDGER.usd = 0.0; before = FakeLLM.calls; A.budget = 0.0025
rc = sf.run_stage(A, FakeLLM, FakeConfig)
rows2 = [json.loads(l) for l in (sf.OUT / "requirements_model.jsonl").open()]
print("rerun: exit", rc, "| new calls", FakeLLM.calls - before, "| rows now", len(rows2), "| duplicates", len(rows2) - len({r['qid'] for r in rows2}))
assert len(rows2) - len({r['qid'] for r in rows2}) == 0
print("guard and resume behave as designed")
