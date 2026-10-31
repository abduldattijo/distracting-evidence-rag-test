"""One place for every LLM call: counts tokens, prices them, and enforces the spend cap."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import config


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self) -> None:
        self.usd = 0.0
        self.calls = 0


LEDGER = Ledger()
_client = None


def _api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    raise RuntimeError("No Anthropic key: set the ANTHROPIC_API_KEY environment variable")


def _price(model: str) -> dict:
    p = config.PRICES.get(model) or {}
    if p.get("in") is None or p.get("out") is None:
        raise RuntimeError(f"Fill in config.PRICES['{model}'] (USD per million tokens) first. "
                           "The harness will not call an LLM without a price.")
    return p


def complete(model: str, system: str, user: str, max_tokens: int = 400):
    """Returns (text, usage) where usage = {in, out, usd, s}."""
    global _client

    price = _price(model)            # checked first: no price, no call
    if LEDGER.usd >= config.MAX_USD:
        raise BudgetExceeded(f"Spend cap reached: ${LEDGER.usd:.2f} >= ${config.MAX_USD:.2f}")
    import anthropic
    if _client is None:
        _client = anthropic.Anthropic(api_key=_api_key())

    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
    if model in config.THINKING:
        kwargs["thinking"] = config.THINKING[model]
    t0 = time.time()
    resp = _client.messages.create(**kwargs)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
    usd = (tin * price["in"] + tout * price["out"]) / 1e6
    LEDGER.usd += usd
    LEDGER.calls += 1
    return text, {"in": tin, "out": tout, "usd": usd, "s": time.time() - t0, "stop": resp.stop_reason}


def extract_json(text: str):
    """First JSON object or array in a model reply, or None."""
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
