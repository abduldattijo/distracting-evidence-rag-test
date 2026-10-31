"""Settings for the agentic-RAG evaluation. Everything a reader might want to change is here."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent

# MODE=pilot (default) is a small dry run to test the harness. MODE=full is the real experiment.
MODE = os.getenv("MODE", "pilot")
# each mode has its own data folder, so a full run never overwrites the pilot corpus or questions
DATA = ROOT / ("data" if MODE == "pilot" else f"data_{MODE}")
RESULTS = ROOT / "results"
SIZES = {
    "pilot": {"corpus": 300, "questions": 20, "seed": 11, "distractors": 0},
    # different seed: no pilot question is reused. Each source paper gets close same-topic papers
    # in the corpus (chosen before any retrieval is run) because the pilot was too easy: dense
    # retrieval scored 17 of 17 at rank 1.
    "full": {"corpus": 2000, "questions": 300, "seed": 7, "distractors": 5},
}
CORPUS_SIZE = SIZES[MODE]["corpus"]
N_QUESTIONS = SIZES[MODE]["questions"]
SEED = SIZES[MODE]["seed"]
DISTRACTORS_PER_SOURCE = SIZES[MODE]["distractors"]

# ---- corpus ---------------------------------------------------------------------------
ARXIV_CATEGORIES = ["cs.AI", "cs.CL", "cs.IR", "cs.LG"]
# Only papers submitted after the question/agent model's training cutoff, so the model
# cannot answer from memory. VERIFY the cutoff of the models below before the full run.
PUBLISHED_AFTER = "2026-07-01"
PUBLISHED_BEFORE = "2026-09-30"
WINDOW_DAYS = 1          # sample day by day so papers spread over the whole period

# ---- models ---------------------------------------------------------------------------
QUESTION_MODEL = os.getenv("QUESTION_MODEL", "claude-sonnet-5-5")   # writes the questions
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-haiku-5-5")          # query rewriting + agent
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                              # local, free
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"              # local, free
# These models return a 400 for any non-default temperature, so none is sent: outputs are not
# bit-for-bit repeatable. Thinking is turned to its lowest setting so billed output tokens and
# latency measure the retrieval design, not hidden reasoning (docs, checked 2026-10-15):
# Sonnet 5.5 rejects "disabled" and uses "between_tools"; Haiku 5.5 accepts "disabled".
THINKING = {
    "claude-sonnet-5-5": {"type": "between_tools"},
    "claude-haiku-5-5": {"type": "disabled"},
}

# USD per million tokens, copied from the provider's pricing page. The harness refuses to make
# LLM calls for a model whose price is None, so a run can never spend on a guessed price.
PRICES = {
    # Source: platform.claude.com models overview + claude.com/pricing, checked 2026-10-15.
    # Haiku's price is higher for prompts over 100K tokens; ours are far below that.
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00},
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50},
}
MAX_USD = float(os.getenv("MAX_USD", "2"))   # hard stop: the run aborts when spend reaches this

# ---- retrieval ------------------------------------------------------------------------
K_FINAL = 5            # recall@5 is the headline number
RANK_DEPTH = 10        # ranked ids kept per question (for MRR@10)
RERANK_CANDIDATES = 50
RRF_K = 60
MULTI_QUERY_N = 3      # rewrites in the fusion design (original query is always added)
AGENT_MAX_SEARCHES = 3 # includes the first search with the original question
