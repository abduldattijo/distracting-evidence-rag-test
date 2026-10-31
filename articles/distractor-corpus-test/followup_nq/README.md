# BEIR Natural Questions corpus-growth experiment

This folder contains the frozen BEIR Natural Questions data split, Tantivy BM25 indexes, confirmatory rankings, analysis, and figures for the primary corpus growth experiment. A supplementary same page stress test and a post hoc answer generation evaluation are also included.

## Main result

For 3,412 confirmatory questions, Success@10 averaged across three random nested corpora fell from 86.8% at 10,000 passages to 75.6% at 100,000, 64.8% at 500,000, and 58.1% at 1,000,000. Success@10 on the fixed full BEIR corpus was 46.6%. The first tested size meeting the predeclared five-point degradation rule was 100,000 passages. See `../FOLLOWUP_NQ_PROTOCOL.md` for definitions, caveats, and interpretation.

## Dataset and freeze

The BEIR `nq.zip` archive is downloaded from the [BEIR dataset server](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/). The archive MD5 is checked against BEIR's published value. Extract it into `data/nq/` so the folder contains `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. The raw corpus is about 1.4 GB locally and is not copied into the article or reproduction scripts. Check the source license and terms before redistributing it.

`nq_frozen.json` records the archive and extracted-file checksums. `pilot_qids.txt` and `confirmatory_qids.txt` are the frozen 40/3,412 split. Do not regenerate these IDs with a different seed for a reproduction of this run.

## Reproduction environment

The recorded machine was a MacBook Pro with Apple M4, 16 GB memory, macOS 27.0.1, Python 3.9.6, Tantivy Python package 0.25.1, NumPy 2.0.2, and Matplotlib 3.9.4. Search latency is machine- and implementation-specific.

Install the pinned direct dependencies in a Python 3.9 environment:

```sh
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With the BEIR data in `data/nq/`, run:

```sh
python freeze_nq.py
python build_confirmatory_indexes.py
python run_confirmatory_retrieval.py
python analyze_confirmatory.py
python plot_confirmatory.py
```

Index construction writes 12 nested sampled indexes and one full-corpus index under `confirmatory_indexes/`. It also writes the exact passage-ID membership manifest and per-file index hashes. The retrieval script refuses to overwrite its results file. Preserve the existing result files or move them aside before a fresh run.

## Main artifacts

- `confirmatory_indexes/id_manifests.json`: exact passage membership for all conditions.
- `confirmatory_indexes/index_metadata.json`: index row counts, sizes, and file hashes.
- `confirmatory_results.jsonl`: raw Tantivy rankings (top 100 per query-condition).
- `confirmatory_results_ranked.jsonl`: score-descending rankings with passage-ID ascending tie-break applied within the retrieved top 100.
- `confirmatory_summary.json`: per-condition metrics and question-bootstrap comparisons.
- `confirmatory_multiplicity_check.json`: Bonferroni-adjusted intervals for the four size comparisons.
- `figures/success_at_10_by_corpus_size.png` and `.svg`: primary figure.

The same page hard passage extension is complete as a secondary stress test. AI assisted screening was followed by the author’s manual review of the candidate passages and labels. This was a single author review, not independent double annotation. The final 100 question sample has 5,000 main candidate labels and six extra passages reviewed for one question. Totals in the main review were 81 full support, 269 partial support, and 4,650 with no support. Only passages manually labeled with no support entered the hard sets. See `../FOLLOWUP_NQ_PROTOCOL.md` for the dated result note and limits. Exact paired corpus IDs, candidate labels, raw rankings, summary metrics, and the runner are retained in `hard_arm_id_manifests.jsonl`, `hard_review_labels_confirm50_ai.jsonl`, `hard_review_supplemental_ai.jsonl`, `hard_arm_results.jsonl`, `hard_arm_summary.json`, and `run_hard_arm.py`.

The generator evaluation used one open model on the ordinary and hard search contexts. See `GENERATOR_EVALUATION.md` for its setup, scores, caveats, and output files.
