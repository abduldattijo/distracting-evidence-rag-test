# How much distracting evidence can your RAG pipeline handle?

Code, saved results and checks for an article that measures how much a search step loses as its collection grows, and what 20 passages from the answer's own page do to it. The article link will be added here after it is published.

## What is in here

- `articles/distractor-corpus-test/` holds the arXiv test (2,000 papers, 298 questions), its protocol and screening files, and a `followup_nq/` folder with the Wikipedia tests (2.68 million passages, a 100 question same page test and an answer model test).
- `agentic-rag-eval/` holds only what the arXiv scripts import (`config.py`, `llm.py`) and the frozen question and paper files in `data_full/`. The paths match the original project, so the scripts run without edits.

## How to check the article's numbers

- Wikipedia size sweep. `followup_nq/confirmatory_results_slim.csv.gz` has one row per question, size and draw (44,356 rows). Success at 10 is the mean of `success_at_10` for each size.
- Same page test. `followup_nq/hard_arm_results.jsonl` has 600 rows. `hard_arm_id_manifests.jsonl.gz` lists the exact passage IDs in every collection.
- Answer model test. `followup_nq/generator_full.jsonl` has all 1,200 answers with the passages shown, and `analyze_generator.py` recomputes the scores.
- arXiv test. `full_gold_screened.jsonl.gz` has the screened rows behind Table 1, and `summary_gold_screened.txt` and the `sens_*.txt` files have the summaries.

Several results files are compressed because they are large. Run `gunzip -k` on a `.gz` file before a script reads it.

## What was checked by a person and what was not

- A person (the author) read and labeled all 5,006 same page Wikipedia passages, after AI screening sorted the candidates and before any search ran. There was one reviewer and no second annotator.
- The arXiv distractor labels came from two AI models (Claude Haiku 5.5 and Claude Sonnet 5.5), and the author read small samples of them (see the correction note at the end of `articles/distractor-corpus-test/PROTOCOL.md`). Not every pair was read by a person.
- Two spot checks of 40 items each (arXiv papers ranked 51 to 200, and 40 swapped Wikipedia passages) were AI reads, not human audits.

## Notes on the files

- The file names `candidates_ranked_PRIVATE_*.jsonl` are kept from the working folder. "Private" meant the similarity ranks were hidden from the labeler until the labels were final.
- `hard_review_labels_confirm50_ai.jsonl` keeps "ai" in its name because the candidates were screened with AI help. Its `reviewer` field says how the final labels were made. The hashes in the manifests match these files. The protocol records when and why the field was corrected.
- Retrieval over the full 2.68 million passages needs the BEIR download and the Tantivy indexes, which are not included. The slim results file lets you recompute every number without them.
- Running the arXiv screening scripts again makes paid model calls and needs the `ANTHROPIC_API_KEY` environment variable.
