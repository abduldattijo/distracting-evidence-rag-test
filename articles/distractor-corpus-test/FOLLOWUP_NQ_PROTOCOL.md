# Follow-up protocol: corpus growth and same-page distractors

**Status:** dataset, question split, and Tantivy configuration are frozen. The 40-query pilot, primary 3,412-question corpus-growth experiment, secondary 100-question same-page hard-distractor arm, and a post-hoc generator add-on are complete. See the dated result notes below. The predeclared primary threshold was met at 100,000 passages; do not generalize it beyond this retriever and benchmark.  
**Study type:** retrieval benchmark with a supplementary, exploratory generator evaluation.  
**Active data:** BEIR Natural Questions (NQ), test queries and Wikipedia passage corpus.

## 1. Why this dataset

BioASQ 10b would give us biomedical questions and gold evidence, but its official download requires registration. The public BioASQ yes/no file released with the biomedical RAG paper has no evidence IDs, so it cannot answer our retrieval question without substantial new annotation.

BEIR NQ is an openly downloadable alternative with 3,452 naturally asked questions, relevance judgments, and 2,681,468 Wikipedia passages. It gives us a much larger collection and questions that were not generated from the target passages. The domain changes from biomedical literature to general web questions; the article must say this clearly and must not imply that the result is biomedical-specific.

## 2. Research questions and hypotheses

### Primary: natural corpus growth

With BM25 held fixed, does the probability of retrieving at least one benchmark-relevant passage in the top 10 fall as the searchable Wikipedia collection grows from 10,000 passages to the full 2.68 million?

### Secondary: concentrated hard distractors

At a fixed 10,000-passage size, do 20 highly query-similar, same-Wikipedia-page passages that reviewers judge unable to answer the question reduce top-10 retrieval success more than replacing the same 20 slots with ordinary random passages?

The second question tests a specific kind of distractor: another passage from the page that contains the answer. It is not a claim about every form of semantic distraction.

## 3. Dataset and integrity checks

Downloaded from the public BEIR dataset server. The archive checksum matches BEIR's published MD5 (`d4d3d2e48787a744b6f6e691ff534307`). Extracted data contain:

- 2,681,468 corpus passages;
- 3,452 questions;
- 4,201 positive relevance judgments, covering all 3,452 questions;
- 4,201 distinct judged relevant passage IDs, all present in the corpus.

The corpus is divided into passages with a Wikipedia page title and passage text. Qrels are positive-only; a missing qrel does not prove a passage is irrelevant. Main results will therefore be called **BEIR-qrel retrieval success/recall**, not exhaustive evidence coverage. The same-page distractor arm gets an additional support review before passages are called non-supporting.

Raw files are retained under `followup_nq/data/nq/`; do not redistribute them without checking the source terms. The source Natural Questions dataset is CC BY-SA 3.0; the BEIR package identifies its benchmark release as CC BY-SA 4.0. Cite both BEIR and Natural Questions.

## 4. Pilot split and confirmatory questions

Before any NQ retrieval score is calculated, sample 40 question IDs uniformly with `random.Random(20261029)` from the sorted 3,452 IDs. These 40 are reserved for engineering and headroom checks and are permanently excluded from the confirmatory analysis. The remaining **3,412 questions** form the confirmatory set.

The planned 40-question pilot uses BM25 at 10,000, 100,000, and full-corpus sizes. It checks corpus construction, query parsing, relevance-ID joins, and whether the scale range leaves measurable headroom. It is not used to choose a preferred result or retune the failure threshold. If a technical change is required, log it before the confirmatory run and rerun the pilot only.

**Pilot status (2026-10-29):** the first SQLite FTS5 pilot pass was too slow at full scale and is retained as an engineering attempt only. The retriever was amended before confirmatory retrieval to Tantivy 0.25.1 BM25. On the same 40 reserved questions, Tantivy returned success@10 of 38/40 at 10,000 passages (macro qrel Recall@10 0.950; MRR@10 0.840), 34/40 at 100,000 (0.838; 0.662), and 22/40 at 2,681,468 (0.512; 0.302). Median per-query latency was 0.40 ms, 2.35 ms, and 47.21 ms respectively. These are headroom and engineering observations, not confirmatory evidence. `pilot_tantivy_results.jsonl` contains the 120 query-condition rows. The SQLite FTS5 pilot outputs are retained separately and must not be combined with Tantivy results.

**Confirmatory primary result (2026-10-29):** all 3,412 held-out questions ran at 10k, 100k, 500k, and 1m with three nested random draws per size, plus the fixed 2,681,468-passage corpus once (44,356 question-condition rows). Mean Success@10 across draws was 86.8%, 75.6%, 64.8%, and 58.1%; full-corpus Success@10 was 46.6%. The first size meeting the predeclared degradation rule was 100,000: −11.1 percentage points versus 10k, paired question-bootstrap 95% interval −12.1 to −10.2 points. With Bonferroni adjustment across the four tested comparisons, the 100k interval is −12.4 to −9.9 points and still excludes zero. At 500k the unadjusted drop was 22.0 points (95% interval −23.4 to −20.7); at 1m, 28.7 points (−30.2 to −27.3); at full scale, 40.2 points (−41.8 to −38.5). Bootstrap intervals resample questions and average the three random draws; they do not include draw-to-draw variation, which is reported separately. Median full-index search latency was 47.38 ms per query on this machine. This is a benchmark-specific operational threshold for Tantivy BM25 on BEIR NQ, not a universal RAG limit.

## 5. Main natural-growth experiment

### Corpus sizes and draws

Test nested corpus sizes of **10,000, 100,000, 500,000, 1,000,000, and 2,681,468 passages**. Ten thousand is the smallest size because the 4,201 distinct qrel-positive passages must all be present in every confirmatory corpus.

For each of three fixed seeds (11, 23, 37), start with all 4,201 positive qrel passage IDs from the complete 3,452-question benchmark, including the 40 reserved pilot questions, then add non-qrel passages sampled without replacement from the remaining corpus. The confirmatory questions have 4,151 distinct qrel-positive passage IDs. Each larger corpus contains the smaller corpus from that seed. The full-corpus endpoint is fixed and run once. Save exact passage-ID manifests and hashes for every size and seed. Report the three draws separately; do not treat them as three independent query samples.

The random fill excludes all 4,201 benchmark-positive passages. This prevents a passage known to answer another NQ question from being deliberately placed in the random background pool. It does not rule out unjudged answer-bearing passages.

### Retriever

Use one fixed **BM25** retrieval system over passage title and text. The frozen implementation is Tantivy 0.25.1 with its default BM25 scoring parameters, title field boost 2, body boost 1, OR-combined unique lowercase alphanumeric query tokens, and passage ID ascending as a deterministic tie-break among the retrieved top 100. Fields use Tantivy's default text tokenizer and store term frequencies; only the ID field is stored for result retrieval. Python is 3.9. The implementation change from SQLite FTS5 was made before confirmatory retrieval because the original full-index query took 11.39 seconds; the same 40 reserved questions were rerun under Tantivy. SQLite and Tantivy pilot outputs are kept separate. Do not change the implementation, tokenization, field weights, parser, or tie-break after confirmatory retrieval starts.

The dense model was too slow on the available machine in the MedAESQA feasibility pass, and building a 2.68-million-passage dense index would require a separate compute plan. Do not add a dense result unless it can be run over every corpus condition with identical settings and verified rankings.

## 6. Optional same-page hard-distractor extension (secondary to the primary corpus-growth analysis)

### Query sample

The feasibility pilot found that a minimum pool of 20 same-page passages leaves no room to obtain 20 reviewed `none` passages when even one candidate supports the answer. The hard-arm sample will therefore be drawn from confirmatory questions with **at least 50 distinct same-page candidate passages**, after excluding every qrel-positive passage ID in the full benchmark and deduplicating exact text. Sample 100 questions uniformly from the sorted eligible IDs using a new, recorded seed; if fewer than 100 gold-valid questions remain, use all gold-valid sampled questions and disclose the number. This capacity rule was added after the small feasibility pilot and before any hard-arm retrieval outcome was examined. The target population is consequently questions with sufficiently large same-page candidate pools, not all NQ questions. The hard arm is secondary and is not part of the completed corpus-growth result.

Before reviewing candidate passages, inspect the qrel-positive passage(s) against the question. A question is gold-valid only if its qrel passage directly answers the requested fact or gives sufficiently specific evidence for the answer. Exclude mismatched, ambiguous, or materially incomplete gold passages before candidate support labeling and replace them using the same seeded sampling procedure from the remaining eligible population. Save each decision, rationale, and replacement lineage. This is AI-assisted adjudication by the assistant, not independent human annotation.

### Candidate selection and review

For each selected question, collect passages from its answer-page title(s), excluding all qrel-positive IDs and deduplicating exact text. Rank these candidates against the question with a frozen local embedding model that is **not** the tested BM25 system. Read and label candidates in blinded order, without showing similarity score or rank.

Label each passage:

- **support:** the passage alone answers the question;
- **partial:** it contributes answer evidence but is insufficient alone;
- **none:** it does not provide answer evidence.

Only passages labeled `none` enter the hard-distractor pool. Exclude `support` and `partial` passages. Review the top 50 candidates per selected question, or all available if fewer; if that does not yield 20 `none` passages, review further candidates in ranked order until 20 are available or the pool is exhausted. Preserve every label and decision.

The assistant performs AI-assisted screening, and the author manually reviews the candidate passages and labels before the retrieval comparison. Report the human review and AI assistance plainly. This is a single-author review, not independent double annotation; retain the question/passage packets and keep the hard-arm claims narrow.

### Paired corpus conditions

For each selected question, construct a 10,000-passage corpus containing that question's qrel-positive passages plus random non-qrel fill. Create three paired random draws. For each draw, compare:

- **ordinary:** the 10,000-passage corpus as sampled;
- **hard-mixed:** replace 20 randomly chosen ordinary slots with 20 reviewed same-page `none` passages for that question.

Both corpora have exactly 10,000 distinct passages. The hard-mixed corpus keeps the question's gold passages and all other sampled passages fixed. The only intended change is replacing 20 ordinary distractors with 20 reviewed, query-similar same-page distractors.

If a question has fewer than 20 reviewed `none` passages, exclude it from this paired arm under the predeclared eligibility rule; do not lower the dose after seeing its retrieval result.

## 7. Retrieval metrics

### Primary outcome

**Success@10:** fraction of questions with at least one BEIR-qrel-positive passage among the top 10. This is the outcome for the predeclared degradation threshold.

### Secondary outcomes

- qrel **Recall@10** (fraction of qrel-positive passages retrieved in the top 10);
- MRR@10 and nDCG@10 using BEIR qrel grades;
- rank of the first qrel-positive passage;
- corpus build time, index size, and query latency, all measured on the same machine and reported as implementation-specific.

Do not describe these metrics as proof that all answer evidence was retrieved. BEIR qrels are positive-only and may omit other answer-bearing passages.

The predeclared primary corpus-growth and secondary hard-retrieval arms do not include a generator. Their results concern retrieval quality and cannot establish whether a language model can reason around misleading context or produce a correct answer. A separate generator evaluation was added after those retrieval results; it is exploratory and documented in the dated note below and `followup_nq/GENERATOR_EVALUATION.md`.

## 8. Statistical plan and predeclared threshold

### Primary growth comparison

Compare the full corpus with the 10,000-passage condition for the 3,412 confirmatory questions. Average across the three random draws for the point estimate and also report each draw separately. Use a paired question bootstrap with 10,000 resamples and fixed seed 20261030.

### Hard-distractor comparison

For the 100-question subset, compare hard-mixed versus ordinary at the same corpus size and draw. Use paired question bootstrap intervals and show all three draw-specific estimates. This arm is a secondary test and does not replace the all-question growth result.

### Operational degradation point

Before the pilot and confirmatory run, define the degradation point as the smallest tested corpus size where:

1. Success@10 is at least **5 percentage points below** the 10,000-passage result; and
2. the paired 95% bootstrap interval for the difference is entirely below zero.

Five percentage points means at least one additional failed retrieval question per 20. This is a predeclared, practical reporting threshold, not a universal RAG failure boundary. The continuous curve is primary. If the rule is not met, report “no degradation point observed within the tested range,” along with the largest estimated drop and its interval. Do not change the threshold after inspecting the pilot or final scores.

The two main hypotheses (growth and hard-versus-ordinary) are distinct; report the hard arm as secondary and use Holm adjustment if inferential claims are made for both.

## 9. Quality checks and reproducibility package

Before retrieval:

- freeze checksums, pilot IDs, confirmatory IDs, qrels, selected hard-arm IDs, random seeds, code commit, and configuration;
- verify every qrel-positive passage is in every natural-growth corpus;
- verify corpus sizes, uniqueness, nested membership, and absence of qrel-positive IDs in the random fill;
- verify every hard passage is same-title, non-qrel, reviewed `none`, and appears only in its intended question's hard-mixed corpus;
- validate BM25 scoring and top-10 tie-breaking on a small fixture, then compare a seeded subset against a second implementation or hand-computed scores;
- save raw top-100 rankings, metric inputs, annotations, logs, and index metadata.

The public reproduction bundle should contain scripts, versions, corpus ID manifests where licensing permits, qrel derived analysis code, query split, annotation rubric and permitted labels, retrieval and generation result files, and machine specifications. The article must state that the corpus is Wikipedia passages, questions are NQ, the primary retriever is BM25, and the supplementary generator result is a post hoc evaluation of one model.

## 10. Current data status

- Dataset download: complete.
- Archive MD5: verified as `d4d3d2e48787a744b6f6e691ff534307`.
- Count and qrel-join checks: complete; 3,452 queries, 2,681,468 passages, 4,201 positive qrels, 0 missing qrel query IDs, 0 missing qrel passages.
- Local MedAESQA/BM25 pilot: complete separately; it is not part of this NQ test.
- NQ pilot: complete for 40 reserved questions at 10k, 100k, and full corpus with Tantivy; it was kept separate from the confirmatory set.
- NQ primary corpus-growth run: complete; 44,356 question-condition rows, all 13 conditions contain 3,412 unique questions, and indexes were checked for exact size and qrel coverage.
- Primary analysis: first predeclared degradation point at 100k (−11.1 pp; paired question-bootstrap 95% interval −12.1 to −10.2 pp).
- Same-page extension: candidate extraction and BGE ranking were prepared for feasibility. It was completed after the initial protocol status was recorded; the dated result note below gives the labels, paired retrieval results, and limits.
- SQLite FTS5 feasibility indexes, Tantivy pilot indexes, confirmatory indexes, raw rankings, tie-broken rankings, and analysis summary are retained separately under `followup_nq/`.
- No paid calls have been made.

## Dated result note — 29 October 2026: same-page hard-distractor arm completed

This note records the completed secondary arm; it does not replace the primary corpus-growth result or change the predeclared 5-point corpus-size threshold.

- The final 100-question sample matches `hard_arm_manifest_confirm50_final100.json`. It was sampled from the frozen eligible confirmatory pool, then gold-checked and replenished under the recorded replacement procedure. All 100 selected questions have a recorded `usable` gold decision. Candidate passage screening used AI assistance, followed by the author's manual review of the passages and labels. This is not independent double annotation.
- Candidate packets contain same-title passages for every selected question. The first 50 query-similar candidates were AI-screened and then manually reviewed by the author, with labels full, partial, or none, blinded to BGE rank and score. Totals were 81 full, 269 partial, and 4,650 none. Six additional passages were reviewed for `test1591`, all labeled none, to reach the predeclared minimum of 20. Only none passages were sampled into hard sets. Thus 5,006 passages were reviewed in total. This was one author's manual review after AI-assisted screening, not independent double annotation; describe the hard arm accordingly.
- For each question and each seed 11, 23, and 37, the ordinary 10,000-passage corpus contained that question's positive qrel passage(s) plus random non-qrel passages. To guarantee a true 20-document paired change, ordinary fill excluded the reviewed same-page candidate pool for that question. The hard-mixed corpus replaced 20 ordinary passages with 20 fixed, randomly selected reviewed-none passages. Both conditions retained all qrel positives, contained exactly 10,000 unique passages, and differed by exactly 20 IDs. This implementation detail is disclosed as a clarification made before retrieval; it does not change either condition after observing results.
- Retrieval used the frozen Tantivy 0.25.1 BM25 settings. There were 600 question-draw-condition rows. Success@10 was 90.7% for ordinary and 82.3% for hard-mixed, a paired difference of −8.3 percentage points (95% question-bootstrap interval −13.7 to −3.7). Draw-specific differences were −9, −8, and −8 points. Qrel Recall@10 was 90.2% versus 78.7% (difference −11.5 points; interval −17.0 to −6.7); MRR@10 was 0.787 versus 0.518 (difference −0.268; interval −0.331 to −0.207); nDCG@10 was 0.809 versus 0.558 (difference −0.251; interval −0.304 to −0.200). The hard arm is secondary and selected; it does not establish a universal retrieval or RAG failure point.
- A Holm-adjusted exact paired sign-test check across the primary corpus-growth and hard-distractor contrasts found the hard-arm direction remained below zero (two-sided raw p = 0.000488; Holm-adjusted p = 0.000977). This sign test is a robustness check, while the paired question-bootstrap interval remains the reported effect interval.
- Median per-index build time was 0.40 s, median index size 2.75 MB, and median query parse-plus-search latency 0.94 ms. These are specific to this implementation and machine; the indexes were built sequentially per question and draw.
- Integrity checks passed: 100 questions; 300 ordinary and 300 hard results; all 600 corpora at 10,000 unique IDs; exact 20-ID paired replacements; qrel-positive passages retained; no top-100 tie extended through rank 10. The instrumented rerun reproduced the same Success@10 counts as the first pass. Two near-tied orderings elsewhere in top 100 differed slightly between passes, without changing any Success@10 result.
- Reproduction artifacts: `followup_nq/hard_review_labels_confirm50_ai.jsonl`, `hard_review_supplemental_ai.jsonl`, `hard_arm_id_manifests.jsonl`, `hard_arm_results.jsonl`, `hard_arm_summary.json`, `hard_arm_run_manifest.json`, and `run_hard_arm.py`. The exact 600 corpus ID lists are frozen in the ID-manifest JSONL.

## Dated result note — 29 October 2026: supplementary generator evaluation completed

This post-hoc add-on was not part of the preregistered retrieval hypotheses and does not alter the 5-point retrieval degradation threshold.

- Qwen3-4B received the actual top 10 passages from the ordinary or hard-mixed BM25 run for all 100 questions and three draws. A second condition placed the same answer-bearing qrel passage first in both contexts, followed by nine non-qrel passages from the corresponding retrieval result.
- The public DPR Natural Questions test-reference file matched 70/100 queries by exact text after case, Unicode, punctuation, and whitespace normalization. Accuracy is reported only on these 70, using SQuAD-style exact match and token F1 against answer aliases. Three fixed draws are averaged within question; uncertainty uses a paired question-cluster bootstrap with 10,000 resamples.
- For actual retrieved top-10 contexts, exact match was 57.6% ordinary and 45.7% hard-mixed (hard minus ordinary −11.9 percentage points; 95% interval −20.0 to −4.3). Token F1 was 67.4% and 55.1% (difference −12.2 points; interval −20.4 to −4.7).
- With the same answer-bearing passage placed first, exact match was 60.5% ordinary and 60.0% hard-mixed (difference −0.5 points; interval −4.3 to +3.3). Token F1 was 73.3% and 73.5% (difference +0.2 points; interval −2.5 to +3.9). This suggests that the end-to-end difference is associated with retrieved evidence being ranked lower or missing; it does not show a clear penalty from the remaining hard context when evidence is first.
- In the full 100-question retrieval arm, top-10 qrel success remained 90.7% ordinary and 82.3% hard-mixed. In the 70-question generator-scored subset, the first qrel passage was rank 1 in 152/210 ordinary contexts and 82/210 hard contexts; it was absent from 21 ordinary and 31 hard contexts.
- An AI-assisted post-hoc review of the 16 questions with different exact-match outcomes found both clear hard-context errors and answer-key/metric edge cases. For example, the model's full list of all accepted FIFA host-country answers was marked wrong against any single alias, and a year-only answer was scored below the full date. Primary scores were not altered after review.
- Scope limits: one model (`Qwen/Qwen3-4B`), one prompt, one decoding configuration, 70 answer-keyed questions, DPR answer aliases, and no independent human answer annotation. The model repository was loaded without recording an immutable commit hash. These results are exploratory and do not establish behavior for other generators or domains.
- Reproduction artifacts: `followup_nq/generator_full.jsonl`, `generator_inputs.jsonl.gz`, `generator_summary.json`, `generator_question_scores.csv`, `figures/generator_paired_effect.svg`, `run_generator.py`, `analyze_generator.py`, and `GENERATOR_EVALUATION.md`. No paid model API calls were made; Colab GPU usage was about 1.7 compute units, and the study's T4/A100 sessions were stopped after download.

## Sources

- [BEIR Natural Questions release](https://github.com/beir-cellar/beir): data format, corpus/query/qrel package, and published archive checksum.
- [BEIR dataset server](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/): downloadable `nq.zip` archive.
- [Google Natural Questions](https://ai.google.com/research/NaturalQuestions/download): original dataset description and CC BY-SA 3.0 terms.
- [BioASQ access page](https://participants-area.bioasq.org/datasets/): registration requirement that led us to this public alternative.

## Dated note — 29 October 2026: reviewer field corrected

The `reviewer` field in `followup_nq/hard_review_labels_confirm50_ai.jsonl` (100 records) and `hard_review_supplemental_ai.jsonl` (6 records) originally read "AI-assisted adjudication by ChatGPT". That did not match the review process, in which AI screening sorted the candidates and the author manually reviewed all 5,006 passages and assigned the final labels before retrieval ran. The field now says so. No other field changed in any record, which was checked against a backup. The new SHA 256 hashes are 3f8a452486b44cb95b823ea17bcecd3375f18ba03be56325b2092bf5660786dc for the main file and 0df8ede6c96cafa93afeb1b02168be776ad0459c8bb4b90f81db7a18347ae975 for the supplemental file, and both manifests (`hard_arm_run_manifest.json`, `hard_arm_manifest_confirm50_final100.json`) now record them. The old hashes were d1ab28ad4c11c588f89f3ef2d0264e57b96105e4d7f6a1c2889a096d6c776fdf and be6e5cef1e0172793a764529a7cec0e490e527eee78c34d763d68622a5c892e9. The retrieval results were not rerun, because the labels did not change.
