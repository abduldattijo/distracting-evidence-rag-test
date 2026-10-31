# Manual adjudication of screening pilot

Date: 29 October 2026

This is an AI-assisted manual review of the full abstracts and frozen questions. It is not an independent human annotation or a substitute for a second human reviewer. Labels use the protocol's meaning: **full** = sufficient to answer the question, **partial** = supports at least one answer component but not all, **none** = related topic alone without answer evidence.

## Gold-requirement gate

| Question | Manual decision |
|---|---|
| q000 | Keep the requirement that the reward/update objective remains tied to the original unhinted prompt. The gold abstract explicitly states this. Drop the anneal/drop-hints requirement: it is neither necessary to answer the question nor stated in the gold abstract. |
| q002 | Keep the homogeneity/parsimony requirement. The abstract describes informative class labels while avoiding fragmentation, and names the homogeneity and parsimony scores. |
| q003 | Rewrite the first requirement. The abstract supports false positives accumulating across repeated, multi-feature monitoring, but does not provide a base numerical rate per test. Keep the batch-size and Bonferroni requirements. |
| q004 | Keep the two asked facets: individual-reaction simulation on uncovered topics, and the effect of intuitive versus analytical prompting. Drop the extra interaction/trade-off requirement. |
| q005 | Do not retain the gate's mechanism claim that coactivation works only when draft and verifier share experts; the abstract does not say this. Drop the added "premise check" as it was not asked. The frozen question asserts fewer experts are loaded, while the abstract only says coactivation improves runtime and mitigates the cost of transferring more experts. Mark the gold support as partial for this mechanistic wording; preserve the frozen question in the primary set and disclose this limitation. |
| q008 | Drop the requirement to explain the benchmark's design and measurement procedure; the question asks for the system and its leaderboard result. |
| q009 | Keep the requirement that target-prefix probability is a sum over source strings mapped by the transducer. The gold abstract states this directly. |

The gate should be a diagnostic. It must not automatically remove requirements. Freeze the manually reviewed requirement set before screening candidates; if a gold abstract cannot support an essential requirement, revise the question under a documented quality rule or mark it as not fully answerable from its abstract.

## Candidate-paper adjudication

### q000

- **Partial:** 2609.13997, *Unlocking the Unsolvable*. Teacher-guided curriculum supplies learning signal on otherwise unsolvable problems and withdraws guidance toward unaided solving, but the abstract does not say that the update objective remains the original unhinted objective.
- **Partial:** 2608.00782, *Distill Where You Fail*. Adaptive teacher guidance targets zero-variance prompts, but the abstract does not establish the original unhinted objective condition.
- **None:** 2607.28582, *β-OPSD*. It studies privileged teacher guidance and reference-policy regularization, but not solution-prefix guidance on zero-signal RLVR prompts or preserving the unhinted objective.

### q003

- **Partial:** 2609.23254, *The Price of Self-Calibration*. It analyzes false-alarm budgets in adaptive monitoring, but not the specified PSI/KS/kernel detectors or the question's batch-size and multiple-testing comparison.
- **Partial:** 2609.09432, *SCCM*. It uses thresholding for local false-alarm mitigation in drift detection, but does not provide the requested detector-by-detector rates or correction comparison.

### q004

- **Partial:** 2609.22607, *Pretrained Persona Mixture Models*. It uses individual dialogue samples and discusses out-of-domain human simulation, but does not address social-media reactions or intuitive-versus-analytical instructions.
- **Partial:** 2608.29455, *Item-Mean Surrogates*. It directly evaluates individual-specific human-surrogate accuracy and finds important limits, but not social-media reactions or the prompt-style comparison.
- **Partial (weak/general):** 2607.26348, *When Synthetic Users Fail*. It supplies cross-domain evidence about LLM-simulated human survey responses, but not the question's social-media or prompt-style conditions.
- **None:** 2609.36365, *Engineering Simplicity*; it studies decisions in auctions and matching, not simulation of individual social-media reactions.
- **None:** 2608.08889, *LLM Reasoning for Subjective Tasks*; it studies reasoning style in subjective verification, not human-reaction simulation.

### q001: manual screen of the saved v3 gold-similarity top 50

No full alternative answer was found. Four papers are **partial**; the other 46 are **none** under the component-based rubric.

- **Partial:** 2607.16112, *Harmonizing AI Safety Thresholds*. It proposes harmonized thresholds and risk modeling for misuse domains including biological risk, but does not evaluate non-expert CBRN uplift studies.
- **Partial:** 2609.00498, *Validity-Aware Jailbreak Evaluation*. It contributes a validity/correctness rubric for harmful-task evaluation, but not cross-study CBRN uplift measurement.
- **Partial:** 2609.30454, *Auditing System-1 Models on Biosecurity-Relevant Benchmarks*. It evaluates biosecurity-relevant benchmarks with calibration and selective-prediction measures, but does not study AI assistance to non-experts or cross-study uplift comparability.
- **Partial:** 2609.23201, *Do Not Trust the Benchmark*. It recommends task-specific evaluation, configuration disclosure, question validation, and scoped generalization claims, but is not specific to CBRN uplift.

The other 46 saved candidates do not provide an answer component under this rubric. The gold abstract for q001 fully supports the question.

## Reproducibility check

The q001 list above is the exact saved v3 candidate list, sorted by its stored ranks. The initial 6/50 comparison was between the gold-similarity and question-similarity lists, not a rerun of the same list; it is not evidence of nondeterminism. The frozen manifests now make the candidate sets auditable. Their recorded hashes were verified locally, and the gold manifest's top 50 matches the saved v3 list for all ten pilot questions. The reported overflow/invalid NumPy warnings are a numerical-library warning despite finite, close scores; the manifests' float64 scores and deterministic tie-break should be the source of truth for future screen construction.

## Audit of model-negative labels

A fixed-seed, question-stratified sample of four Haiku-"none" distractors from each of the nine judged pilot questions (36 pairs total; seed 20261029) was manually checked against the full abstracts and frozen questions. No sampled paper provided answer evidence under the rubric; all 36 were none. This small audit found no false negatives, but it does not establish a low false-negative rate for the full screen.

## Recommendations

1. Use the manual requirement decisions above as a frozen, dated adjudication; do not use the automatic union of model gates to delete requirements.
2. Use the frozen manifests. Treat model labels as triage, manually adjudicate flags, and audit a seeded sample of "none" results. For unresolved labeler disagreements, exclude the paper from the distractor pool if either reviewer finds full or partial support; report this as the conservative union rule and give a sensitivity analysis using adjudicated labels only.
3. Keep q001 in the benchmark and use the manual screen for its saved candidate set. Preserve q001's refusal as a process note.
4. Keep q005 in the frozen primary set, but report its abstract-support limitation and a sensitivity result excluding it if the final analysis depends on full gold support.

## q001: question-similarity list and requirement check (29 October 2026)

The question-similarity manifest's top 50 shares six papers with the already reviewed gold-similarity list. I read the 44 new abstracts in full. None is a full answer. Three are **partial** under the conservative component rubric and should be excluded from this order's distractor pool:

- **Partial:** 2609.06222, *Relevance is not enough*. It studies consequential scientific communication to non-experts and evaluates completeness, supporting the measurement/helpfulness component at a general level. It does not address CBRN uplift-study comparability.
- **Partial:** 2609.38559, *Defining and Categorising Human-AI Interactions in Clinical Trials*. It proposes a multidimensional framework for consistent categorisation and comparison of human-AI studies, including participating human groups. It does not supply CBRN-specific definitions, baselines, or uplift measures.
- **Partial:** 2608.00794, *Measurement Without Validity*. It addresses validity, task generation, evaluator calibration, and reliability in agentic evaluation. It supports general evaluation-quality considerations, but not CBRN uplift design.

The other 41 new papers are **none**: they do not provide an answer component for harmonising non-expert CBRN uplift studies. The six papers shared with the gold-order list are none under the earlier adjudication. The question-order list therefore adds three partial supporters and no full supporters. Remove these three and backfill from the frozen question-order ranking before constructing its corpora.

Freeze these answer requirements for q001 before screening:

1. State a framework that makes CBRN uplift studies comparable by specifying non-expert eligibility, threat scope, and how material uplift is estimated.
2. State the evaluation design safeguards or reporting rules needed for comparison, including explicit baselines and scoring/decision criteria; distinguish generative from revisionist uplift where applicable.

The gold abstract supports both requirements: it names the Threshold Exceedance Criteria framework and its eligibility, scope, and uplift components; it also calls for prespecified criteria, explicit baselines, separate generative and revisionist estimates, and separation of screening signals from confirmed risk.
