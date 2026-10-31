# Notice on data and licenses

The code in this repository is released under the MIT License (see `LICENSE`).

Some data files contain text from other sources, which keep their own licenses.

- **Wikipedia passages.** The label, review packet, candidate pool and generator files under `articles/distractor-corpus-test/followup_nq/` contain passages from English Wikipedia, as packaged by the BEIR version of Natural Questions. Google lists Natural Questions under the Creative Commons Attribution Share Alike 3.0 license. Treat these passages as CC BY-SA 3.0 and credit Wikipedia and its contributors.
- **Natural Questions and BEIR.** Kwiatkowski et al., Natural Questions (Google Research, 2019), and Thakur et al., BEIR (2021). This repository does not include the 1.4 GB BEIR download. `followup_nq/freeze_nq.py` records which files and checksums were used.
- **Answer references.** `generator_full.jsonl` includes accepted answers for 70 of the 100 hard arm questions. They come from the Natural Questions test references released with the Dense Passage Retrieval project. The full file is not included. Download it from its original source and check its license before any other use.
- **arXiv abstracts.** `agentic-rag-eval/data_full/papers.jsonl` holds the titles and abstracts of 2,000 arXiv papers used in the first test. The papers belong to their authors, and arXiv makes its metadata available through its own terms.
