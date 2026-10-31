"""Freeze the candidate manifests: the ranked non-gold papers for every question, both similarity modes.

Scores are computed in float64 (the float32 matrix product raises spurious divide by zero, overflow and
invalid value warnings on this machine's numpy 2.0.2; its scores are finite and agree with float64 to about
1.5e-7, and the nearest-n sets are identical for every question and every corpus size, but float64 is
what the manifest records). Ties break by paper index. Writes manifest_<mode>.jsonl (one row per question:
ranked paper ids and scores) and manifest_hashes.json (sha256 of each file plus model revisions).
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_pilot as rp  # noqa: E402

papers, questions = rp.load_data()
pidx = {p["id"]: i for i, p in enumerate(papers)}
model = rp.load_sentence_transformer(rp.DEFAULT_SELECTION_MODEL, rp.DEFAULT_SELECTION_REVISION)
pv = rp.encode(model, [p["abstract"] for p in papers]).astype(np.float64)
hashes = {"selection_model": rp.DEFAULT_SELECTION_MODEL, "selection_revision": rp.DEFAULT_SELECTION_REVISION}
for mode in ("gold", "question"):
    texts = [papers[pidx[q["gold_id"]]]["abstract"] if mode == "gold" else q["question"] for q in questions]
    qv = rp.encode(model, texts).astype(np.float64)
    path = HERE / f"manifest_{mode}.jsonl"
    with path.open("w") as f:
        for qi, q in enumerate(questions):
            s = np.array([float(np.dot(pv[i], qv[qi])) for i in range(len(papers))])
            s[pidx[q["gold_id"]]] = -np.inf
            order = np.argsort(-s, kind="stable").tolist()[:-1]
            f.write(json.dumps({"qid": q["qid"], "gold_id": q["gold_id"], "mode": mode,
                                "ranked_ids": [papers[i]["id"] for i in order],
                                "scores": [round(float(s[i]), 8) for i in order]}) + "\n")
    hashes[f"manifest_{mode}.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
(HERE / "manifest_hashes.json").write_text(json.dumps(hashes, indent=1))
print(json.dumps(hashes, indent=1))
