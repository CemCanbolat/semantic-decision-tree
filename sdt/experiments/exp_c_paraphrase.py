"""Experiment C: paraphrase ensemble over the selected questions.

Iteration 6 selected 5 questions and kept BOTH delivery-status phrasings.
This experiment formalizes the effect: the frontier writes 2 paraphrases of
each selected question (plain nouls, no criteria); all are scored; the
combiner trains on 5 originals + 10 paraphrases + computed features.

    python3.11 -m sdt.experiments.exp_c_paraphrase \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --selection artifacts/bank_selection.json \
        --out eval/reports/expC_paraphrase.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.decision_types import noul as mk_noul
from sdt.experiments.common import score_with_cache
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.training.cache import ScoreCache
from sdt.training.combiner import build_features, cross_validate, fit_temperature


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    state = json.loads(args.selection.read_text())
    selected_texts = state["included"]
    frontier = GeminiFrontierClient()
    client = GeminiDecisionClient()

    originals = [mk_noul(t, "expC:original") for t in selected_texts]
    paraphrases = []
    for t in selected_texts:
        system = ("You rephrase yes/no screening questions without changing what "
                  "they ask. Keep them positive-phrased and concrete.")
        user = ("Question: " + t + chr(10) + "Write exactly 2 rephrasings, one per "
                "line, no numbering.")
        raw = frontier.complete(system, user)
        for line in raw.splitlines():
            q = line.strip().lstrip("-*0123456789. ").strip()
            if len(q) > 8 and q.endswith("?") and q.lower() != t.lower():
                paraphrases.append(mk_noul(q, "expC:paraphrase"))
    paraphrases = paraphrases[:2 * len(selected_texts)]
    print("paraphrases written: %d for %d originals" % (len(paraphrases), len(originals)))

    t0 = time.time()
    scored = score_with_cache(client, examples, originals + paraphrases, cache)
    print("scored %d examples (%.0fs)" % (scored, time.time() - t0))

    from sdt.induction.gain import class_weights
    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]

    results = {}
    X5, _ = build_features(examples, originals, cache, include_computed=True)
    _, a5, f5 = cross_validate(X5, y, sample_w, labels, iters=700)
    results["originals_only"] = {"n_features": len(X5[0]), "cv_accuracy": round(a5, 4),
                                 "cv_macro_f1": round(f5, 4)}
    X15, names = build_features(examples, originals + paraphrases, cache, include_computed=True)
    oof, a15, f15 = cross_validate(X15, y, sample_w, labels, iters=700)
    T = fit_temperature(oof, y)
    results["with_paraphrases"] = {"n_features": len(names), "cv_accuracy": round(a15, 4),
                                   "cv_macro_f1": round(f15, 4), "temperature": T}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "C_paraphrase_ensemble",
        "paraphrases": [q.text for q in paraphrases],
        "results": results, "best_feature_set": "expC",
    }, indent=2), encoding="utf-8")
    print("originals:    acc %.4f f1 %.4f" % (a5, f5))
    print("with paraphr: acc %.4f f1 %.4f T %.2f" % (a15, f15, T))
    print("report ->", args.out)


if __name__ == "__main__":
    main()
