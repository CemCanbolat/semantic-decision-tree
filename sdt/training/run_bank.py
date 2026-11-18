"""Bank combiner: select over every banked question at once.

    python3.11 -m sdt.training.run_bank \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --out eval/reports/iteration6_bank.json --backfill
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.eval import metrics
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.gain import class_weights
from sdt.schemas import DecisionRecord, GateOutcome
from sdt.training.cache import ScoreCache
from sdt.training.combiner import (SoftmaxModel, build_features, cross_validate,
                                   fit_temperature)
from sdt.training.question_bank import QuestionBank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backfill", action="store_true",
                    help="score missing bank questions live before combining")
    ap.add_argument("--iters", type=int, default=900)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    bank = QuestionBank(str(args.bank))
    questions = bank.semantic_questions()

    missing = sum(len(cache.missing(q, [e.example_id for e in examples])) for q in questions)
    print("bank: %d semantic questions | missing score pairs: %d" % (len(questions), missing))
    if missing and args.backfill:
        client = GeminiDecisionClient()
        done = 0
        for ex in examples:
            todo = [q for q in questions if cache.get(q, ex.example_id) is None]
            if todo:
                sv = client.score_batch(ex.text, ex.context, todo)
                for q in todo:
                    cache.put(q, ex.example_id, sv.scores[q.qid])
                done += 1
        cache.save()
        print("backfilled %d examples" % done)
    elif missing:
        raise SystemExit("bank incomplete and --backfill not set; cannot combine")

    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]
    X, names = build_features(examples, questions, cache, include_computed=True)

    oof, acc, f1 = cross_validate(X, y, sample_w, labels, iters=args.iters)
    T = fit_temperature(oof, y)
    records = []
    for z, ex in zip(oof, examples):
        m = max(z)
        expst = [math.exp((v - m) / T) for v in z]
        st = sum(expst)
        probs = [e / st for e in expst]
        order = sorted(range(len(labels)), key=lambda c: -probs[c])
        records.append(DecisionRecord(example_id=ex.example_id, tree_id="bank", scores={},
                                      class_posterior={}, margin=probs[order[0]] - probs[order[1]],
                                      bottleneck_nodes=[], outcome=GateOutcome.AUTO_EXECUTED,
                                      predicted=labels[order[0]]))
    cov97 = metrics.coverage_at_precision(records, examples, 0.97)
    preds = [max(range(len(labels)), key=lambda c: z[c]) for z in oof]
    per_class = {labels[i].value: "%d/%d" % (
        sum(1 for yy, p in zip(y, preds) if yy == i and p == i),
        sum(1 for yy in y if yy == i)) for i in range(len(labels))}

    model = SoftmaxModel(labels, len(names)).fit(X, y, sample_w)
    model_path = args.out.with_name("bank_model.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps({
        "labels": [l.value for l in labels], "features": names,
        "temperature": T, "weights": model.W, "bias": model.b,
    }, indent=2), encoding="utf-8")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bank_summary": bank.summary(), "n_features": len(names),
        "cv_accuracy": round(acc, 4), "cv_macro_f1": round(f1, 4),
        "temperature": T, "coverage_at_0.97": cov97,
        "per_class_recall": per_class, "best_feature_set": "bank",
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("bank combiner: features %d | cv acc %.4f | cv macro_f1 %.4f | T %.2f | cov@0.97 %s" % (
        len(names), acc, f1, T, json.dumps(cov97)))
    print("per-class:", json.dumps(per_class))
    print("report ->", args.out, "| model ->", model_path)


if __name__ == "__main__":
    main()
