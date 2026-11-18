"""Evaluate a bank-selection state: full metrics + deployment model.

    python3.11 -m sdt.training.run_selected \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --state artifacts/bank_selection_v2.json \
        --out eval/reports/selected_v2_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.eval import metrics
from sdt.induction.gain import class_weights
from sdt.schemas import DecisionRecord, GateOutcome
from sdt.training.cache import ScoreCache
from sdt.training.combiner import SoftmaxModel, build_features, cross_validate, fit_temperature
from sdt.training.question_bank import QuestionBank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-out", type=Path, default=None)
    args = ap.parse_args()

    state = json.loads(args.state.read_text())
    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    bank = QuestionBank(str(args.bank))
    selected = [q for q in bank.semantic_questions() if q.text in state["included"]]

    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]
    X, names = build_features(examples, selected, cache, include_computed=True)
    oof, acc, f1 = cross_validate(X, y, sample_w, labels, iters=900)
    T = fit_temperature(oof, y)

    records = []
    for z, ex in zip(oof, examples):
        m = max(z)
        expst = [math.exp((v - m) / T) for v in z]
        st = sum(expst)
        probs = [e / st for e in expst]
        order = sorted(range(len(labels)), key=lambda c: -probs[c])
        records.append(DecisionRecord(example_id=ex.example_id, tree_id="selected", scores={},
                                      class_posterior={}, margin=probs[order[0]] - probs[order[1]],
                                      bottleneck_nodes=[], outcome=GateOutcome.AUTO_EXECUTED,
                                      predicted=labels[order[0]]))
    cov97 = metrics.coverage_at_precision(records, examples, 0.97)
    preds = [max(range(len(labels)), key=lambda c: z[c]) for z in oof]
    per_class = {labels[i].value: "%d/%d" % (
        sum(1 for yy, p in zip(y, preds) if yy == i and p == i), sum(1 for yy in y if yy == i))
        for i in range(len(labels))}

    model = SoftmaxModel(labels, len(names)).fit(X, y, sample_w)
    model_path = args.model_out or args.out.with_name(args.out.stem + "_model.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps({
        "labels": [l.value for l in labels], "features": names, "temperature": T,
        "weights": model.W, "bias": model.b, "selected_questions": state["included"],
    }, indent=2), encoding="utf-8")

    report = {"system": "bank-selected (%d semantic + computed)" % len(selected),
              "cv_accuracy": round(acc, 4), "cv_macro_f1": round(f1, 4),
              "temperature": T, "coverage_at_0.97": cov97,
              "per_class_recall": per_class,
              "trajectory": [h.get("macro_f1") for h in state.get("history", [])]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("selected system: features %d | cv acc %.4f | cv macro_f1 %.4f | T %.2f | cov@0.97 %s" % (
        len(names), acc, f1, T, json.dumps(cov97)))
    print("per-class:", json.dumps(per_class))
    print("report ->", args.out, "| model ->", model_path)


if __name__ == "__main__":
    main()
