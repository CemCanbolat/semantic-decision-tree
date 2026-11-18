"""Iteration 12: fixes 1-4 applied - interactions, mild-weight option,
multi-type selection, dedup + class caps + OVR pass.

    python3.11 -m sdt.experiments.run_iteration12 \
        --corpus data/curated/tau2_telecom_full/examples.jsonl \
        --cache cache/scores_telecom_full.json \
        --bank artifacts/question_bank_telecom_full.json \
        --tree artifacts/trees/induced_iter9.json \
        --ovr-target reseat_sim_card \
        --out eval/reports/iteration12_telecom_full.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.eval import metrics
from sdt.experiments.common import score_with_cache
from sdt.experiments.soft_grower import SoftGrower
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.gain import class_weights
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.schemas import DecisionLabel, DecisionRecord, GateOutcome, Tree
from sdt.training.cache import ScoreCache
from sdt.training.combiner import cross_validate, fit_temperature
from sdt.training.interactions import augmented_features, mild_class_weights
from sdt.training.question_bank import QuestionBank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tree", type=Path, default=None,
                    help="existing tree artifact for leaf-posterior features")
    ap.add_argument("--ovr-target", default=None, help="label for the one OVR pass")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=900)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    frontier = GeminiFrontierClient()
    client = GeminiDecisionClient()
    bank = QuestionBank(str(args.bank))

    t0 = time.time()
    grower = SoftGrower(frontier, client, cache, examples, max_depth=4)
    try:
        tree = grower.induce(examples, "iter12")
        if args.ovr_target:
            target = DecisionLabel(args.ovr_target)
            grower.ovr_pass(examples, target)
    finally:
        cache.save()
    questions = list(grower.questions.values())
    bank.deposit(questions, "iter12")
    bank.save()
    tree_path = Path("artifacts/trees/induced_iter12_%s.json" % args.corpus.parent.name)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    tree_path.write_text(json.dumps(tree.to_dict(grower.questions), indent=2), encoding="utf-8")
    print("grown: %d nodes, %d questions (bank %d)" % (len(tree.nodes), len(questions), len(bank)))

    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    mild = mild_class_weights(examples, 0.35)
    w_mild = [mild[e.gold] for e in examples]
    wmap = class_weights(examples)
    w_alpha1 = [wmap[e.gold] for e in examples]

    prev_tree = None
    if args.tree and args.tree.exists():
        prev_tree = Tree.from_dict(json.loads(args.tree.read_text()))

    results = {}
    for tag, weights in (("alpha1", w_alpha1), ("mild", w_mild)):
        X, names = augmented_features(examples, questions, cache,
                                      tree=prev_tree, include_computed=True)
        oof, acc, f1 = cross_validate(X, y, weights, labels, iters=args.iters)
        T = fit_temperature(oof, y)
        records = []
        for z, ex in zip(oof, examples):
            m = max(z)
            expst = [math.exp((v - m) / T) for v in z]
            st = sum(expst)
            probs = [e / st for e in expst]
            order = sorted(range(len(labels)), key=lambda c: -probs[c])
            records.append(DecisionRecord(example_id=ex.example_id, tree_id="iter12", scores={},
                                          class_posterior={}, margin=probs[order[0]] - probs[order[1]],
                                          bottleneck_nodes=[], outcome=GateOutcome.AUTO_EXECUTED,
                                          predicted=labels[order[0]]))
        cov = metrics.coverage_at_precision(records, examples, 0.97)
        preds = [max(range(len(labels)), key=lambda c: z[c]) for z in oof]
        per_class = {labels[i].value: "%d/%d" % (
            sum(1 for yy, p in zip(y, preds) if yy == i and p == i),
            sum(1 for yy in y if yy == i)) for i in range(len(labels))}
        results[tag] = {"dims": len(names), "cv_accuracy": round(acc, 4),
                        "cv_macro_f1": round(f1, 4), "temperature": T,
                        "coverage_at_0.97": cov, "per_class_recall": per_class}
        print("%s: dims %d | acc %.4f | macro_f1 %.4f | cov@0.97 %s" % (
            tag, len(names), acc, f1, json.dumps(cov)), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "iteration12_fix1234",
        "questions": [q.text for q in questions],
        "results": results,
        "vs_previous": {"grow_macro_f1": 0.2101, "selected_macro_f1": 0.2183,
                         "coverage": "0.750 at 0.992"},
        "wall_seconds": round(time.time() - t0),
        "best_feature_set": "iter12",
    }, indent=2), encoding="utf-8")
    print("report ->", args.out)


if __name__ == "__main__":
    main()
