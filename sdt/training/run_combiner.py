"""Iteration 3 CLI: leaf relabeling + combiner on cached p-vectors. Zero API calls.

    python3.11 -m sdt.training.run_combiner \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --tree artifacts/trees/induced_v3.json \
        --extra-tree artifacts/trees/induced_v1.json \
        --cache cache/scores.json \
        --out eval/reports/iteration3_combiner.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.eval import metrics
from sdt.eval.baselines import run_b2_soft
from sdt.induction.gain import class_weights
from sdt.jev_client import JevClient
from sdt.schemas import DecisionRecord, GateOutcome, ScoreVector, Tree
from sdt.training.cache import ScoreCache
from sdt.training.combiner import (
    SoftmaxModel, build_features, cross_validate, fit_temperature,
    relabel_tree_leaves,
)


class CacheScorer(JevClient):
    def __init__(self, cache):
        self.cache = cache
        self.current = None

    def score_batch(self, text, context, qs):
        scores = {}
        for q in qs:
            p = self.cache.get(q, self.current)
            scores[q.qid] = p if p is not None else 0.5
        return ScoreVector(example_id=self.current, scores=scores)


def coverage_from_oof(oof_logits, y, labels, temperature, examples):
    records = []
    for z, yi, ex in zip(oof_logits, y, examples):
        m = max(z)
        expst = [math.exp((v - m) / temperature) for v in z]
        st = sum(expst)
        probs = [e / st for e in expst]
        order = sorted(range(len(labels)), key=lambda c: -probs[c])
        margin = probs[order[0]] - probs[order[1]] if len(order) > 1 else 1.0
        records.append(DecisionRecord(
            example_id=ex.example_id, tree_id="combiner", scores={},
            class_posterior={labels[c].value: round(probs[c], 4) for c in order[:3]},
            margin=margin, bottleneck_nodes=[],
            outcome=GateOutcome.AUTO_EXECUTED if margin > 0.60 else GateOutcome.ESCALATED_MARGIN,
            predicted=labels[order[0]],
        ))
    return metrics.coverage_at_precision(records, examples, 0.97), records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--extra-tree", type=Path, default=None)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--relabel-alpha", type=float, default=None,
                    help="experimental: per-leaf power weighting for relabeling. "
                         "WARNING: any alpha > 1 means the rarest class present "
                         "always wins the leaf (refuted 2026-11-14: acc 0.64 -> 0.12). "
                         "Omit to skip relabeling.")
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    label_index = {l: i for i, l in enumerate(labels)}
    y = [label_index[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]

    tree, questions = Tree.from_dict(json.loads(args.tree.read_text()))
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_examples": len(examples), "labels": [l.value for l in labels],
              "leaf_relabel": {}, "feature_sets": {}}

    # 1) post-hoc leaf relabeling (path scores only - always cached)
    scorer = CacheScorer(cache)
    report["leaf_relabel"] = {"skipped": "refuted: any alpha > 1 gives the rarest "
                                         "class every leaf; rare classes must earn "
                                         "influence via features (the combiner)"}
    if args.relabel_alpha is not None:
        base_records = []
        for ex in examples:
            scorer.current = ex.example_id
            base_records.extend(run_b2_soft(scorer, [ex], tree=tree, questions=questions))
        relabeled, changes = relabel_tree_leaves(tree, questions, examples, cache,
                                                 alpha=args.relabel_alpha)
        rel_records = []
        for ex in examples:
            scorer.current = ex.example_id
            rel_records.extend(run_b2_soft(scorer, [ex], tree=relabeled, questions=questions))
        report["leaf_relabel"] = {
            "alpha": args.relabel_alpha, "changes": changes,
            "before": {"accuracy": round(metrics.accuracy(base_records, examples), 4),
                       "macro_f1": round(metrics.macro_f1(base_records, examples), 4)},
            "after": {"accuracy": round(metrics.accuracy(rel_records, examples), 4),
                      "macro_f1": round(metrics.macro_f1(rel_records, examples), 4)},
        }
        print("leaf relabel (alpha %.2f): %d changes | acc %.4f -> %.4f" % (
            args.relabel_alpha, len(changes),
            report["leaf_relabel"]["before"]["accuracy"],
            report["leaf_relabel"]["after"]["accuracy"]))

    # 2) combiner feature sets (tolerant of missing cache)
    base_name = args.tree.stem
    feature_sets = {base_name + "_plus_computed": questions}
    if args.extra_tree:
        _, q2 = Tree.from_dict(json.loads(args.extra_tree.read_text()))
        seen = {q.text for q in questions}
        feature_sets[base_name + "_plus_extra_plus_computed"] = questions + [q for q in q2 if q.text not in seen]

    best_name, best_f1 = None, -1.0
    for name, sem in feature_sets.items():
        try:
            X, names = build_features(examples, sem, cache, include_computed=True)
        except ValueError as e:
            print("%-30s SKIPPED (%s; run sdt.training.backfill_cache)" % (name, e))
            report["feature_sets"][name] = {"skipped": str(e)}
            continue
        oof_logits, acc, f1 = cross_validate(X, y, sample_w, labels)
        T = fit_temperature(oof_logits, y)
        cov97, _ = coverage_from_oof(oof_logits, y, labels, T, examples)
        report["feature_sets"][name] = {
            "n_features": len(names), "cv_accuracy": round(acc, 4),
            "cv_macro_f1": round(f1, 4), "temperature": T,
            "coverage_at_0.97": cov97,
        }
        print("%-30s features %2d | cv acc %.4f | cv macro_f1 %.4f | T %.2f | cov@0.97 %s" % (
            name, len(names), acc, f1, T, json.dumps(cov97)))
        if f1 > best_f1:
            best_name, best_f1 = name, f1

    # 3) final model on the best usable feature set
    if best_name is not None:
        sem = feature_sets[best_name]
        X, names = build_features(examples, sem, cache, include_computed=True)
        model = SoftmaxModel(labels, len(names)).fit(X, y, sample_w)
        model_path = args.out.with_name("combiner_model.json")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps({
            "labels": [l.value for l in labels], "features": names,
            "feature_set": best_name, "weights": model.W, "bias": model.b,
        }, indent=2), encoding="utf-8")
        report["best_feature_set"] = best_name
        report["model_path"] = str(model_path)
        print("final model ->", model_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("report ->", args.out)


if __name__ == "__main__":
    main()
