"""Iteration 9: the full upgrade kit on the FULL corpus - no CV fragmentation.

Iteration 8's mechanisms (soft fractional routing, gain-ratio with both
lopsided guards, alpha-weighted entropy, computed seeds, free types) were
starved by nested CV: each fold grew on ~64 examples. Here the same grower
runs on all 114, exactly like iteration 7's experiment B. No data-driven
pruning; only structural collapse of uniform-label subtrees.

Judged the established way: tree soft-routing numbers (in-sample, labeled
as such) for shape, and the combiner's cross-validated macro-F1 as the
honest comparison - the protocol in place since iteration 3.

    python3.11 -m sdt.experiments.run_iteration9 \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --out eval/reports/iteration9_full_corpus.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict

from sdt.data.loader import load_examples
from sdt.eval import metrics
from sdt.eval.baselines import run_b2_soft
from sdt.experiments.common import score_with_cache
from sdt.experiments.soft_grower import SoftGrower
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.gain import class_weights
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.jev_client import JevClient
from sdt.routing.soft_tree import route
from sdt.schemas import DecisionRecord, DecisionType, GateOutcome, ScoreVector, TreeNode
from sdt.training.cache import ScoreCache
from sdt.training.combiner import build_features, cross_validate, fit_temperature
from sdt.training.question_bank import QuestionBank


def collapse_uniform(tree):
    """Structural prune only: collapse subtrees whose leaves share one label."""
    pruned = 0
    labels = {}

    def uniform(nid):
        if nid in labels:
            return labels[nid]
        node = tree.nodes[nid]
        if node.is_leaf:
            labels[nid] = node.leaf_label
            return node.leaf_label
        child = [uniform(c) for c in node.branches.values()]
        if all(c is not None for c in child) and len(set(child)) == 1:
            labels[nid] = child[0]
            return child[0]
        labels[nid] = None
        return None

    def collapse(nid):
        nonlocal pruned
        node = tree.nodes[nid]
        if node.is_leaf:
            return
        for c in node.branches.values():
            collapse(c)
        if uniform(nid) is not None:
            tree.nodes[nid] = TreeNode(nid, leaf_label=uniform(nid))
            pruned += 1

    collapse(tree.root)
    return pruned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    frontier = GeminiFrontierClient()
    client = GeminiDecisionClient()
    bank = QuestionBank(str(args.bank))

    t0 = time.time()
    grower = SoftGrower(frontier, client, cache, examples, max_depth=args.max_depth)
    try:
        tree = grower.induce(examples, "iter9_full")
    finally:
        cache.save()
    pruned = collapse_uniform(tree)
    questions = list(grower.questions.values())
    print("grown: %d nodes | %d questions | uniform-collapsed %d" % (
        len(tree.nodes), len(questions), pruned))

    # full matrix for routing + combiner
    score_with_cache(client, examples, questions, cache)
    bank.deposit(questions, "iter9")
    bank.save()

    # tree soft-routing (in-sample, labeled as such)
    class ColScorer(JevClient):
        def __init__(self, cache):
            self.cache, self.current = cache, None
        def score_batch(self, text, context, qs):
            scores = {}
            for q in qs:
                if q.decision_type is DecisionType.NOUL:
                    p = self.cache.get(q, self.current)
                    scores[q.qid] = 0.5 if p is None else p
                else:
                    for col, p in (self.cache.get_columns(q, self.current) or {}).items():
                        scores[q.qid + "::" + col] = p
            return ScoreVector(example_id=self.current, scores=scores)

    scorer = ColScorer(cache)
    records = []
    for ex in examples:
        scorer.current = ex.example_id
        records.extend(run_b2_soft(scorer, [ex], tree=tree, questions=questions))
    tree_acc = metrics.accuracy(records, examples)
    tree_f1 = metrics.macro_f1(records, examples)
    tree_cov = metrics.coverage_at_precision(records, examples, 0.97)

    # combiner CV - the honest number since iteration 3
    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]
    X, names = build_features(examples, questions, cache, include_computed=True)
    oof, cacc, cf1 = cross_validate(X, y, sample_w, labels, iters=900)
    T = fit_temperature(oof, y)

    depths = []
    parents = {}
    for nid, n in tree.nodes.items():
        if n.branches:
            for c in n.branches.values():
                parents[c] = nid
    for leaf in tree.leaves():
        d, cur = 0, leaf
        while cur in parents:
            cur = parents[cur]
            d += 1
        depths.append(d)

    type_counts = {}
    for q in questions:
        type_counts[q.decision_type.value] = type_counts.get(q.decision_type.value, 0) + 1
    out_tree = Path("artifacts/trees/induced_iter9.json")
    out_tree.parent.mkdir(parents=True, exist_ok=True)
    out_tree.write_text(json.dumps(tree.to_dict(grower.questions), indent=2), encoding="utf-8")

    report = {
        "experiment": "iteration9_full_corpus_no_cv",
        "tree_insample": {"accuracy": round(tree_acc, 4), "macro_f1": round(tree_f1, 4),
                          "coverage_at_0.97": tree_cov, "nodes": len(tree.nodes),
                          "leaves": len(tree.leaves()), "mean_depth": round(sum(depths) / len(depths), 2),
                          "max_depth": max(depths), "types": type_counts,
                          "uniform_pruned": pruned},
        "combiner_cv": {"n_features": len(names), "cv_accuracy": round(cacc, 4),
                        "cv_macro_f1": round(cf1, 4), "temperature": T},
        "vs": {"champion_bank_selected": 0.5903, "iter7_B_tree_insample": 0.7018,
               "iter8_deep_oof": 0.3154},
        "best_feature_set": "iter9",
        "wall_seconds": round(time.time() - t0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("tree (in-sample): acc %.4f f1 %.4f | depth mean %.1f max %d | cov@0.97 %s" % (
        tree_acc, tree_f1, report["tree_insample"]["mean_depth"], max(depths), json.dumps(tree_cov)))
    print("combiner CV: acc %.4f macro_f1 %.4f T %.2f" % (cacc, cf1, T))
    print("report ->", args.out, "| tree ->", out_tree)


if __name__ == "__main__":
    main()
