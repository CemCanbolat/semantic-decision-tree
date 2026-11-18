"""Iteration 8 CLI: deep soft trees, gain-ratio guarded, pruned leak-free.

Protocol: 5-fold nested cross-validation. Within each fold the tree is
GROWN on a 70% split of the training folds and PRUNED on the other 30%;
the test fold only scores the finished tree. No decision ever touches its
own test data. A final tree is grown+pruned on all data for deployment.

    python3.11 -m sdt.experiments.run_iteration8 \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --out eval/reports/iteration8_deep_soft.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from sdt.data.loader import load_examples
from sdt.experiments.soft_grower import SoftGrower, prune_by_validation, tree_macro_f1
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.routing.soft_tree import route
from sdt.schemas import DecisionLabel, ScoreVector, DecisionType
from sdt.training.cache import ScoreCache
from sdt.training.combiner import stratified_folds
from sdt.training.question_bank import QuestionBank


def stratified_split(examples, fraction_first=0.7):
    by_class = {}
    for e in sorted(examples, key=lambda x: x.example_id):
        by_class.setdefault(e.gold, []).append(e)
    first, second = [], []
    for label in sorted(by_class, key=lambda l: l.value):
        exs = by_class[label]
        k = max(1, int(round(len(exs) * fraction_first)))
        first.extend(exs[:k]); second.extend(exs[k:])
    return first, second


def tree_scores(tree, questions, cache, examples):
    by_qid = {q.qid: q for q in questions}
    correct = total = 0
    per_class: Dict[str, List[int]] = {}
    for ex in examples:
        scores = {}
        for q in questions:
            if q.decision_type is DecisionType.NOUL:
                p = cache.get(q, ex.example_id)
                scores[q.qid] = 0.5 if p is None else p
            else:
                for col, p in (cache.get_columns(q, ex.example_id) or {}).items():
                    scores[q.qid + "::" + col] = p
        r = route(tree, ScoreVector(example_id=ex.example_id, scores=scores), by_qid)
        pred = max(r.class_posterior.items(), key=lambda kv: kv[1])[0]
        total += 1
        if pred is ex.gold:
            correct += 1
        slot = per_class.setdefault(ex.gold.value, [0, 0])
        slot[1] += 1
        if pred is ex.gold:
            slot[0] += 1
    return correct / total, per_class


def run_fold(frontier, client, cache, train, test, fold_id):
    grow_split, prune_split = stratified_split(train, 0.7)
    grower = SoftGrower(frontier, client, cache, grow_split)
    raw = grower.induce(grow_split, "iter8_fold%d_raw" % fold_id)
    questions = list(grower.questions.values())
    # growth scores questions on the grow split only; pruning and testing
    # route OTHER examples through the tree - backfill the full matrix
    from sdt.experiments.common import score_with_cache
    score_with_cache(client, prune_split + test, questions, cache)
    pruned = prune_by_validation(raw, questions, cache, prune_split)
    return pruned, questions, grower


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    frontier = GeminiFrontierClient()
    client = GeminiDecisionClient()
    bank = QuestionBank(str(args.bank))

    t0 = time.time()
    fold_of = stratified_folds(examples, args.folds)
    oof_correct = 0
    oof_per_class: Dict[str, List[int]] = {}
    fold_f1s = []
    all_questions = {}
    for f in range(args.folds):
        train = [e for e, ff in zip(examples, fold_of) if ff != f]
        test = [e for e, ff in zip(examples, fold_of) if ff == f]
        pruned, questions, grower = run_fold(frontier, client, cache, train, test, f)
        acc, per_class = tree_scores(pruned, questions, cache, test)
        f1 = tree_macro_f1(pruned, questions, cache, test)
        fold_f1s.append(round(f1, 4))
        oof_correct += int(round(acc * len(test)))
        for k, (c, t) in per_class.items():
            slot = oof_per_class.setdefault(k, [0, 0])
            slot[0] += c; slot[1] += t
        for q in questions:
            all_questions[q.qid] = q
        print("[fold %d] raw->pruned %d nodes | test acc %.4f macro_f1 %.4f (%d frontier calls)" % (
            f, len(pruned.nodes), acc, f1, grower.frontier_calls), flush=True)
        cache.save()

    oof_acc = oof_correct / len(examples)
    oof_recall = {k: "%d/%d" % (c, t) for k, (c, t) in sorted(oof_per_class.items())}

    # final deployment tree on all data; pruned on a 30% holdout (pessimistic)
    grow_all, prune_all = stratified_split(examples, 0.7)
    grower = SoftGrower(frontier, client, cache, grow_all)
    raw = grower.induce(grow_all, "iter8_final_raw")
    final_questions = list(grower.questions.values())
    from sdt.experiments.common import score_with_cache
    score_with_cache(client, examples, final_questions, cache)
    final = prune_by_validation(raw, final_questions, cache, prune_all)
    cache.save()
    bank.deposit(final_questions, "iter8")
    bank.save()
    out_tree = Path("artifacts/trees/induced_iter8.json")
    out_tree.parent.mkdir(parents=True, exist_ok=True)
    out_tree.write_text(json.dumps(final.to_dict(grower.questions), indent=2), encoding="utf-8")

    type_counts = {}
    for q in final_questions:
        type_counts[q.decision_type.value] = type_counts.get(q.decision_type.value, 0) + 1
    report = {
        "experiment": "iteration8_deep_soft_pruned",
        "protocol": "5-fold nested CV: grow on 70% of train folds, prune on 30%, test fold only scores",
        "oof_accuracy": round(oof_acc, 4),
        "oof_macro_f1_mean": round(sum(fold_f1s) / len(fold_f1s), 4),
        "fold_macro_f1": fold_f1s,
        "oof_per_class_recall": oof_recall,
        "final_tree": {"nodes": len(final.nodes), "leaves": len(final.leaves()),
                       "questions": len(final_questions), "types": type_counts,
                       "max_depth": max(len(_path(final, n)) for n in final.leaves())},
        "vs": {"champion_bank_selected": 0.5903, "iter7_B_tree_insample": 0.7018,
               "iter7_B_tree_macrosample": 0.5277},
        "best_feature_set": "iter8",
        "wall_seconds": round(time.time() - t0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("OOF: acc %.4f | macro_f1 %.4f | folds %s" % (oof_acc, report["oof_macro_f1_mean"], fold_f1s))
    print("per-class:", json.dumps(oof_recall))
    print("final tree: %d nodes %d leaves | %s" % (
        len(final.nodes), len(final.leaves()), json.dumps(type_counts)))
    print("report ->", args.out, "| tree ->", out_tree)


def _path(tree, leaf_id):
    node = tree.nodes[leaf_id]
    path = [leaf_id]
    # walk up via parent search
    parents = {}
    for nid, n in tree.nodes.items():
        if n.branches:
            for c in n.branches.values():
                parents[c] = nid
    cur = leaf_id
    while cur in parents:
        cur = parents[cur]
        path.append(cur)
    return path


if __name__ == "__main__":
    main()
