"""Ablation: one-shot question generation vs iterative tree induction.

Isolates what the ITERATIVE loop adds: the frontier sees every class with
labeled examples in ONE pass and writes the whole question set at once.
Those questions run through the identical downstream machinery (decision
model scoring, combiner, and a greedy tree restricted to the one-shot
pool). Compared against the iteratively-induced champion.

    python3.11 -m sdt.experiments.exp_oneshot \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --out eval/reports/exp_oneshot_retail.json
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
from sdt.experiments.exp_b_free import FREE_SYSTEM, parse_free
from sdt.experiments.soft_grower import SoftGrower
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.gain import class_weights
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.schemas import DecisionRecord, DecisionType, GateOutcome, Tree
from sdt.training.cache import ScoreCache
from sdt.training.combiner import cross_validate, fit_temperature
from sdt.training.interactions import augmented_features
from sdt.training.question_bank import QuestionBank

ONESHOT_USER_TMPL = """You are designing a support-ticket classifier as a batch of decision questions. You see EVERY category with labeled example tickets. Write the complete question set in ONE pass: up to 12 questions that together determine the correct category for any new ticket. Use whatever mix of formats is best.

%s

Propose up to 12 questions, one per line, in any of the three formats (NOUL: / CHOICE: ... OPTIONS: / SCORE: ... LEVELS:)."""


def build_oneshot_prompt(examples, per_class=3):
    by_class = {}
    for e in sorted(examples, key=lambda x: x.example_id):
        by_class.setdefault(e.gold.value, []).append(e)
    blocks = []
    for label in sorted(by_class):
        sample = by_class[label][:per_class]
        lines = chr(10).join("- " + s.text.replace(chr(10), " ")[:200] for s in sample)
        blocks.append("%s:" % label + chr(10) + lines)
    return ONESHOT_USER_TMPL % chr(10).join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=700)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    frontier = GeminiFrontierClient()
    client = GeminiDecisionClient()
    bank = QuestionBank(str(args.bank))

    t0 = time.time()
    raw = frontier.complete(FREE_SYSTEM, build_oneshot_prompt(examples))
    questions = parse_free(raw, "oneshot")
    print("one-shot generated %d questions:" % len(questions))
    for q in questions:
        print("  [%s] %s" % (q.decision_type.value, q.text[:80]))

    score_with_cache(client, examples, questions, cache)
    bank.deposit(questions, "oneshot")
    bank.save()

    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]

    # (a) flat combiner over the one-shot set (same machinery as champion)
    X, names = augmented_features(examples, questions, cache, tree=None, include_computed=True)
    oof, acc_flat, f1_flat = cross_validate(X, y, sample_w, labels, iters=args.iters)
    T = fit_temperature(oof, y)

    # (b) greedy tree RESTRICTED to the one-shot pool (no further proposals)
    grower = SoftGrower(frontier, client, cache, examples, max_depth=4, max_frontier_calls=0)
    grower._propose = lambda items: []  # freeze proposals: pool is fixed
    pool = list(questions)
    from sdt.jev_client import COMPUTED_SEEDS
    seen = {q.text.lower() for q in pool}
    for seed in COMPUTED_SEEDS:
        if seed.text.lower() not in seen:
            pool.append(seed)
    grower2 = _PoolGrower(frontier, client, cache, examples, pool)
    tree = grower2.induce(examples, "oneshot_tree")
    tree_questions = list(grower2.questions.values())
    from sdt.eval.baselines import run_b2_soft
    from sdt.jev_client import JevClient
    from sdt.schemas import ScoreVector

    class S(JevClient):
        def __init__(self, cache):
            self.cache, self.current = cache, None
        def score_batch(self, text, context, qs):
            scores = {}
            for q in qs:
                if q.decision_type.value == "noul":
                    p = self.cache.get(q, self.current)
                    scores[q.qid] = 0.5 if p is None else float(p)
                else:
                    for col, p in (self.cache.get_columns(q, self.current) or {}).items():
                        scores[q.qid + "::" + col] = float(p)
            return ScoreVector(example_id=self.current, scores=scores)

    scorer = S(cache)
    records = []
    for ex in examples:
        scorer.current = ex.example_id
        records.extend(run_b2_soft(scorer, [ex], tree=tree, questions=tree_questions))
    tree_acc = metrics.accuracy(records, examples)
    tree_f1 = metrics.macro_f1(records, examples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "one_shot_generation_ablation",
        "n_questions": len(questions),
        "questions": [q.text for q in questions],
        "flat_combiner": {"dims": len(names), "cv_accuracy": round(acc_flat, 4),
                          "cv_macro_f1": round(f1_flat, 4), "temperature": T},
        "greedy_tree_from_pool": {"accuracy": round(tree_acc, 4),
                                  "macro_f1": round(tree_f1, 4),
                                  "nodes": len(tree.nodes), "leaves": len(tree.leaves()),
                                  "questions_used": len(tree_questions)},
        "vs": {"champion_iterative_cv": 0.6453, "champion_holdout": 0.7555,
               "b1_zero_shot": 0.5566, "b1_few_shot": 0.5432},
        "wall_seconds": round(time.time() - t0),
        "best_feature_set": "oneshot",
    }, indent=2), encoding="utf-8")
    print()
    print("one-shot flat combiner: acc %.4f macro_f1 %.4f" % (acc_flat, f1_flat))
    print("greedy tree from pool:  acc %.4f macro_f1 %.4f (%d questions used)" % (
        tree_acc, tree_f1, len(tree_questions)))
    print("report ->", args.out)


class _PoolGrower(SoftGrower):
    """Growth restricted to a fixed question pool: no frontier proposals."""

    def __init__(self, frontier, client, cache, examples, pool):
        super().__init__(frontier, client, cache, examples)
        self.pool = list(pool)

    def _propose(self, items):
        return list(self.pool)


if __name__ == "__main__":
    main()
