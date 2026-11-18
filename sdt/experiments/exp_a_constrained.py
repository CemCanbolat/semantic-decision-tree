"""Experiment A: exactly ONE choice question + ONE score question, no criteria.

The choice question's options are the outcome classes with frontier-written
descriptions; the score question is a frontier-designed ordered rubric.
The tree is genuinely non-binary: the choice node branches per class.
Evaluation: multi-way tree soft routing + flat combiner on all columns.

    python3.11 -m sdt.experiments.exp_a_constrained \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --out eval/reports/expA_choice_score.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.decision_types import choice as mk_choice, score as mk_score
from sdt.eval import metrics
from sdt.eval.baselines import run_b2_soft
from sdt.experiments.common import (build_multiway_tree, majority, parse_json_object,
                                    score_with_cache)
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.gain import class_weights
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.jev_client import JevClient
from sdt.routing.soft_tree import route
from sdt.schemas import DecisionRecord, GateOutcome, ScoreVector
from sdt.training.cache import ScoreCache
from sdt.training.combiner import build_features, cross_validate, fit_temperature
from sdt.training.question_bank import QuestionBank


def class_descriptions(frontier, examples):
    groups = {}
    for e in examples:
        groups.setdefault(e.gold.value, []).append(e.text.replace(chr(10), " ")[:160])
    listing = chr(10).join(
        "%s:" % g + chr(10) + chr(10).join("  - " + t for t in ts[:3])
        for g, ts in sorted(groups.items()))
    system = ("You write one-line descriptions of support ticket categories for a "
              "classification decision model. Be concrete about what distinguishes "
              "each category from its nearest neighbor.")
    user = ("Categories with example tickets:" + chr(10) + listing + chr(10) + chr(10)
            + "Return a JSON object mapping each category name to a one-sentence "
              "description (max 20 words). JSON only.")
    from sdt.experiments.common import complete_json
    fallback = {g: g.replace("_", " ") for g in groups}
    return complete_json(frontier, system, user, fallback=fallback)


def design_rubric(frontier, examples):
    sample = chr(10).join("- " + e.text.replace(chr(10), " ")[:160] for e in examples[:8])
    system = ("You design ordered grading rubrics for a decision model. Levels run "
              "lowest to highest. The dimension must be observable from the ticket "
              "text and useful for routing it.")
    user = ("Tickets:" + chr(10) + sample + chr(10) + chr(10)
            + "Design ONE rubric of 3 to 5 levels that grades these tickets on a "
              "single useful dimension (for example clarity of the request). Return "
              "a JSON array of level descriptions, lowest first, each under 15 words. JSON only.")
    from sdt.experiments.common import complete_json
    fallback = ["Request is vague or incomplete",
                "Request is partially specified",
                "Request is explicit and complete"]
    return complete_json(frontier, system, user, fallback=fallback)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    frontier = GeminiFrontierClient()
    client = GeminiDecisionClient()
    bank = QuestionBank(str(args.bank))

    t0 = time.time()
    classes = sorted({e.gold.value for e in examples})
    desc = class_descriptions(frontier, examples)
    desc = {c: desc.get(c, c.replace("_", " ")) for c in classes}
    rubric = design_rubric(frontier, examples)
    if not isinstance(rubric, list) or len(rubric) < 2:
        print("[warn] unusable rubric %r; using fallback" % (rubric,))
        rubric = ["Request is vague or incomplete",
                  "Request is partially specified",
                  "Request is explicit and complete"]

    q_choice = mk_choice("Which category best matches this ticket's request?", desc,
                         provenance="expA:choice")
    q_score = mk_score("Grade this ticket on the rubric.", rubric, provenance="expA:score")
    print("choice options:", list(desc))
    print("rubric:", rubric)

    scored = score_with_cache(client, examples, [q_choice, q_score], cache)
    print("scored %d examples (%.0fs)" % (scored, time.time() - t0))

    tree, stats = build_multiway_tree(q_choice, examples, cache, "expA",
                                      second_q=q_score)
    bank.deposit([q_choice, q_score], "expA")
    bank.save()

    # tree soft routing from cached columns
    class ColScorer(JevClient):
        def __init__(self, cache, questions):
            self.cache, self.questions, self.current = cache, questions, None
        def score_batch(self, text, context, qs):
            scores = {}
            for q in qs:
                if q.decision_type.value == "noul":
                    scores[q.qid] = self.cache.get(q, self.current) or 0.5
                else:
                    for col, p in (self.cache.get_columns(q, self.current) or {}).items():
                        scores[q.qid + "::" + col] = p
            return ScoreVector(example_id=self.current, scores=scores)

    questions = [q_choice, q_score]
    records = []
    scorer = ColScorer(cache, questions)
    for ex in examples:
        scorer.current = ex.example_id
        records.extend(run_b2_soft(scorer, [ex], tree=tree, questions=questions))
    tree_acc = metrics.accuracy(records, examples)
    tree_f1 = metrics.macro_f1(records, examples)
    cov = metrics.coverage_at_precision(records, examples, 0.97)

    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]
    X, names = build_features(examples, questions, cache, include_computed=True)
    oof, cacc, cf1 = cross_validate(X, y, sample_w, labels, iters=700)
    T = fit_temperature(oof, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "A_constrained_choice_score",
        "choice_options": desc, "rubric": rubric, "branch_sizes": stats["branch_sizes"],
        "tree": {"accuracy": round(tree_acc, 4), "macro_f1": round(tree_f1, 4),
                 "n_leaves": len(tree.leaves()), "coverage_at_0.97": cov},
        "combiner": {"n_features": len(names), "cv_accuracy": round(cacc, 4),
                     "cv_macro_f1": round(cf1, 4), "temperature": T},
        "best_feature_set": "expA",
        "no_criteria": True,
    }, indent=2), encoding="utf-8")
    print("tree: acc %.4f f1 %.4f leaves %d | combiner: acc %.4f f1 %.4f T %.2f" % (
        tree_acc, tree_f1, len(tree.leaves()), cacc, cf1, T))
    print("report ->", args.out)


if __name__ == "__main__":
    main()
