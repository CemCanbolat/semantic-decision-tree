"""Experiment B: free-type induction - the frontier picks noul, choice, or score.

The proposal prompt explains all three formats; parsed candidates keep their
type. The inducer grows a genuinely mixed tree: noul nodes split two ways,
choice nodes K ways, score nodes L ways. Same selection discipline as the
champion pipeline: class-weighted gain, validation veto, rarity discount.

    python3.11 -m sdt.experiments.exp_b_free \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --out eval/reports/expB_free_types.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from sdt.data.loader import load_examples
from sdt.decision_types import DecisionType, branch_names, columns_of
from sdt.decision_types import choice as mk_choice
from sdt.decision_types import noul as mk_noul
from sdt.decision_types import score as mk_score
from sdt.eval import metrics
from sdt.eval.baselines import run_b2_soft
from sdt.experiments.common import majority, score_with_cache, weighted_majority
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.gain import class_weights, weighted_entropy
from sdt.induction.propose import leakage_check
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.jev_client import JevClient
from sdt.schemas import DecisionLabel, Example, Question, ScoreVector, SdtError, TreeNode, Tree
from sdt.training.cache import ScoreCache
from sdt.training.combiner import build_features, cross_validate, fit_temperature
from sdt.training.question_bank import QuestionBank

FREE_SYSTEM = (
    "You write screening questions for a support adjudication decision model. "
    "Choose the best FORMAT per question:" + chr(10)
    + "NOUL: a yes/no question - format: NOUL: <question>" + chr(10)
    + "CHOICE: multiple choice, 2-6 options - format: CHOICE: <question> OPTIONS: opt1; opt2; opt3" + chr(10)
    + "SCORE: an ordered rubric, 3-5 levels lowest first - format: SCORE: <question> LEVELS: level1; level2; level3" + chr(10)
    + "Pick the format each question deserves - use CHOICE when the answer is a "
    "category, SCORE when it is a graded degree, NOUL when it is a simple condition. "
    "Questions must be positive-phrased, atomic, generalize to unseen tickets, "
    "and never reference classification labels. One question per line."
)


def parse_free(raw: str, provenance: str) -> List[Question]:
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            if s.upper().startswith("CHOICE:"):
                rest = s[7:]
                if "OPTIONS:" not in rest.upper():
                    continue
                qpart, opts = rest.split("OPTIONS:", 1) if "OPTIONS:" in rest else rest.split("options:", 1)
                options = {}
                for i, o in enumerate([o.strip() for o in opts.split(";") if o.strip()]):
                    options["o%d" % i] = o[:80]
                if 2 <= len(options) <= 6:
                    out.append(mk_choice(qpart.strip(), options, provenance))
            elif s.upper().startswith("SCORE:"):
                rest = s[6:]
                if "LEVELS:" not in rest.upper():
                    continue
                qpart, lvls = rest.split("LEVELS:", 1) if "LEVELS:" in rest else rest.split("levels:", 1)
                rubric = [l.strip() for l in lvls.split(";") if l.strip()]
                if 2 <= len(rubric) <= 5:
                    out.append(mk_score(qpart.strip(), rubric, provenance))
            else:
                q = s.split(":", 1)[1].strip() if s.upper().startswith("NOUL:") else s
                if len(q) > 8 and q.endswith("?") and not leakage_check(q):
                    out.append(mk_noul(q, provenance))
        except Exception:
            continue
    return out[:10]


class FreeInducer:
    def __init__(self, frontier, client, cache, corpus_texts, max_depth=3,
                 max_questions=12, min_branch=2, veto_delta=0.05, seed=17):
        import random
        self.frontier, self.client, self.cache = frontier, client, cache
        self.corpus_texts = corpus_texts
        self.max_depth, self.max_questions = max_depth, max_questions
        self.min_branch, self.veto_delta = min_branch, veto_delta
        self.rng = random.Random(seed)
        self.nodes: Dict[str, TreeNode] = {}
        self.questions: Dict[str, Question] = {}
        self.counter = 0
        self.frontier_calls = 0
        self.log: List[dict] = []

    def _next(self):
        self.counter += 1
        return "f%03d" % self.counter

    def _split_train_val(self, examples):
        by_class = {}
        for e in sorted(examples, key=lambda x: x.example_id):
            by_class.setdefault(e.gold, []).append(e)
        train, val = [], []
        for label in sorted(by_class, key=lambda l: l.value):
            exs = by_class[label]
            n_val = max(1, int(round(len(exs) * 0.3))) if len(exs) >= 2 else 0
            val.extend(exs[:n_val]); train.extend(exs[n_val:])
        return train, val

    def _propose(self, examples):
        counts = {}
        for e in examples:
            counts[e.gold] = counts.get(e.gold, 0) + 1
        top2 = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].value))[:2]
        pos = [e for e in examples if e.gold is top2[0][0]][:8]
        neg = [e for e in examples if e.gold is top2[1][0]][:8]
        block = lambda exs: chr(10).join("- " + e.text.replace(chr(10), " ")[:160] for e in exs)
        user = (chr(10).join([
            "GROUP A tickets (" + top2[0][0].value + "):", block(pos), "",
            "GROUP B tickets (" + top2[1][0].value + "):", block(neg), "",
            "Propose up to 8 questions in ANY of the three formats that separate "
            "GROUP A from GROUP B. One per line.",
        ]))
        raw = self.frontier.complete(FREE_SYSTEM, user)
        self.frontier_calls += 1
        return parse_free(raw, "expB")

    def _branch_of(self, q, ex):
        if q.decision_type is DecisionType.NOUL:
            p = self.cache.get(q, ex.example_id)
            return "yes" if (p if p is not None else 0.5) >= 0.5 else "no"
        cols = self.cache.get_columns(q, ex.example_id) or {}
        return max(cols.items(), key=lambda kv: kv[1])[0] if cols else "missing"

    def _gain(self, examples, q, wmap):
        branches = branch_names(q)
        groups = {b: [] for b in branches}
        for e in examples:
            groups[self._branch_of(q, e)].append(e)
        if min(len(g) for g in groups.values()) < self.min_branch:
            return None
        total_w = sum(wmap.get(e.gold, 1.0) for e in examples) or 1.0
        ig = weighted_entropy([e.gold for e in examples], wmap)
        for g in groups.values():
            w = sum(wmap.get(e.gold, 1.0) for e in g)
            ig -= (w / total_w) * weighted_entropy([e.gold for e in g], wmap)
        # certainty: mean (top1 - top2) over examples
        cert = 0.0
        for e in examples:
            if q.decision_type is DecisionType.NOUL:
                p = self.cache.get(q, e.example_id) or 0.5
                cert += abs(2 * p - 1)
            else:
                vals = sorted((self.cache.get_columns(q, e.example_id) or {}).values(),
                              reverse=True)
                cert += (vals[0] - vals[1]) if len(vals) > 1 else 1.0
        return ig * (cert / max(len(examples), 1)), groups

    def grow(self, examples, depth=0):
        self.counter += 1
        nid = "f%03d" % self.counter
        if depth >= self.max_depth or len(examples) < 4 * self.min_branch:
            self.nodes[nid] = TreeNode(nid, leaf_label=majority(examples))
            return nid
        if len(self.questions) >= self.max_questions:
            self.nodes[nid] = TreeNode(nid, leaf_label=majority(examples))
            return nid
        counts = {}
        for e in examples:
            counts[e.gold] = counts.get(e.gold, 0) + 1
        if max(counts.values()) / len(examples) >= 0.95:
            self.nodes[nid] = TreeNode(nid, leaf_label=majority(examples))
            return nid

        train, val = self._split_train_val(examples)
        cands = self._propose(train)
        if not cands:
            self.nodes[nid] = TreeNode(nid, leaf_label=majority(examples))
            return nid
        score_with_cache(self.client, train + val, cands, self.cache)
        wmap = class_weights(examples)
        best = None
        for q in cands:
            res = self._gain(train, q, wmap)
            if res is None:
                continue
            score, groups = res
            if score < 0.05:
                continue
            # validation veto: multi-leaf predictor accuracy
            if val:
                base = majority(examples)
                hits = wsum = 0.0
                for e in val:
                    w = wmap.get(e.gold, 1.0); wsum += w
                    g = [t for t in train if self._branch_of(q, t) == self._branch_of(q, e)]
                    pred = weighted_majority(g) if g else base
                    if pred is e.gold:
                        hits += w
                vac = hits / wsum if wsum else 0.0
                baseline = sum(wmap.get(e.gold, 1.0) for e in val if e.gold is base) / wsum
                if vac < baseline - self.veto_delta:
                    continue
            if best is None or score > best[0]:
                best = (score, q)
        if best is None:
            self.nodes[nid] = TreeNode(nid, leaf_label=majority(examples))
            return nid
        q = best[1]
        self.questions[q.qid] = q
        score_with_cache(self.client, examples, [q], self.cache)
        groups = {b: [] for b in branch_names(q)}
        for e in examples:
            groups[self._branch_of(q, e)].append(e)
        children = {}
        for b, members in groups.items():
            if not members:
                leaf = self._next()
                self.nodes[leaf] = TreeNode(leaf, leaf_label=majority(examples))
            else:
                leaf = self.grow(list(members), depth + 1)
            children[b] = leaf
        self.nodes[nid] = TreeNode(nid, question_qid=q.qid, branches=children)
        self.log.append({"depth": depth, "question": q.text,
                         "type": q.decision_type.value, "score": round(best[0], 4)})
        print("[expB] depth %d %s: %s (score %.3f)" % (
            depth, q.decision_type.value, q.text[:70], best[0]), flush=True)
        return nid


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
    inducer = FreeInducer(frontier, client, cache, [e.text for e in examples])
    try:
        root = inducer.grow(examples)
    finally:
        cache.save()
    tree = Tree("expB_free", root, inducer.nodes)
    questions = list(inducer.questions.values())
    # induction scores each question only on the examples that reach its node;
    # the flattened combiner needs every question against every example
    score_with_cache(client, examples, questions, cache)
    bank.deposit(questions, "expB")
    bank.save()
    print("tree: %d nodes %d leaves %d questions (%d frontier calls, %.0fs)" % (
        len(tree.nodes), len(tree.leaves()), len(questions),
        inducer.frontier_calls, time.time() - t0))

    class ColScorer(JevClient):
        def __init__(self, cache):
            self.cache, self.current = cache, None
        def score_batch(self, text, context, qs):
            scores = {}
            for q in qs:
                if q.decision_type.value == "noul":
                    scores[q.qid] = self.cache.get(q, self.current) or 0.5
                else:
                    for col, p in (self.cache.get_columns(q, self.current) or {}).items():
                        scores[q.qid + "::" + col] = p
            return ScoreVector(example_id=self.current, scores=scores)

    records = []
    scorer = ColScorer(cache)
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

    type_counts = {}
    for q in questions:
        type_counts[q.decision_type.value] = type_counts.get(q.decision_type.value, 0) + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "experiment": "B_free_types",
        "question_types": type_counts,
        "picked": inducer.log,
        "tree": {"accuracy": round(tree_acc, 4), "macro_f1": round(tree_f1, 4),
                 "n_leaves": len(tree.leaves()), "coverage_at_0.97": cov},
        "combiner": {"n_features": len(names), "cv_accuracy": round(cacc, 4),
                     "cv_macro_f1": round(cf1, 4), "temperature": T},
        "best_feature_set": "expB",
    }, indent=2), encoding="utf-8")
    print("tree: acc %.4f f1 %.4f | combiner: acc %.4f f1 %.4f | types %s" % (
        tree_acc, tree_f1, cacc, cf1, json.dumps(type_counts)))
    print("report ->", args.out)


if __name__ == "__main__":
    main()
