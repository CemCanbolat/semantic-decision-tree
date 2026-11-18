"""Grow a tree by contrastive induction.

At each node: pick the top-2 labels, ask the frontier LLM for questions that
separate them, score every candidate against every node example in scorer
batches (cache-aware), split on the certainty-weighted gain winner, recurse.
Post-pass prunes subtrees whose leaves share one majority label.

The frontier LLM is the expensive System 2 oracle (called once per node);
the scorer is the cheap System 1 engine (one batch call per example per
candidate set). Nothing here assumes network - both are interfaces.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sdt.induction.gain import SplitInfo, class_weights, rank_questions, weighted_entropy
from sdt.induction.propose import (build_ovr_prompt, build_proposal_prompt, contrastive_pair,
                                    parse_questions, rare_fraction)
from sdt.jev_client import JevClient
from sdt.jev_client import COMPUTED_SEEDS
from sdt.llm_client import FrontierClient
from sdt.schemas import DecisionLabel, Example, Question, Tree, TreeNode
from sdt.training.cache import ScoreCache


@dataclass
class InductionStats:
    n_nodes: int = 0
    n_leaves: int = 0
    n_questions: int = 0
    frontier_calls: int = 0
    scorer_calls: int = 0
    cache_hits: int = 0
    pruned_subtrees: int = 0
    per_node: List[dict] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)


def _majority(examples: List[Example]) -> DecisionLabel:
    counts: Dict[DecisionLabel, int] = {}
    for e in examples:
        counts[e.gold] = counts.get(e.gold, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0].value))[0]


def _sample(examples: List[Example], label: DecisionLabel, k: int,
            rng: Optional[random.Random] = None) -> List[Example]:
    pool = [e for e in examples if e.gold is label]
    if rng is not None and len(pool) > k:
        return sorted(rng.sample(pool, k), key=lambda e: e.example_id)
    return pool[:k]


class TreeInducer:
    def __init__(
        self,
        frontier: FrontierClient,
        scorer: JevClient,
        max_depth: int = 4,
        min_child: int = 5,
        min_node: Optional[int] = None,
        purity: float = 0.95,
        max_questions: int = 31,
        min_gain: float = 0.05,
        sample_per_side: int = 8,
        val_fraction: float = 0.3,
        minority_threshold: float = 0.10,
        rare_min_df: int = 3,
        rare_penalty: float = 1.0,
        veto_delta: float = 0.05,
        seed: int = 7,
        corpus_texts: Optional[List[str]] = None,
        cache: Optional[ScoreCache] = None,
    ) -> None:
        self.frontier = frontier
        self.scorer = scorer
        self.max_depth = max_depth
        self.min_child = min_child
        self.min_node = min_node if min_node is not None else 2 * min_child
        self.purity = purity
        self.max_questions = max_questions
        self.min_gain = min_gain
        self.sample_per_side = sample_per_side
        self.val_fraction = val_fraction
        self.minority_threshold = minority_threshold
        self.rare_min_df = rare_min_df
        self.rare_penalty = rare_penalty
        self.veto_delta = veto_delta
        self.seed = seed
        self._rng = random.Random(seed)
        self.corpus_texts = corpus_texts or []
        self.cache = cache
        self.stats = InductionStats()
        self._questions: Dict[str, Question] = {}
        self._nodes: Dict[str, TreeNode] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return "n%03d" % self._counter

    def _score_candidates(self, examples: List[Example], candidates: List[Question]) -> Dict[str, List[float]]:
        """p-vectors for every candidate over every node example; cache-aware."""
        by_example: Dict[str, Dict[str, float]] = {e.example_id: {} for e in examples}
        to_score = []
        for e in examples:
            missing = [q for q in candidates if self.cache is None or self.cache.get(q, e.example_id) is None]
            if missing:
                to_score.append((e, missing))
            for q in candidates:
                if self.cache is not None:
                    cached = self.cache.get(q, e.example_id)
                    if cached is not None:
                        by_example[e.example_id][q.qid] = cached
                        self.stats.cache_hits += 1
        for e, missing in to_score:
            sv = self.scorer.score_batch(e.text, e.context, missing)
            self.stats.scorer_calls += 1
            for q in missing:
                p = sv.scores[q.qid]
                by_example[e.example_id][q.qid] = p
                if self.cache is not None:
                    self.cache.put(q, e.example_id, p)
        return {q.qid: [by_example[e.example_id][q.qid] for e in examples] for q in candidates}

    def _split_train_val(self, examples: List[Example]) -> Tuple[List[Example], List[Example]]:
        """Deterministic stratified split: every class with >=2 examples keeps
        at least one example on each side."""
        by_class: Dict[DecisionLabel, List[Example]] = {}
        for e in sorted(examples, key=lambda x: x.example_id):
            by_class.setdefault(e.gold, []).append(e)
        train, val = [], []
        for label in sorted(by_class, key=lambda l: l.value):
            exs = by_class[label]
            n_val = max(1, int(round(len(exs) * self.val_fraction))) if len(exs) >= 2 else 0
            val.extend(exs[:n_val])
            train.extend(exs[n_val:])
        return train, val

    def _propose(self, examples: List[Example]) -> List[Question]:
        """Pairwise prompt for the top-2 labels PLUS one-vs-rest prompts for
        every minority class (support below minority_threshold, at least 2):
        rare classes otherwise never get a question proposed for them."""
        counts: Dict[DecisionLabel, int] = {}
        for e in examples:
            counts[e.gold] = counts.get(e.gold, 0) + 1
        prompts = []
        pair = contrastive_pair(examples)
        if pair is not None:
            pos_label, neg_label = pair
            prompts.append(("pair:" + pos_label.value + "_vs_" + neg_label.value,
                            build_proposal_prompt(_sample(examples, pos_label, self.sample_per_side, self._rng),
                                                  _sample(examples, neg_label, self.sample_per_side, self._rng),
                                                  pos_label, neg_label)))
        for label, c in sorted(counts.items(), key=lambda kv: kv[0].value):
            if c < self.minority_threshold * len(examples) and c >= 2 and (pair is None or label not in pair):
                others = [e for e in examples if e.gold is not label]
                prompts.append(("ovr:" + label.value,
                                build_ovr_prompt(_sample(examples, label, self.sample_per_side, self._rng),
                                                 label, others[:self.sample_per_side])))
        # exact features first: computed seeds compete with frontier proposals
        # on identical terms (gain, validation veto, rarity discount)
        candidates: List[Question] = list(COMPUTED_SEEDS)
        seen = set()
        for q in candidates:
            seen.add(q.text.lower())
        for source, (system, user) in prompts:
            raw = self.frontier.complete(system, user)
            self.stats.frontier_calls += 1
            for q in parse_questions(raw, "induced:" + source):
                if q.text.lower() in seen:
                    continue
                seen.add(q.text.lower())
                candidates.append(q)
        return candidates

    def _val_selection_score(self, train: List[Example], val: List[Example],
                             info_by_qid, candidates: List[Question],
                             wmap: Dict[DecisionLabel, float]) -> Dict[str, float]:
        """Class-weighted validation accuracy of the two-leaf predictor each
        candidate induces, times the rarity multiplier. Splitting must beat
        the no-split majority baseline to be worth a node."""
        if not val:
            return {info.qid: info.score for info in info_by_qid}
        val_scores = self._score_candidates(val, candidates)
        node_majority = _majority(train)
        baseline = sum(wmap.get(e.gold, 1.0) for e in val if e.gold is node_majority) / sum(
            wmap.get(e.gold, 1.0) for e in val)
        out: Dict[str, float] = {}
        for q in candidates:
            info = info_by_qid.get(q.qid)
            if info is None:
                continue
            ps = val_scores.get(q.qid)
            if ps is None:
                continue
            yes_majority = _majority([e for e in train if e.example_id in set(info.yes_ids)] or train)
            no_majority = _majority([e for e in train if e.example_id in set(info.no_ids)] or train)
            correct_w = 0.0
            total_w = 0.0
            for e, p in zip(val, ps):
                w = wmap.get(e.gold, 1.0)
                total_w += w
                pred = yes_majority if p >= 0.5 else no_majority
                if pred is e.gold:
                    correct_w += w
            acc = correct_w / total_w if total_w else 0.0
            rf = rare_fraction(q.text, self.corpus_texts, self.rare_min_df) if self.corpus_texts else 0.0
            out[q.qid] = round(acc * (1.0 - self.rare_penalty * rf), 6)
        return out

    def _grow(self, examples: List[Example], depth: int) -> str:
        self.stats.n_nodes += 1
        counts: Dict[DecisionLabel, int] = {}
        for e in examples:
            counts[e.gold] = counts.get(e.gold, 0) + 1
        top_frac = max(counts.values()) / len(examples)
        leaf_id = self._next_id()

        if depth >= self.max_depth or len(examples) < self.min_node or top_frac >= self.purity:
            self._nodes[leaf_id] = TreeNode(leaf_id, leaf_label=_majority(examples))
            self.stats.n_leaves += 1
            return leaf_id

        if len(self._questions) >= self.max_questions:
            self._nodes[leaf_id] = TreeNode(leaf_id, leaf_label=_majority(examples))
            self.stats.n_leaves += 1
            return leaf_id

        train, val = self._split_train_val(examples)
        candidates = self._propose(train)
        if not candidates:
            self._nodes[leaf_id] = TreeNode(leaf_id, leaf_label=_majority(examples))
            self.stats.n_leaves += 1
            return leaf_id

        wmap = class_weights(examples)
        scored = self._score_candidates(train, candidates)
        ranked = rank_questions(train, scored, min_child=self.min_child, weight_map=wmap)
        ranked = [r for r in ranked if r.score >= self.min_gain]
        if not ranked:
            self._nodes[leaf_id] = TreeNode(leaf_id, leaf_label=_majority(examples))
            self.stats.n_leaves += 1
            return leaf_id

        info_by_qid = {r.qid: r for r in ranked}
        val_scores = self._val_selection_score(train, val, info_by_qid, candidates, wmap)
        node_majority = _majority(examples)
        val_total = sum(wmap.get(e.gold, 1.0) for e in val) or 1.0
        baseline = sum(wmap.get(e.gold, 1.0) for e in val if e.gold is node_majority) / val_total
        # Validation is a VETO, not a gate: selection is class-weighted train
        # gain (rarity-discounted); validation only rejects candidates that are
        # clearly harmful on held-out data. A must-beat-baseline gate starves
        # multi-class nodes because two-leaf terminal accuracy is a noisy proxy
        # for a question's entropy reduction (iteration-1 lesson).
        scored_candidates = [(q, val_scores.get(q.qid), info_by_qid.get(q.qid)) for q in candidates]
        pickable = []
        for q, vs, inf in scored_candidates:
            if inf is None:
                continue
            if vs is not None and vs < baseline - self.veto_delta:
                continue  # clearly harmful on held-out data
            rf = (rare_fraction(q.text, self.corpus_texts, self.rare_min_df)
                  if self.corpus_texts else 0.0)
            final = inf.score * (1.0 - self.rare_penalty * rf)
            pickable.append((q, final, inf))
        for q, vs, inf in scored_candidates:
            self.stats.candidates.append({
                "node": leaf_id, "question": q.text, "source": q.provenance,
                "train_ig": inf.ig if inf else None, "train_cert": inf.certainty if inf else None,
                "train_score": inf.score if inf else None,
                "val_score": vs, "baseline": round(baseline, 4), "picked": False,
            })
        if not pickable:
            self._nodes[leaf_id] = TreeNode(leaf_id, leaf_label=_majority(examples))
            self.stats.n_leaves += 1
            return leaf_id

        pickable.sort(key=lambda x: x[1], reverse=True)
        question, _, best = pickable[0]
        for row in self.stats.candidates:
            if row["node"] == leaf_id and row["question"] == question.text:
                row["picked"] = True
                break
        self._questions[question.qid] = question

        full_ps = self._score_candidates(examples, [question])[question.qid]
        yes = [e for e, p in zip(examples, full_ps) if p >= 0.5]
        no = [e for e, p in zip(examples, full_ps) if p < 0.5]

        self.stats.per_node.append({
            "node": leaf_id,
            "depth": depth,
            "n_examples": len(examples),
            "question": question.text,
            "ig": best.ig,
            "certainty": best.certainty,
            "score": best.score,
        })
        print("[induction] depth %d picked: %s (ig %.3f, cert %.3f)" % (
            depth, question.text, best.ig, best.certainty), file=sys.stderr)

        yes_id = self._grow(yes, depth + 1)
        no_id = self._grow(no, depth + 1)
        self._nodes[leaf_id] = TreeNode(leaf_id, question_qid=question.qid,
                                        branches={'yes': yes_id, 'no': no_id})
        return leaf_id

    def induce(self, examples: List[Example], tree_id: str = "induced_v1") -> Tree:
        self._nodes = {}
        self._questions = {}
        self._counter = 0
        self.stats = InductionStats()
        root = self._grow(examples, 0)
        self.stats.n_questions = len(self._questions)
        tree = Tree(tree_id, root, self._nodes)
        pruned = self._prune(tree)
        self.stats.pruned_subtrees = pruned
        self.stats.n_leaves = len(tree.leaves())
        return tree

    def _prune(self, tree: Tree) -> int:
        """Collapse subtrees whose leaves all share one majority label."""
        pruned = 0
        leaf_labels: Dict[str, DecisionLabel] = {}

        def uniform(nid: str) -> Optional[DecisionLabel]:
            if nid in leaf_labels:
                return leaf_labels[nid]
            node = tree.nodes[nid]
            if node.is_leaf:
                leaf_labels[nid] = node.leaf_label
                return node.leaf_label
            child_labels = [uniform(child) for child in node.branches.values()]
            if all(cl is not None for cl in child_labels) and len(set(child_labels)) == 1:
                leaf_labels[nid] = child_labels[0]
                return child_labels[0]
            leaf_labels[nid] = None
            return None

        def collapse(nid: str) -> None:
            nonlocal pruned
            node = tree.nodes[nid]
            if node.is_leaf:
                return
            for child in node.branches.values():
                collapse(child)
            u = uniform(nid)
            if u is not None:
                tree.nodes[nid] = TreeNode(nid, leaf_label=u)
                pruned += 1

        collapse(tree.root)
        return pruned


def induced_questions(tree: Tree, questions: Dict[str, Question]) -> List[Question]:
    return [questions[qid] for qid in tree.questions()]
