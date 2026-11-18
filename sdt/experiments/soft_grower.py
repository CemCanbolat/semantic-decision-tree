"""Iteration 8: soft-mass free-type grower with gain-ratio discipline.

Four changes over the iteration-7 free inducer, plus two gain-ratio guards:

1. Class-weighted entropy with a soft exponent (alpha = 0.6): rare classes
   register without the alpha=1 degeneracy (balanced weights tie exactly).
2. Gain Ratio, guarded: IG / max(H_split, 0.35) - the C4.5 lopsided-split
   exploit (97/3 splits have tiny denominators) is blocked by the floor AND
   by a minimum minority-branch effective mass of 3.0, plus a small
   (K - 1) complexity penalty so choice questions must prove per-branch
   signal before fragmenting the tree.
3. Fractional routing during growth (weights < 0.05 dropped): growth
   finally matches soft inference semantics; a 0.52 ticket keeps 48% of
   its mass in the other branch instead of vanishing.
4. Grow deep (relaxed brakes), then prune bottom-up by validation
   macro-F1 lift - run LEAKAGE-FREE: growth sees only the grow split,
   pruning decisions only the prune split, the test fold only scores.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from sdt.decision_types import DecisionType, branch_names
from sdt.decision_types import choice as mk_choice
from sdt.decision_types import noul as mk_noul
from sdt.decision_types import score as mk_score
from sdt.experiments.common import score_with_cache
from sdt.experiments.exp_b_free import FREE_SYSTEM, parse_free
from sdt.schemas import DecisionLabel, Example, Question, TreeNode, Tree

ALPHA = 0.6
DENOM_FLOOR = 0.35
MIN_BRANCH_MASS = 3.0
MIN_NODE_MASS = 8.0
LAMBDA_K = 0.02
WEIGHT_CUTOFF = 0.05
SCORE_FLOOR = 0.01
PURITY = 0.97
CLASS_CAP = 250          # balanced growth: max examples of one class shown per prompt
DEDUP_JACCARD = 0.75     # reject candidates too similar to accepted questions

Item = Tuple[Example, float]  # (example, fractional weight)


def alpha_class_weights(train: List[Example], alpha: float = ALPHA) -> Dict[DecisionLabel, float]:
    counts: Dict[DecisionLabel, int] = {}
    for e in train:
        counts[e.gold] = counts.get(e.gold, 0) + 1
    return {c: 1.0 / (n ** alpha) for c, n in counts.items()}


def branch_probabilities(q: Question, cache, ex: Example) -> Dict[str, float]:
    if q.decision_type is DecisionType.NOUL:
        p = cache.get(q, ex.example_id)
        p = 0.5 if p is None else p
        return {"yes": p, "no": 1.0 - p}
    cols = cache.get_columns(q, ex.example_id) or {}
    total = sum(cols.values()) or 1.0
    return {b: cols.get(b, 0.0) / total for b in branch_names(q)}


def _weighted_entropy(items: List[Item], cw: Dict[DecisionLabel, float]) -> float:
    mass: Dict[DecisionLabel, float] = {}
    total = 0.0
    for ex, w in items:
        m = w * cw.get(ex.gold, 1.0)
        mass[ex.gold] = mass.get(ex.gold, 0.0) + m
        total += m
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in mass.values() if v > 0)


def _mass(items: List[Item], cw) -> float:
    return sum(w * cw.get(ex.gold, 1.0) for ex, w in items)


def _majority(items: List[Item], cw) -> DecisionLabel:
    mass: Dict[DecisionLabel, float] = {}
    for ex, w in items:
        mass[ex.gold] = mass.get(ex.gold, 0.0) + w * cw.get(ex.gold, 1.0)
    return max(mass.items(), key=lambda kv: (kv[1], kv[0].value))[0]


def split_score(items: List[Item], q: Question, cache, cw) -> Optional[Tuple[float, Dict[str, List[Item]]]]:
    """Gain-ratio score with the lopsided-split guards. None = reject."""
    branches = branch_names(q)
    children: Dict[str, List[Item]] = {b: [] for b in branches}
    cert_sum = 0.0
    for ex, w in items:
        probs = branch_probabilities(q, cache, ex)
        ranked = sorted(probs.values(), reverse=True)
        cert_sum += (ranked[0] - ranked[1]) if len(ranked) > 1 else 1.0
        for b in branches:
            contribution = w * probs.get(b, 0.0)
            if contribution >= WEIGHT_CUTOFF:
                children[b].append((ex, contribution))
    total_mass = _mass(items, cw)
    branch_masses = {b: _mass(children[b], cw) for b in branches}
    # minority guard in RAW effective tickets (fractional mass, no class
    # weighting): class-alpha weights shrink rare-class mass by design, so
    # guarding on weighted mass would reject exactly the rare-class splits
    # the alpha weighting exists to enable
    raw_branch_mass = {b: sum(w for _, w in children[b]) for b in branches}
    if min(raw_branch_mass.values()) < MIN_BRANCH_MASS:
        return None  # no 1-ticket peel-offs
    h_node = _weighted_entropy(items, cw)
    ig = h_node
    for b in branches:
        frac = branch_masses[b] / total_mass
        ig -= frac * _weighted_entropy(children[b], cw)
    h_split = -sum((m / total_mass) * math.log2(m / total_mass)
                   for m in branch_masses.values() if m > 0)
    denom = max(h_split, DENOM_FLOOR)
    certainty = cert_sum / max(len(items), 1)
    score = (ig / denom - LAMBDA_K * (len(branches) - 1)) * certainty
    if score < SCORE_FLOOR:
        return None
    return score, children


class SoftGrower:
    def __init__(self, frontier, client, cache, train: List[Example],
                 max_depth: int = 4, max_frontier_calls: int = 40, seed: int = 23):
        self.frontier, self.client, self.cache = frontier, client, cache
        self.train = train
        self.cw = alpha_class_weights(train)
        self.max_depth = max_depth
        self.max_frontier_calls = max_frontier_calls
        self.rng = random.Random(seed)
        self.nodes: Dict[str, TreeNode] = {}
        self.questions: Dict[str, Question] = {}
        self.counter = 0
        self.frontier_calls = 0

    def _leaf(self, items: List[Item]) -> str:
        self.counter += 1
        nid = "s%03d" % self.counter
        self.nodes[nid] = TreeNode(nid, leaf_label=_majority(items, self.cw) if items else DecisionLabel.OUT_OF_SCOPE)
        return nid

    def _propose(self, items: List[Item]):
        mass_by_class: Dict[DecisionLabel, List[Item]] = {}
        for it in items:
            mass_by_class.setdefault(it[0].gold, []).append(it)
        ranked = sorted(mass_by_class.items(), key=lambda kv: (-_mass(kv[1], self.cw), kv[0].value))
        if len(ranked) < 2:
            return []
        top = [e for e, _ in sorted(ranked[0][1], key=lambda it: -it[1])[:8]]
        second = [e for e, _ in sorted(ranked[1][1], key=lambda it: -it[1])[:8]]
        # balanced growth: cap any single class at CLASS_CAP examples per prompt
        top = top[:CLASS_CAP]
        second = second[:CLASS_CAP]
        block = lambda exs: chr(10).join("- " + e.text.replace(chr(10), " ")[:150] for e in exs)
        user = chr(10).join([
            "GROUP A tickets (" + ranked[0][0].value + "):", block(top), "",
            "GROUP B tickets (" + ranked[1][0].value + "):", block(second), "",
            "Propose up to 8 questions in ANY of the three formats (NOUL / CHOICE with OPTIONS / SCORE with LEVELS) that separate GROUP A from GROUP B. One per line.",
        ])
        raw = self.frontier.complete(FREE_SYSTEM, user)
        self.frontier_calls += 1
        return parse_free(raw, "iter8")

    def ovr_pass(self, examples: List[Example], target: DecisionLabel) -> Optional[Question]:
        """One one-vs-rest proposal pass for a target class, scored on the
        full corpus and added to the question pool. Returns the question or None."""
        pos = [e for e in examples if e.gold is target][:8]
        if not pos:
            return None
        others = [e for e in examples if e.gold is not target][:8]
        block = lambda exs: chr(10).join("- " + e.text.replace(chr(10), " ")[:150] for e in exs)
        system = FREE_SYSTEM
        user = chr(10).join([
            "GROUP A tickets (" + target.value + "):", block(pos), "",
            "GROUP B tickets (other):", block(others), "",
            "Propose up to 8 questions in ANY format that separate GROUP A from GROUP B. One per line.",
        ])
        raw = self.frontier.complete(system, user)
        self.frontier_calls += 1
        cands = parse_free(raw, "ovr:" + target.value)
        if not cands:
            return None
        score_with_cache(self.client, examples, cands, self.cache)
        best = None
        for q in cands:
            res = split_score([(e, 1.0) for e in examples], q, self.cache, self.cw)
            if res is not None and (best is None or res[0] > best[0]):
                best = (res[0], q)
        if best is None:
            return None
        q = best[1]
        self.questions[q.qid] = q
        print("[ovr:%s] %s: %s (score %.3f)" % (
            target.value, q.decision_type.value, q.text[:70], best[0]), flush=True)
        return q

    @staticmethod
    def _token_set(text: str):
        return {w.lower().strip(".,!?;:'\"()") for w in text.split() if len(w) > 2}

    def _too_similar(self, candidate: Question) -> bool:
        """Lexical dedup vs already-accepted questions (Jaccard > 0.75)."""
        cand_tokens = self._token_set(candidate.text)
        for q in self.questions.values():
            other = self._token_set(q.text)
            union = cand_tokens | other
            if union and len(cand_tokens & other) / len(union) > DEDUP_JACCARD:
                return True
        return False

    def grow(self, items: List[Item], depth: int = 0) -> str:
        effective = sum(w for _, w in items)
        class_mass: Dict[DecisionLabel, float] = {}
        for ex, w in items:
            class_mass[ex.gold] = class_mass.get(ex.gold, 0.0) + w
        if (depth >= self.max_depth or effective < MIN_NODE_MASS
                or max(class_mass.values()) / max(sum(class_mass.values()), 1e-9) >= PURITY):
            return self._leaf(items)
        if self.frontier_calls >= self.max_frontier_calls:
            return self._leaf(items)

        proposals = self._propose(items)
        from sdt.jev_client import COMPUTED_SEEDS
        seen_texts = set()
        candidates = []
        for q in list(COMPUTED_SEEDS) + proposals:
            if q.text.lower() not in seen_texts:
                seen_texts.add(q.text.lower())
                candidates.append(q)
        if not candidates:
            return self._leaf(items)
        seen_examples = [e for e, _ in items]
        score_with_cache(self.client, seen_examples, candidates, self.cache)
        best = None
        for q in candidates:
            if q.decision_type is DecisionType.NOUL and self._too_similar(q):
                continue  # semantic dedup: force diversity, not copies of banked questions
            res = split_score(items, q, self.cache, self.cw)
            if res is not None and (best is None or res[0] > best[0]):
                best = res
                best_q = q
        if best is None:
            return self._leaf(items)
        score, children = best
        self.questions[best_q.qid] = best_q
        child_ids = {b: self.grow(children[b], depth + 1) for b in children}
        self.counter += 1
        nid = "s%03d" % self.counter
        self.nodes[nid] = TreeNode(nid, question_qid=best_q.qid, branches=child_ids)
        print("[iter8] depth %d %s: %s (score %.3f)" % (
            depth, best_q.decision_type.value, best_q.text[:70], score), flush=True)
        return nid

    def induce(self, examples: List[Example], tree_id: str) -> Tree:
        root = self.grow([(e, 1.0) for e in examples])
        return Tree(tree_id, root, self.nodes)


# ---------- leakage-free pruning ----------

def tree_macro_f1(tree: Tree, questions: List[Question], cache,
                  examples: List[Example]) -> float:
    from sdt.routing.soft_tree import route
    from sdt.schemas import ScoreVector
    by_qid = {q.qid: q for q in questions}
    correct: Dict[DecisionLabel, int] = {}
    totals: Dict[DecisionLabel, int] = {}
    preds: Dict[DecisionLabel, int] = {}
    for ex in examples:
        scores: Dict[str, float] = {}
        for q in questions:
            if q.decision_type is DecisionType.NOUL:
                p = cache.get(q, ex.example_id)
                scores[q.qid] = 0.5 if p is None else p
            else:
                for col, p in (cache.get_columns(q, ex.example_id) or {}).items():
                    scores[q.qid + "::" + col] = p
        r = route(tree, ScoreVector(example_id=ex.example_id, scores=scores), by_qid)
        pred = max(r.class_posterior.items(), key=lambda kv: kv[1])[0]
        totals[ex.gold] = totals.get(ex.gold, 0) + 1
        preds[pred] = preds.get(pred, 0) + 1
        if pred is ex.gold:
            correct[ex.gold] = correct.get(ex.gold, 0) + 1
    f1s = []
    for c in totals:
        tp = correct.get(c, 0)
        fp = preds.get(c, 0) - tp
        fn = totals[c] - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def prune_by_validation(tree: Tree, questions: List[Question], cache,
                        prune_examples: List[Example], min_lift: float = 0.002) -> Tree:
    """Bottom-up collapse search; decisions use ONLY prune_examples."""
    best = tree_macro_f1(tree, questions, cache, prune_examples)
    changed = True
    while changed:
        changed = False
        candidates = [nid for nid, n in tree.nodes.items()
                      if not n.is_leaf and all(tree.nodes[c].is_leaf for c in n.branches.values())]
        for nid in candidates:
            node = tree.nodes[nid]
            original = tree.nodes[nid]
            subtree_labels = {tree.nodes[c].leaf_label for c in node.branches.values()}
            counts: Dict[DecisionLabel, int] = {}
            for lbl in subtree_labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            collapsed_label = max(counts.items(), key=lambda kv: (kv[1], kv[0].value))[0]
            tree.nodes[nid] = TreeNode(nid, leaf_label=collapsed_label)
            f1 = tree_macro_f1(tree, questions, cache, prune_examples)
            if f1 > best + min_lift:
                best = f1
                changed = True
            else:
                tree.nodes[nid] = original
    # drop unreachable orphans
    reachable = set()

    def mark(nid):
        if nid in reachable:
            return
        reachable.add(nid)
        node = tree.nodes[nid]
        if not node.is_leaf:
            for c in node.branches.values():
                mark(c)

    mark(tree.root)
    tree.nodes = {nid: n for nid, n in tree.nodes.items() if nid in reachable}
    return tree
