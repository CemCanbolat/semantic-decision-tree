"""Phase-4 combiner: temperature-scaled softmax regression on cached p-vectors.

The tree's job was feature discovery; this module learns the decision
surface on top of those features. Everything runs from the score cache -
zero API calls. Pure Python (no numpy): the matrices here are ~114 x 18.

Also provides post-hoc class-weighted leaf relabeling for a tree artifact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sdt.induction.gain import class_weights
from sdt.jev_client import COMPUTED_SEEDS, DEFAULT_RESOLVERS
from sdt.schemas import DecisionLabel, Example, Question, QuestionType, Tree, TreeNode
from sdt.training.cache import ScoreCache

COMPUTED_SEEDS = COMPUTED_SEEDS  # re-exported from jev_client for compatibility


def computed_feature_vector(context: Dict) -> List[float]:
    return [1.0 if DEFAULT_RESOLVERS[q.resolver](context) else 0.0 for q in COMPUTED_SEEDS]


def build_features(examples: List[Example], semantic: List[Question],
                   cache: ScoreCache, include_computed: bool = True) -> Tuple[List[List[float]], List[str]]:
    names, matrix, missing = [], [], 0
    seed_texts = {q.text.lower() for q in COMPUTED_SEEDS}
    semantic = [q for q in semantic if q.text.lower() not in seed_texts]  # no duplicate seed columns
    from sdt.decision_types import columns_of
    sem_names = []
    for q in semantic:
        for col in columns_of(q):
            sem_names.append(q.text + ("[" + col.split("::")[-1] + "]" if "::" in col else ""))
    for ex in examples:
        row = []
        for q in semantic:
            if q.decision_type.value == "noul":
                p = cache.get(q, ex.example_id)
                if p is None:
                    missing += 1
                    p = 0.5
                row.append(float(p))
            else:
                cols = cache.get_columns(q, ex.example_id)
                if cols is None:
                    missing += 1
                    cols = {b: 1.0 / len(columns_of(q)) for b in
                            [c.split("::")[-1] for c in columns_of(q)]}
                for col in columns_of(q):
                    row.append(float(cols.get(col.split("::")[-1], 0.0)))
        if include_computed:
            row.extend(computed_feature_vector(ex.context))
        matrix.append(row)
    names = sem_names + ([q.text for q in COMPUTED_SEEDS] if include_computed else [])
    if missing:
        raise ValueError("cache missing %d (question, example) scores for the requested feature set" % missing)
    return matrix, names


def _softmax(z: List[float]) -> List[float]:
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]


class SoftmaxModel:
    """Full-batch gradient descent, class-weighted cross-entropy, L2."""

    def __init__(self, labels: List[DecisionLabel], n_features: int,
                 lr: float = 0.5, l2: float = 0.01, iters: int = 1500) -> None:
        self.labels = labels
        self.K = len(labels)
        self.D = n_features
        self.lr, self.l2, self.iters = lr, l2, iters
        self.W = [[0.0] * n_features for _ in range(self.K)]
        self.b = [0.0] * self.K

    def logits(self, x: List[float]) -> List[float]:
        return [sum(w * xi for w, xi in zip(self.W[k], x)) + self.b[k]
                for k in range(self.K)]

    def fit(self, X: List[List[float]], y: List[int], sample_w: List[float]) -> "SoftmaxModel":
        n = len(X)
        for _ in range(self.iters):
            grads_W = [[0.0] * self.D for _ in range(self.K)]
            grads_b = [0.0] * self.K
            for xi, yi, wi in zip(X, y, sample_w):
                probs = _softmax(self.logits(xi))
                for k in range(self.K):
                    g = (probs[k] - (1.0 if k == yi else 0.0)) * wi
                    grads_b[k] += g
                    row = grads_W[k]
                    for d, xid in enumerate(xi):
                        row[d] += g * xid
            for k in range(self.K):
                self.b[k] -= self.lr * grads_b[k] / n
                for d in range(self.D):
                    self.W[k][d] -= self.lr * (grads_W[k][d] / n + self.l2 * self.W[k][d])
        return self

    def predict_index(self, x: List[float]) -> int:
        z = self.logits(x)
        return max(range(self.K), key=lambda k: z[k])

    def predict_proba(self, x: List[float], temperature: float = 1.0) -> List[float]:
        return _softmax([v / temperature for v in self.logits(x)])


def stratified_folds(examples: List[Example], k: int = 5) -> List[int]:
    """Deterministic: sort per class by example_id, deal round-robin."""
    fold_of: Dict[str, int] = {}
    by_class: Dict[DecisionLabel, List[str]] = {}
    for e in sorted(examples, key=lambda x: x.example_id):
        by_class.setdefault(e.gold, []).append(e.example_id)
    for label in sorted(by_class, key=lambda l: l.value):
        for i, eid in enumerate(by_class[label]):
            fold_of[eid] = i % k
    return [fold_of[e.example_id] for e in examples]


def accuracy_on(y_true: List[int], y_pred: List[int]) -> float:
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else 0.0


def macro_f1_on(labels: List[DecisionLabel], y_true: List[int], y_pred: List[int]) -> float:
    f1s = []
    for c in range(len(labels)):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def cross_validate(X, y, sample_w, labels, k=5, **fit_kw):
    """Returns (oof_logits, oof_accuracy, oof_macro_f1) - honest, no leakage."""
    n = len(X)
    folds = stratified_folds_by_index(y, labels, n, k)
    oof_logits: List[Optional[List[float]]] = [None] * n
    for f in range(k):
        tr = [i for i in range(n) if folds[i] != f]
        te = [i for i in range(n) if folds[i] == f]
        model = SoftmaxModel(labels, len(X[0]), **fit_kw).fit(
            [X[i] for i in tr], [y[i] for i in tr], [sample_w[i] for i in tr])
        for i in te:
            oof_logits[i] = model.logits(X[i])
    preds = [max(range(len(labels)), key=lambda c: z[c]) for z in oof_logits]
    return oof_logits, accuracy_on(y, preds), macro_f1_on(labels, y, preds)


def stratified_folds_by_index(y: List[int], labels, n: int, k: int) -> List[int]:
    per_class: Dict[int, List[int]] = {}
    for i in range(n):
        per_class.setdefault(y[i], []).append(i)
    folds = [0] * n
    for c in per_class:
        for j, i in enumerate(per_class[c]):
            folds[i] = j % k
    return folds


def fit_temperature(oof_logits: List[List[float]], y: List[int]) -> float:
    """Grid search T minimizing NLL (calibration for the margin gate)."""
    best_t, best_nll = 1.0, float("inf")
    t = 0.05
    while t <= 10.0:
        nll = 0.0
        for z, yi in zip(oof_logits, y):
            probs = _softmax([v / t for v in z])
            nll -= math.log(max(probs[yi], 1e-12))
        if nll < best_nll:
            best_t, best_nll = t, nll
        t = round(t * 1.15, 4)
    return round(best_t, 4)


def relabel_tree_leaves(tree: Tree, questions: List[Question], examples: List[Example],
                        cache: ScoreCache, alpha: float = 2.0) -> Tuple[Tree, List[dict]]:
    """Post-hoc class-weighted leaf labels. Hard-routes each cached example to
    its dominant leaf, then relabels by weighted majority.

    Balanced weights (w = (n/k)/count) tie exactly in a majority vote
    (w_c * count_c = n/k for every class), so the weight is raised to
    alpha > 1: rare classes need fewer examples to claim a leaf. alpha is
    a measured heuristic, not a theorem - the report shows what it does."""
    by_qid = {q.qid: q for q in questions}
    leaf_examples: Dict[str, List[Example]] = {nid: [] for nid in tree.leaves()}
    for ex in examples:
        cur = tree.root
        while not tree.nodes[cur].is_leaf:
            node = tree.nodes[cur]
            q = by_qid[node.question_qid]
            p = cache.get(q, ex.example_id)
            if p is None:
                p = 0.5
            cur = node.child_yes if p >= 0.5 else node.child_no
        leaf_examples[cur].append(ex)
    changes = []
    new_nodes = dict(tree.nodes)
    for nid, exs in leaf_examples.items():
        if not exs:
            continue
        wmap = class_weights(exs)
        wmap = {label: w ** alpha for label, w in wmap.items()}
        scores: Dict[DecisionLabel, float] = {}
        for e in exs:
            scores[e.gold] = scores.get(e.gold, 0.0) + wmap.get(e.gold, 1.0)
        winner = max(scores.items(), key=lambda kv: (kv[1], kv[0].value))[0]
        old = tree.nodes[nid].leaf_label
        if old is not winner:
            changes.append({"leaf": nid, "from": old.value, "to": winner.value,
                            "n_examples": len(exs)})
            new_nodes[nid] = TreeNode(nid, leaf_label=winner)
    return Tree(tree.tree_id + "_relabeled", tree.root, new_nodes, tree.version), changes
