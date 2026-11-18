"""RuleFit-style interaction features for the combiner. Zero API calls.

Raw question columns are additive: the linear layer cannot express
"option 3 AND NOT roaming" without interaction terms. We add:

- pairwise products p_i * p_j of the semantic probability columns
  (complements are omitted deliberately: p_i * (1 - p_j) is linearly
  spanned by p_i and p_i * p_j, so products alone suffice)
- optionally, the grown tree's soft leaf posteriors - the tree already
  encodes the useful IF/ELSE chains as features
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from sdt.decision_types import DecisionType, columns_of
from sdt.routing.soft_tree import route
from sdt.schemas import Example, Question, ScoreVector, Tree
from sdt.training.cache import ScoreCache


def raw_columns(examples: List[Example], semantic: List[Question],
                cache: ScoreCache) -> Tuple[List[List[float]], List[str]]:
    matrix, names = [], []
    for q in semantic:
        for col in columns_of(q):
            names.append(col)
    for ex in examples:
        row = []
        for q in semantic:
            if q.decision_type is DecisionType.NOUL:
                p = cache.get(q, ex.example_id)
                row.append(0.5 if p is None else float(p))
            else:
                cols = cache.get_columns(q, ex.example_id) or {}
                for col in columns_of(q):
                    row.append(float(cols.get(col.split("::")[-1], 0.0)))
        matrix.append(row)
    return matrix, names


def augmented_features(examples: List[Example], semantic: List[Question],
                       cache: ScoreCache, tree: Optional[Tuple[Tree, List[Question]]] = None,
                       include_computed: bool = True,
                       max_interactions: int = 64) -> Tuple[List[List[float]], List[str]]:
    from sdt.jev_client import COMPUTED_SEEDS
    from sdt.training.combiner import computed_feature_vector

    X_raw, raw_names = raw_columns(examples, semantic, cache)
    n_sem = len(raw_names)
    pairs = [(i, j) for i in range(n_sem) for j in range(i + 1, n_sem)]
    pairs = pairs[:max_interactions]
    X = []
    names = list(raw_names)
    for i, j in pairs:
        names.append(raw_names[i] + " x " + raw_names[j])
    leaf_names = []
    leaf_probs: List[List[float]] = []
    if tree is not None:
        t, t_questions = tree
        by_qid = {q.qid: q for q in t_questions}
        leaf_names = ["leaf:" + nid for nid in sorted(t.leaves())]
        for ex in examples:
            scores: Dict[str, float] = {}
            for q in t_questions:
                if q.decision_type is DecisionType.NOUL:
                    p = cache.get(q, ex.example_id)
                    scores[q.qid] = 0.5 if p is None else float(p)
                else:
                    for col, p in (cache.get_columns(q, ex.example_id) or {}).items():
                        scores[q.qid + "::" + col] = float(p)
            r = route(t, ScoreVector(example_id=ex.example_id, scores=scores), by_qid)
            leaf_probs.append([r.leaf_probs.get(nid, 0.0) for nid in sorted(t.leaves())])
        names = names + leaf_names
    if include_computed:
        names = names + [q.text for q in COMPUTED_SEEDS]
    for idx, ex in enumerate(examples):
        row = list(X_raw[idx])
        for i, j in pairs:
            row.append(X_raw[idx][i] * X_raw[idx][j])
        if leaf_probs:
            row.extend(leaf_probs[idx])
        if include_computed:
            row.extend([1.0 if __import__("sdt.jev_client", fromlist=["DEFAULT_RESOLVERS"]).DEFAULT_RESOLVERS[q.resolver](ex.context) else 0.0
                        for q in COMPUTED_SEEDS])
        X.append(row)
    return X, names


def mild_class_weights(examples: List[Example], alpha: float = 0.35) -> Dict:
    """(N / N_c)^alpha - mild enough to keep empirical calibration usable."""
    from sdt.schemas import DecisionLabel
    counts: Dict = {}
    for e in examples:
        counts[e.gold] = counts.get(e.gold, 0) + 1
    n = len(examples)
    return {c: (n / k) ** alpha for c, k in counts.items()}
