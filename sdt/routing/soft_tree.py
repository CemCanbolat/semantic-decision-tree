"""Soft probabilistic routing over a flattened question batch.

Works for binary noul trees AND multi-way trees (choice = K branches,
score = L rubric levels). Every branch carries a probability column; the
leaf probability is the product along the path, computed in log space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sdt.decision_types import DecisionType, branch_names, column_id
from sdt.schemas import DecisionLabel, Question, ScoreVector, SdtError, Tree

_EPS = 1e-9


def _clamp(p) -> float:
    if p is None:
        p = 1.0  # missing column: uniform is safer than zero for renormalized leaves
    return min(max(float(p), _EPS), 1.0 - _EPS)


@dataclass
class RoutingResult:
    leaf_log_probs: Dict[str, float]
    class_posterior: Dict[DecisionLabel, float]
    dominant_path: List[str]
    dominant_leaf: str
    path_probs: Dict[str, float]

    @property
    def leaf_probs(self) -> Dict[str, float]:
        return {k: math.exp(v) for k, v in self.leaf_log_probs.items()}


def _branch_probs(node, scores: Dict[str, float],
                  questions: Optional[Dict[str, Question]]) -> Dict[str, float]:
    qid = node.question_qid
    q = (questions or {}).get(qid)
    if q is not None and q.decision_type is not DecisionType.NOUL:
        return {b: _clamp(scores.get(column_id(qid, b))) for b in branch_names(q)}
    if qid in scores:  # noul (with or without an explicit question object)
        p = _clamp(scores.get(qid))
        return {"yes": p, "no": 1.0 - p}
    # inference for multi-way nodes without a question object: scan columns
    prefix = qid + "::"
    return {col[len(prefix):]: _clamp(scores.get(col))
            for col in scores if col.startswith(prefix)}


def route(tree: Tree, scores, questions: Optional[Dict[str, Question]] = None) -> RoutingResult:
    flat = scores.scores if isinstance(scores, ScoreVector) else scores
    for qid in tree.questions():
        if qid in flat or any(c.startswith(qid + "::") for c in flat):
            continue
        raise SdtError("missing scores for question %s" % qid)

    leaf_log: Dict[str, float] = {}

    def visit(node_id: str, log_weight: float) -> None:
        node = tree.nodes[node_id]
        if node.is_leaf:
            leaf_log[node_id] = log_weight
            return
        probs = _branch_probs(node, flat, questions)
        for branch, child in node.branches.items():
            p = probs.get(branch)
            if p is None:
                raise SdtError("no probability column for branch %r of %s" % (branch, node_id))
            visit(child, log_weight + math.log(p))

    visit(tree.root, 0.0)

    mx = max(leaf_log.values())
    exp_weights = {nid: math.exp(v - mx) for nid, v in leaf_log.items()}
    total = sum(exp_weights.values())
    posterior: Dict[DecisionLabel, float] = {}
    for nid, w in exp_weights.items():
        label = tree.nodes[nid].leaf_label
        posterior[label] = posterior.get(label, 0.0) + w / total

    path_nodes: List[str] = [tree.root]
    path_probs: Dict[str, float] = {}
    cur = tree.root
    while not tree.nodes[cur].is_leaf:
        probs = _branch_probs(tree.nodes[cur], flat, questions)
        best = max(probs.items(), key=lambda kv: kv[1])[0]
        path_probs[cur] = probs[best]
        cur = tree.nodes[cur].branches[best]
        path_nodes.append(cur)

    return RoutingResult(
        leaf_log_probs=leaf_log,
        class_posterior=posterior,
        dominant_path=path_nodes,
        dominant_leaf=cur,
        path_probs=path_probs,
    )
