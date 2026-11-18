"""Certainty-weighted, class-balanced information gain for candidate splits.

Two scores live here:

1. Train-time split score:  IG(q) * mean_i( 2 * |p_i - 0.5| )
   IG can be class-weighted (balanced weights, capped) so separating 4
   rare-class examples earns comparable credit to separating 30 majority
   examples. Certainty penalizes questions the scorer answers near 0.5.

2. Nothing here does validation; selection-on-validation lives in grow.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sdt.schemas import DecisionLabel, Example


def entropy(labels: List[DecisionLabel]) -> float:
    if not labels:
        return 0.0
    counts: Dict[DecisionLabel, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    n = len(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def class_weights(examples: List[Example], cap: float = 8.0) -> Dict[DecisionLabel, float]:
    """Balanced weights: w_c = (n / k) / count_c, capped. Sum == n."""
    counts: Dict[DecisionLabel, int] = {}
    for e in examples:
        counts[e.gold] = counts.get(e.gold, 0) + 1
    n, k = len(examples), len(counts)
    return {label: min((n / k) / c, cap) for label, c in counts.items()}


def weighted_entropy(labels: List[DecisionLabel], wmap: Dict[DecisionLabel, float]) -> float:
    agg: Dict[DecisionLabel, float] = {}
    total = 0.0
    for l in labels:
        w = wmap.get(l, 1.0)
        agg[l] = agg.get(l, 0.0) + w
        total += w
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in agg.values())


def certainty(ps: List[float]) -> float:
    if not ps:
        return 0.0
    return sum(2.0 * abs(p - 0.5) for p in ps) / len(ps)


@dataclass
class SplitInfo:
    qid: str
    ig: float
    certainty: float
    score: float
    yes_ids: List[str]
    no_ids: List[str]


def evaluate_question(examples: List[Example], qid: str, ps: List[float],
                      weight_map: Optional[Dict[DecisionLabel, float]] = None) -> SplitInfo:
    assert len(examples) == len(ps)
    wmap = weight_map or {}
    node_h = weighted_entropy([e.gold for e in examples], wmap)
    yes = [e for e, p in zip(examples, ps) if p >= 0.5]
    no = [e for e, p in zip(examples, ps) if p < 0.5]
    n = len(examples)
    wsum = lambda exs: sum(wmap.get(e.gold, 1.0) for e in exs)
    total_w = wsum(examples) or 1.0
    ig = node_h
    if yes:
        ig -= (wsum(yes) / total_w) * weighted_entropy([e.gold for e in yes], wmap)
    if no:
        ig -= (wsum(no) / total_w) * weighted_entropy([e.gold for e in no], wmap)
    cert = certainty(ps)
    return SplitInfo(
        qid=qid,
        ig=round(ig, 6),
        certainty=round(cert, 6),
        score=round(ig * cert, 6),
        yes_ids=[e.example_id for e in yes],
        no_ids=[e.example_id for e in no],
    )


def rank_questions(examples: List[Example],
                   scores: Dict[str, List[float]],
                   min_child: int = 2,
                   weight_map: Optional[Dict[DecisionLabel, float]] = None) -> List[SplitInfo]:
    ranked = []
    for qid, ps in scores.items():
        info = evaluate_question(examples, qid, ps, weight_map)
        if min(len(info.yes_ids), len(info.no_ids)) >= min_child:
            ranked.append(info)
    return sorted(ranked, key=lambda s: s.score, reverse=True)
