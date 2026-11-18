"""Confidence gates: margin auto-execute plus path-bottleneck escalation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sdt.routing.soft_tree import RoutingResult
from sdt.schemas import DecisionLabel, GateOutcome

DEFAULT_MARGIN_THRESHOLD = 0.60
DEFAULT_BAND = (0.35, 0.65)


@dataclass
class GateResult:
    outcome: GateOutcome
    margin: float
    top: Optional[DecisionLabel]
    runner_up: Optional[DecisionLabel]
    bottleneck_nodes: List[str] = field(default_factory=list)


def decide(
    routing: RoutingResult,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    band: tuple = DEFAULT_BAND,
) -> GateResult:
    ranked = sorted(routing.class_posterior.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = ranked[0]
    runner_up, runner_p = (ranked[1] if len(ranked) > 1 else (None, 0.0))
    margin = top_p - runner_p
    lo, hi = band
    bottlenecks = [
        nid for nid, p in routing.path_probs.items() if lo < p < hi
    ]
    margin_ok = margin > margin_threshold
    bottleneck_ok = not bottlenecks
    if margin_ok and bottleneck_ok:
        outcome = GateOutcome.AUTO_EXECUTED
    elif not margin_ok and not bottleneck_ok:
        outcome = GateOutcome.ESCALATED_BOTH
    elif not margin_ok:
        outcome = GateOutcome.ESCALATED_MARGIN
    else:
        outcome = GateOutcome.ESCALATED_BOTTLENECK
    return GateResult(
        outcome=outcome,
        margin=round(margin, 6),
        top=top,
        runner_up=runner_up,
        bottleneck_nodes=bottlenecks,
    )
