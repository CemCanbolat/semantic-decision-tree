"""HeuristicJevClient: a deterministic keyword scorer standing in for the live
Jev endpoint so the full pipeline runs offline.

It is NOT a model of Jev quality. Its job is to exercise plumbing end to end:
tree flattening, batching, soft routing, gates, metrics, reports. Every report
produced with it is marked accordingly. When credentials arrive, the same
runner swaps in HttpJevClient and nothing downstream changes.

Rules are soft (0.9 / 0.1, not 1.0 / 0.0) so soft routing has real
probabilities to multiply; computed questions stay exact via resolvers.
"""

from __future__ import annotations

from typing import Dict, Tuple

from sdt.jev_client import MockJevClient
from sdt.schemas import Question

YES = 0.9
NO = 0.1


def _any_in(blob: str, words: Tuple[str, ...]) -> bool:
    return any(w in blob for w in words)


# rule table: substring of the question text -> predicate over the input text
_RULES: Tuple[Tuple[Tuple[str, ...], object], ...] = (
    (("concrete account or order action",),
     lambda blob: _any_in(blob, ("return", "refund", "money back", "exchange", "cancel",
                                 "modify", "change", "delete", "transfer", "address", "payment", "ship to"))),
    (("transfer to a human agent",),
     lambda blob: not _any_in(blob, ("lawyer", "delete my", "sue", "chargeback", "legal action", "regulator"))),
    (("order id or enough details",),
     lambda blob: "#" in blob),
    (("return or refund",),
     lambda blob: _any_in(blob, ("return", "refund", "money back"))),
    (("exchange",),
     lambda blob: "exchange" in blob),
    (("cancellation of a pending order",),
     lambda blob: "cancel" in blob),
    (("modification of the pending order",),
     lambda blob: _any_in(blob, ("address", "payment", "modify", "switch"))),
    (("concern a specific order",),
     lambda blob: "order" in blob),
    (("already been delivered",),
     lambda blob: "deliv" in blob),
)


class HeuristicJevClient(MockJevClient):
    scorer_name = "heuristic-keyword-standin"

    def _semantic_p(self, q: Question, text: str, context: Dict) -> float:
        blob = text.lower()
        for markers, rule in _RULES:
            if any(m in q.text.lower() for m in markers):
                return YES if rule(blob) else NO
        return super()._semantic_p(q, text, context)
