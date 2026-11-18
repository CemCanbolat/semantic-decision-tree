"""Jev scoring client interface + deterministic mock.

The real client talks to a batched scoring endpoint: multiple typed questions
against one (text, context) pair in a single call. Credentials/endpoint are not
available in this environment, so HttpJevClient fails loudly with configuration
guidance instead of guessing. MockJevClient is deterministic and drives tests
and the offline induction smoke path.

Computed questions never call the model at all: a registered resolver answers
them exactly from the JSON context (p in {0.0, 1.0}). Resolvers are named in
the Question artifact, which keeps the tree self-describing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from sdt.schemas import Question, QuestionType, SdtError, ScoreVector


Resolver = Callable[[Dict], bool]


def _focus_order(ctx: Dict) -> Dict:
    oid = ctx.get("focus_order_id")
    orders = ctx.get("orders") or {}
    if oid and oid in orders:
        return orders[oid]
    if orders:
        return next(iter(orders.values()))
    return None  # order-less context: no status holds, questions answer False


def resolve_order_delivered(ctx: Dict) -> bool:
    order = _focus_order(ctx)
    return order is not None and order.get("status") == "delivered"


def resolve_order_pending(ctx: Dict) -> bool:
    order = _focus_order(ctx)
    return order is not None and order.get("status") == "pending"


def resolve_refund_count_gt_2(ctx: Dict) -> bool:
    """tau2 has no 90-day window; count refund transactions across the user's orders."""
    count = 0
    for order in (ctx.get("orders") or {}).values():
        for tx in order.get("payment_history", []):
            if tx.get("transaction_type") == "refund":
                count += 1
    return count > 2


def resolve_has_gift_card(ctx: Dict) -> bool:
    methods = (ctx.get("user") or {}).get("payment_methods") or {}
    return any(m.get("source") == "gift_card" for m in methods.values())


DEFAULT_RESOLVERS: Dict[str, Resolver] = {
    "order_status_delivered": resolve_order_delivered,
    "order_status_pending": resolve_order_pending,
    "refund_count_gt_2": resolve_refund_count_gt_2,
    "has_gift_card": resolve_has_gift_card,
}


def _computed_seed(text: str, resolver: str) -> Question:
    return Question(text=text, qtype=QuestionType.COMPUTED,
                    resolver=resolver, provenance="seed-computed")


# Exact features available to every candidate pool: they answer from the JSON
# context for free (p in {0, 1}) and compete on gain like any proposal.
COMPUTED_SEEDS: List[Question] = [
    _computed_seed("Is the referenced order status delivered?", "order_status_delivered"),
    _computed_seed("Is the referenced order status pending?", "order_status_pending"),
    _computed_seed("Have there been more than two refunds on the account's orders?",
                   "refund_count_gt_2"),
    _computed_seed("Does the account hold a gift card?", "has_gift_card"),
]


class JevClient(ABC):
    @abstractmethod
    def score_batch(self, text: str, context: Dict, questions: List[Question]) -> ScoreVector:
        """Score every question against one (text, context) pair in one pass."""


class MockJevClient(JevClient):
    """Deterministic mock.

    - computed questions: resolved exactly from context via registered resolvers
    - semantic questions: looked up in an oracle mapping question cache_key -> p;
      unseen questions return a seeded pseudo-probability derived from the hash,
      so behavior is stable across runs without being degenerate.
    """

    def __init__(self, oracle: Optional[Dict[str, float]] = None,
                 resolvers: Optional[Dict[str, Resolver]] = None,
                 latency_ms: float = 0.1) -> None:
        self.oracle = dict(oracle or {})
        self.resolvers = dict(DEFAULT_RESOLVERS)
        self.resolvers.update(resolvers or {})
        self.latency_ms = latency_ms

    def score_batch(self, text: str, context: Dict, questions: List[Question]) -> ScoreVector:
        start = time.perf_counter()
        scores: Dict[str, float] = {}
        for q in questions:
            if q.qtype is QuestionType.COMPUTED:
                fn = self.resolvers.get(q.resolver or "")
                if fn is None:
                    raise SdtError("no resolver registered for %r" % q.resolver)
                scores[q.qid] = 1.0 if fn(context) else 0.0
            else:
                p = self._semantic_p(q, text, context)
                scores[q.qid] = max(0.0, min(1.0, p))
        elapsed = (time.perf_counter() - start) * 1000.0
        return ScoreVector(example_id="", scores=scores, latency_ms=round(max(elapsed, self.latency_ms), 3))

    def _semantic_p(self, q: Question, text: str, context: Dict) -> float:
        """Default: oracle lookup, else a stable hash-derived pseudo-probability."""
        if q.cache_key in self.oracle:
            return float(self.oracle[q.cache_key])
        return (int(q.cache_key[:8], 16) % 1000) / 999.0


class HttpJevClient(JevClient):
    """Placeholder for the real endpoint. Configure via env when available."""

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def score_batch(self, text: str, context: Dict, questions: List[Question]) -> ScoreVector:
        raise SdtError(
            "HttpJevClient requires JEV_ENDPOINT and JEV_API_KEY configuration; "
            "the sandbox has no network access. Use MockJevClient for offline work."
        )
