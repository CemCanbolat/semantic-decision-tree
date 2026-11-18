"""Seeded synthetic ticket generator: unit-test fixture and fallback corpus.

NOT the primary dataset. The anchor set is curated from tau2-bench retail
(see sdt/data/tau2_curate.py). This generator exists so tests are hermetic and
so the label space the tau2 action space cannot exercise (notably
request_photo_proof and emotional-but-ineligible denial) still has coverage.

Every example mirrors the tau2 retail context shape (user profile, orders with
status/items/payment history, payment methods) so loaders and resolvers treat
synthetic and curated data identically.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from sdt.schemas import DecisionLabel, Example


_EMOTIONAL_OPENERS = [
    "I have been a loyal customer for years and this is unacceptable!",
    "I am furious. I shop with you all the time and this is how you treat me?",
    "This is ridiculous. I will take my business elsewhere if this is not fixed today.",
    "Absolutely outrageous service. I demand to be compensated.",
]

_TERSE = [
    "Return the coffee maker on my last order please.",
    "I want to return one item. Order attached.",
    "Refund for the espresso machine, thanks.",
]


def _user_ctx(uid: str, first: str, last: str, zipc: str, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "user": {
            "user_id": uid,
            "name": {"first_name": first, "last_name": last},
            "email": first.lower() + "." + last.lower() + "@example.com",
            "address": {"city": "Philadelphia", "state": "PA", "zip": zipc, "country": "USA"},
            "payment_methods": {
                "credit_card_9513926": {"source": "credit_card", "id": "credit_card_9513926", "brand": "visa", "last_four": "1234"},
                "gift_card_8839201": {"source": "gift_card", "id": "gift_card_8839201", "balance": 50.0},
            },
            "orders": [o["order_id"] for o in orders],
        },
        "orders": {o["order_id"]: o for o in orders},
    }


def _order(oid: str, status: str, items: List[Dict[str, Any]], refunded: int = 0) -> Dict[str, Any]:
    history = [{"transaction_type": "payment", "amount": sum(i["price"] for i in items), "payment_method_id": "credit_card_9513926"}]
    for _ in range(refunded):
        history.append({"transaction_type": "refund", "amount": 9.99, "payment_method_id": "credit_card_9513926"})
    return {
        "order_id": oid,
        "status": status,
        "items": items,
        "fulfillments": [{"tracking_id": ["TRK" + oid[-4:]], "item_ids": [i["item_id"] for i in items]}] if status == "delivered" else [],
        "payment_history": history,
        "cancel_reason": None,
        "return_items": None,
        "return_payment_method_id": None,
    }


def _items() -> List[Dict[str, Any]]:
    return [
        {"item_id": "1151293680", "name": "Mechanical Keyboard", "product_id": "1656367028", "price": 109.0, "options": {"switch": "linear", "backlight": "RGB", "size": "full"}},
        {"item_id": "4983901480", "name": "Smart Thermostat", "product_id": "4896585277", "price": 89.0, "options": {"compatibility": "Apple HomeKit"}},
    ]


FAMILIES = {
    DecisionLabel.AUTO_APPROVE: "return_delivered_clean",
    DecisionLabel.POLITELY_DENY: "return_pending_denied",
    DecisionLabel.EXCHANGE: "exchange_delivered",
    DecisionLabel.CANCEL_ORDER: "cancel_pending",
    DecisionLabel.MODIFY_ORDER: "modify_pending",
    DecisionLabel.MODIFY_USER: "modify_user_address",
    DecisionLabel.ESCALATE: "out_of_policy_transfer",
    DecisionLabel.REQUEST_PHOTO_PROOF: "damaged_item_proof",
    DecisionLabel.NEEDS_INFO: "missing_order_reference",
    DecisionLabel.OUT_OF_SCOPE: "info_only",
}


def _render(rng: random.Random, label: DecisionLabel, variant: str, ctx: Dict[str, Any]) -> str:
    order_ids = list(ctx["orders"].keys())
    oid = order_ids[0] if order_ids else "#unknown"
    if label is DecisionLabel.AUTO_APPROVE:
        if variant == "emotional":
            return rng.choice(_EMOTIONAL_OPENERS) + " The coffee maker from order " + oid + " arrived and I want my money back."
        return rng.choice(_TERSE) + " Order " + oid + "."
    if label is DecisionLabel.POLITELY_DENY:
        return rng.choice(_EMOTIONAL_OPENERS) + " I want to return the keyboard from order " + oid + " right now."
    if label is DecisionLabel.EXCHANGE:
        return "Hi, please exchange the thermostat in order " + oid + " for the Google Home compatible version. Happy to pay any difference."
    if label is DecisionLabel.CANCEL_ORDER:
        return "Please cancel order " + oid + ", I ordered it by mistake."
    if label is DecisionLabel.MODIFY_ORDER:
        return "Order " + oid + " should ship to my new address, and please switch payment to my gift card."
    if label is DecisionLabel.MODIFY_USER:
        return "Please update my default address on file to my new home address."
    if label is DecisionLabel.ESCALATE:
        return "Delete my entire account and all my data immediately, then confirm in writing to my lawyer."
    if label is DecisionLabel.REQUEST_PHOTO_PROOF:
        return "The espresso machine from order " + oid + " arrived cracked on one side and does not power on. I want a refund for this damaged item."
    if label is DecisionLabel.NEEDS_INFO:
        return "I want a refund for my order but I cannot remember the order number and I no longer have access to my email."
    return "Quick question: do you also price-match other retailers? Asking for a friend."


def _context_for(label: DecisionLabel, idx: int) -> Dict[str, Any]:
    uid = "synth_user_%03d" % idx
    orders: List[Dict[str, Any]]
    if label is DecisionLabel.POLITELY_DENY:
        orders = [_order("#PENDING%03d" % idx, "pending", _items())]
    elif label in (DecisionLabel.CANCEL_ORDER, DecisionLabel.MODIFY_ORDER):
        orders = [_order("#PENDMOD%03d" % idx, "pending", _items())]
    else:
        orders = [_order("#DELIV%03d" % idx, "delivered", _items())]
    ctx = _user_ctx(uid, "Casey", "Synth", "19104", orders)
    ctx["focus_order_id"] = orders[0]["order_id"]
    return ctx


def generate(n_per_family: int = 4, seed: int = 7, variants: bool = True) -> List[Example]:
    """Deterministic generator. Same seed, same examples, same order."""
    rng = random.Random(seed)
    out: List[Example] = []
    idx = 0
    for label, family in FAMILIES.items():
        for k in range(n_per_family):
            idx += 1
            ctx = _context_for(label, idx)
            variant = "verbatim"
            if variants and label is DecisionLabel.AUTO_APPROVE and k == 0:
                variant = "emotional"
            ex = Example(
                example_id="synth_%s_%03d" % (family, idx),
                text=_render(rng, label, variant, ctx),
                context=ctx,
                gold=label,
                split="fixture",
                family=family,
                variant=variant,
                provenance={"source": "synthetic", "seed": seed, "generator": "sdt.data.synth_tickets"},
            )
            out.append(ex)
    return out
