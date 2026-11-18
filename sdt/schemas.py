"""Core artifacts for the semantic decision tree prototype.

Stdlib-only by necessity (this sandbox has no package index access) and by
preference for Phase 0: every artifact validates itself via __post_init__ and
to_dict/from_dict round-trips, so swapping in pydantic later is mechanical.

Two schema facts worth knowing, both grounded in the pinned tau2-bench retail
sources (policy.md, data_model.py, tasks.json fetched 2026-11-13):

1. tau2 retail has no photo-proof or chargeback flow, so the label taxonomy is
   derived from the actual action space. request_photo_proof is kept only for
   the synthetic/production extension.
2. tau2 retail has no 30-day return window: any delivered order may be returned
   on confirmation. Computed questions must therefore derive from policy.md,
   never from the prototype's earlier assumptions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SdtError(ValueError):
    """Raised when an artifact fails validation."""


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class DecisionType(str, Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


class QuestionType(str, Enum):
    SEMANTIC = "semantic"    # answered by Jev against the unstructured text
    COMPUTED = "computed"    # answered exactly by code against the JSON context


class DecisionLabel(str, Enum):
    """Action-outcome taxonomy derived from the tau2-bench retail action space."""

    AUTO_APPROVE = "auto_approve"                # return/refund write executed
    BOOK_ORDER = "book_order"                    # new booking/reservation created
    EXCHANGE = "exchange"                        # exchange write executed
    CANCEL_ORDER = "cancel_order"                # pending order cancelled (refund)
    MODIFY_ORDER = "modify_order"                # pending order modified
    MODIFY_USER = "modify_user"                  # user profile write (default address)
    DEVICE_SETTING = "device_setting"            # telecom: device/network setting write
    PAYMENT = "payment"                          # telecom: payment/billing write
    # telecom fine-grained action labels (the domain's real decision space)
    TOGGLE_AIRPLANE_MODE = "toggle_airplane_mode"
    TOGGLE_ROAMING = "toggle_roaming"
    TOGGLE_DATA = "toggle_data"
    TOGGLE_DATA_SAVER = "toggle_data_saver_mode"
    TOGGLE_WIFI_CALLING = "toggle_wifi_calling"
    REFUEL_DATA = "refuel_data"
    REBOOT_DEVICE = "reboot_device"
    RESET_APN = "reset_apn_settings"
    RESEAT_SIM = "reseat_sim_card"
    DISCONNECT_VPN = "disconnect_vpn"
    GRANT_APP_PERMISSION = "grant_app_permission"
    SET_NETWORK_MODE = "set_network_mode_preference"
    RESUME_LINE = "resume_line"
    MAKE_PAYMENT = "make_payment"
    SEND_PAYMENT_REQUEST = "send_payment_request"
    POLITELY_DENY = "politely_deny"              # refund-ish intent, no write allowed
    ESCALATE = "escalate"                        # transfer_to_human_agents
    REQUEST_PHOTO_PROOF = "request_photo_proof"  # synthetic/production extension
    NEEDS_INFO = "needs_info"                    # multi-turn elicitation required
    OUT_OF_SCOPE = "out_of_scope"                # info-only or unrelated intent

    @classmethod
    def from_action_prefix(cls, action_name: str) -> Optional["DecisionLabel"]:
        for prefix, label in ACTION_PREFIX_MAP.items():
            if action_name.startswith(prefix):
                return label
        return None


ACTION_PREFIX_MAP: Dict[str, DecisionLabel] = {
    "return_delivered_order": DecisionLabel.AUTO_APPROVE,
    "exchange_delivered_order_items": DecisionLabel.EXCHANGE,
    "cancel_pending_order": DecisionLabel.CANCEL_ORDER,
    "modify_pending_order_items": DecisionLabel.MODIFY_ORDER,
    "modify_pending_order_address": DecisionLabel.MODIFY_ORDER,
    "modify_pending_order_payment": DecisionLabel.MODIFY_ORDER,
    "modify_user_address": DecisionLabel.MODIFY_USER,
    "transfer_to_human_agents": DecisionLabel.ESCALATE,
}

REFUND_INTENT_TOKENS = (
    "return", "refund", "money back", "exchange", "cancel",
    "get reimbursed", "chargeback", "cancel the order",
)

INFO_ONLY_PREFIXES = ("get_", "find_", "list_", "search_", "authenticate")


@dataclass
class Question:
    """An atomic yes/no question. The unit of both induction and inference."""

    text: str
    qtype: QuestionType
    qid: Optional[str] = None
    version: int = 1
    provenance: str = "seed"
    resolver: Optional[str] = None  # registered resolver name, computed only
    criteria: Optional[Dict[str, str]] = None  # true_description / false_description for nouls
    decision_type: "DecisionType" = DecisionType.NOUL
    options: Optional[Dict[str, str]] = None  # choice: branch name -> description (API-required)
    rubric: Optional[List[str]] = None        # score: ordered levels, lowest first (API-required)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise SdtError("question text must be a non-empty string")
        if not isinstance(self.qtype, QuestionType):
            raise SdtError("qtype must be a QuestionType")
        if self.version < 1:
            raise SdtError("version must be >= 1")
        if self.qid is None:
            self.qid = "q_" + stable_hash(self.text)
        if self.qtype is QuestionType.COMPUTED and not self.resolver:
            raise SdtError("computed question requires a resolver name")
        if self.qtype is QuestionType.SEMANTIC and self.resolver:
            raise SdtError("semantic question must not carry a resolver")

    @property
    def cache_key(self) -> str:
        """Cache identity: rewording a question invalidates exactly its column."""
        base = self.text + "|v" + str(self.version)
        if self.criteria:
            base += "|c:" + json.dumps(self.criteria, sort_keys=True)
        return stable_hash(base)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "text": self.text,
            "qtype": self.qtype.value,
            "version": self.version,
            "provenance": self.provenance,
            "resolver": self.resolver,
            "criteria": self.criteria,
            "decision_type": self.decision_type.value,
            "options": self.options,
            "rubric": self.rubric,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Question":
        return cls(
            text=d["text"],
            qtype=QuestionType(d["qtype"]),
            qid=d.get("qid"),
            version=int(d.get("version", 1)),
            provenance=d.get("provenance", "seed"),
            resolver=d.get("resolver"),
            criteria=d.get("criteria"),
            decision_type=DecisionType(d.get("decision_type", "noul")),
            options=d.get("options"),
            rubric=d.get("rubric"),
        )


@dataclass
class Example:
    """One training/eval instance: unstructured text + JSON context + gold label."""

    example_id: str
    text: str
    context: Dict[str, Any]
    gold: DecisionLabel
    split: str = "train"
    family: str = "unassigned"
    variant: str = "verbatim"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not isinstance(self.example_id, str):
            raise SdtError("example_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise SdtError("example text must be a non-empty string")
        if not isinstance(self.context, dict):
            raise SdtError("context must be a dict (the JSON side)")
        if not isinstance(self.gold, DecisionLabel):
            raise SdtError("gold must be a DecisionLabel")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "text": self.text,
            "context": self.context,
            "gold": self.gold.value,
            "split": self.split,
            "family": self.family,
            "variant": self.variant,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Example":
        return cls(
            example_id=d["example_id"],
            text=d["text"],
            context=d["context"],
            gold=DecisionLabel(d["gold"]),
            split=d.get("split", "train"),
            family=d.get("family", "unassigned"),
            variant=d.get("variant", "verbatim"),
            provenance=d.get("provenance", {}),
        )


@dataclass
class ScoreVector:
    """Per-example probability vector: qid -> p(True) in [0, 1]."""

    example_id: str
    scores: Dict[str, float]
    latency_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.scores and self.scores != {}:
            raise SdtError("scores must be a dict")
        for qid, p in self.scores.items():
            if not 0.0 <= float(p) <= 1.0:
                raise SdtError("probability out of range for %s: %r" % (qid, p))


@dataclass
class TreeNode:
    """Internal node (question + branch->child map) or leaf (label).

    Binary noul trees use branches {"yes": ..., "no": ...}; choice nodes
    have one branch per option; score nodes one branch per rubric level.
    """

    node_id: str
    question_qid: Optional[str] = None
    branches: Optional[Dict[str, str]] = None
    leaf_label: Optional[DecisionLabel] = None

    @property
    def child_yes(self) -> Optional[str]:
        return (self.branches or {}).get("yes")

    @property
    def child_no(self) -> Optional[str]:
        return (self.branches or {}).get("no")

    @property
    def is_leaf(self) -> bool:
        return self.leaf_label is not None


@dataclass
class Tree:
    """A flattened tree artifact: the unit of deployment."""

    tree_id: str
    root: str
    nodes: Dict[str, TreeNode]
    version: int = 1

    def __post_init__(self) -> None:
        if self.root not in self.nodes:
            raise SdtError("root %s not present in nodes" % self.root)
        self._validate_structure()

    def _validate_structure(self) -> None:
        seen: set = set()

        def visit(node_id: str, path: Tuple[str, ...]) -> None:
            if node_id in path:
                raise SdtError("cycle detected at %s" % node_id)
            if node_id in seen:
                return
            seen.add(node_id)
            node = self.nodes.get(node_id)
            if node is None:
                raise SdtError("dangling child reference: %s" % node_id)
            if node.is_leaf:
                if node.branches or node.question_qid:
                    raise SdtError("leaf %s must not carry children/questions" % node_id)
                return
            if not node.question_qid:
                raise SdtError("internal node %s requires a question" % node_id)
            if not node.branches or len(node.branches) < 2:
                raise SdtError("internal node %s requires >= 2 branches" % node_id)
            for child in node.branches.values():
                visit(child, path + (node_id,))

        visit(self.root, ())

    def to_dict(self, questions: Optional[Dict[str, "Question"]] = None) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "version": self.version,
            "root": self.root,
            "nodes": {
                nid: {
                    "question_qid": n.question_qid,
                    "branches": n.branches,
                    "leaf_label": n.leaf_label.value if n.leaf_label else None,
                }
                for nid, n in self.nodes.items()
            },
            "questions": {qid: q.to_dict() for qid, q in (questions or {}).items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> tuple:
        nodes = {}
        for nid, n in d["nodes"].items():
            branches = n.get("branches")
            if branches is None and (n.get("child_yes") or n.get("child_no")):
                branches = {}  # legacy binary artifact migration
                if n.get("child_yes"):
                    branches["yes"] = n["child_yes"]
                if n.get("child_no"):
                    branches["no"] = n["child_no"]
            nodes[nid] = TreeNode(
                node_id=nid,
                question_qid=n.get("question_qid"),
                branches=branches,
                leaf_label=DecisionLabel(n["leaf_label"]) if n.get("leaf_label") else None,
            )
        tree = cls(tree_id=d["tree_id"], root=d["root"], nodes=nodes, version=int(d.get("version", 1)))
        questions = [Question.from_dict(q) for q in d.get("questions", {}).values()]
        return tree, questions

    def questions(self) -> List[str]:
        return [n.question_qid for n in self.nodes.values() if n.question_qid]

    def leaves(self) -> List[str]:
        return [nid for nid, n in self.nodes.items() if n.is_leaf]


class GateOutcome(str, Enum):
    AUTO_EXECUTED = "auto_executed"
    ESCALATED_MARGIN = "escalated_margin"
    ESCALATED_BOTTLENECK = "escalated_bottleneck"
    ESCALATED_BOTH = "escalated_both"


@dataclass
class DecisionRecord:
    """Everything logged per inference request. Fuel for Phase 4 and distillation."""

    example_id: str
    tree_id: str
    scores: Dict[str, float]
    class_posterior: Dict[str, float]
    margin: float
    bottleneck_nodes: List[str]
    outcome: GateOutcome
    predicted: Optional[DecisionLabel] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "tree_id": self.tree_id,
            "scores": self.scores,
            "class_posterior": self.class_posterior,
            "margin": self.margin,
            "bottleneck_nodes": self.bottleneck_nodes,
            "outcome": self.outcome.value,
            "predicted": self.predicted.value if self.predicted else None,
            "latency_ms": self.latency_ms,
        }
