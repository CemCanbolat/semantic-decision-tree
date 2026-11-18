"""Baselines: B1 single-prompt classifier, B2 hand-written question tree.

The B2 question set is derived from the actual tau2 retail policy (not the
prototype's original assumptions): returns require a delivered order, cancels
and modifications require a pending order, out-of-scope requests transfer to
a human, and orders that are neither delivered nor pending cannot be acted on.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

from sdt.jev_client import JevClient
from sdt.llm_client import FrontierClient
from sdt.routing.gates import decide
from sdt.routing.soft_tree import route
from sdt.schemas import (
    DecisionLabel,
    DecisionRecord,
    Example,
    GateOutcome,
    Question,
    QuestionType,
    ScoreVector,
    SdtError,
    Tree,
    TreeNode,
)

CLASS_LIST = [l.value for l in DecisionLabel]


def hand_tree_questions() -> List[Question]:
    return [
        Question(text="Does the user request a concrete account or order action rather than only asking for information?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
        Question(text="Is the request within the agent scope so that no transfer to a human agent is required?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
        Question(text="Does the user provide the order id or enough details to identify the order in question?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
        Question(text="Is the referenced order status delivered?",
                 qtype=QuestionType.COMPUTED, resolver="order_status_delivered", provenance="hand-b2"),
        Question(text="Does the user request a return or refund of items from the order?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
        Question(text="Does the user request an exchange of items for a different variant of the same product?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
        Question(text="Is the referenced order status pending?",
                 qtype=QuestionType.COMPUTED, resolver="order_status_pending", provenance="hand-b2"),
        Question(text="Does the user request cancellation of a pending order?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
        Question(text="Does the user request a modification of the pending order such as address, payment, or item options?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
        Question(text="Does the request concern a specific order rather than the user profile or account settings?",
                 qtype=QuestionType.SEMANTIC, provenance="hand-b2"),
    ]


def build_hand_tree() -> Tuple[Tree, List[Question]]:
    questions = hand_tree_questions()
    q = {i: question.qid for i, question in enumerate(questions)}

    def leaf(nid: str, label: DecisionLabel) -> TreeNode:
        return TreeNode(nid, leaf_label=label)

    def internal(nid: str, qi: int, yes: str, no: str) -> TreeNode:
        return TreeNode(nid, question_qid=q[qi], branches={'yes': yes, 'no': no})

    nodes = {
        "n_action": internal("n_action", 0, "n_scope", "leaf_out"),
        "n_scope": internal("n_scope", 1, "n_order_concern", "leaf_esc"),
        "n_order_concern": internal("n_order_concern", 9, "n_identify", "leaf_modify_user"),
        "n_identify": internal("n_identify", 2, "n_status", "leaf_needs"),
        "n_status": internal("n_status", 3, "n_return", "n_pending"),
        "n_return": internal("n_return", 4, "leaf_auto", "n_exchange"),
        "n_exchange": internal("n_exchange", 5, "leaf_exchange", "leaf_deny_delivered"),
        "n_pending": internal("n_pending", 6, "n_cancel", "leaf_deny_status"),
        "n_cancel": internal("n_cancel", 7, "leaf_cancel", "n_modify"),
        "n_modify": internal("n_modify", 8, "leaf_modify", "leaf_deny_other"),
        "leaf_out": leaf("leaf_out", DecisionLabel.OUT_OF_SCOPE),
        "leaf_esc": leaf("leaf_esc", DecisionLabel.ESCALATE),
        "leaf_needs": leaf("leaf_needs", DecisionLabel.NEEDS_INFO),
        "leaf_auto": leaf("leaf_auto", DecisionLabel.AUTO_APPROVE),
        "leaf_exchange": leaf("leaf_exchange", DecisionLabel.EXCHANGE),
        "leaf_deny_delivered": leaf("leaf_deny_delivered", DecisionLabel.POLITELY_DENY),
        "leaf_cancel": leaf("leaf_cancel", DecisionLabel.CANCEL_ORDER),
        "leaf_modify": leaf("leaf_modify", DecisionLabel.MODIFY_ORDER),
        "leaf_modify_user": leaf("leaf_modify_user", DecisionLabel.MODIFY_USER),
        "leaf_deny_status": leaf("leaf_deny_status", DecisionLabel.POLITELY_DENY),
        "leaf_deny_other": leaf("leaf_deny_other", DecisionLabel.POLITELY_DENY),
    }
    return Tree("b2_hand_retail_v1", "n_action", nodes), questions


def _score_all(client: JevClient, examples: List[Example], questions: List[Question]) -> List[ScoreVector]:
    out = []
    for i, ex in enumerate(examples, 1):
        sv = client.score_batch(ex.text, ex.context, questions)
        sv.example_id = ex.example_id
        out.append(sv)
        if i % 10 == 0 or i == len(examples):
            print("  scored %d/%d" % (i, len(examples)), file=sys.stderr)
    return out


def run_b2_soft(client: JevClient, examples: List[Example],
                tree: Optional[Tree] = None,
                questions: Optional[List[Question]] = None) -> List[DecisionRecord]:
    """Flatten -> single batch per example -> soft routing -> gates."""
    if tree is None:
        tree, questions = build_hand_tree()
    else:
        by_qid = {q.qid: q for q in (questions or [])}
        questions = [by_qid[qid] for qid in tree.questions() if qid in by_qid]
        if len(questions) != len(tree.questions()):
            raise SdtError("tree references unknown questions")
    vectors = _score_all(client, examples, questions)
    records: List[DecisionRecord] = []
    for ex, sv in zip(examples, vectors):
        r = route(tree, sv, {q.qid: q for q in questions})
        g = decide(r)
        records.append(DecisionRecord(
            example_id=ex.example_id,
            tree_id=tree.tree_id,
            scores=sv.scores,
            class_posterior={k.value: v for k, v in r.class_posterior.items()},
            margin=g.margin,
            bottleneck_nodes=g.bottleneck_nodes,
            outcome=g.outcome,
            predicted=g.top,
            latency_ms=sv.latency_ms,
        ))
    return records


def run_b2_hard(client: JevClient, examples: List[Example]) -> List[DecisionRecord]:
    """Hard splits: argmax branch at each node, no abstention. The fragile baseline."""
    tree, questions = build_hand_tree()
    vectors = _score_all(client, examples, questions)
    records: List[DecisionRecord] = []
    for ex, sv in zip(examples, vectors):
        cur = tree.root
        while not tree.nodes[cur].is_leaf:
            node = tree.nodes[cur]
            p = sv.scores[node.question_qid]
            cur = node.child_yes if p >= 0.5 else node.child_no
        label = tree.nodes[cur].leaf_label
        records.append(DecisionRecord(
            example_id=ex.example_id,
            tree_id=tree.tree_id,
            scores=sv.scores,
            class_posterior={label.value: 1.0},
            margin=1.0,
            bottleneck_nodes=[],
            outcome=GateOutcome.AUTO_EXECUTED,
            predicted=label,
            latency_ms=sv.latency_ms,
        ))
    return records


def build_b1_prompt(ex: Example) -> Tuple[str, str]:
    system = (
        "You are a retail support adjudicator. Classify the ticket into exactly "
        "one of: " + ", ".join(CLASS_LIST) + ". Answer with the label only."
    )
    user = "Ticket:" + chr(10) + ex.text + chr(10) + chr(10) + "Account/order JSON:" + chr(10) + str(ex.context)
    return system, user


def run_b1(client: FrontierClient, examples: List[Example]) -> List[DecisionRecord]:
    records: List[DecisionRecord] = []
    for ex in examples:
        system, user = build_b1_prompt(ex)
        raw = client.complete(system, user)
        try:
            pred = DecisionLabel(raw.strip().lower())
        except ValueError:
            pred = None
        records.append(DecisionRecord(
            example_id=ex.example_id,
            tree_id="b1_single_prompt",
            scores={},
            class_posterior={pred.value: 1.0} if pred else {},
            margin=1.0 if pred else 0.0,
            bottleneck_nodes=[],
            outcome=GateOutcome.AUTO_EXECUTED if pred else GateOutcome.ESCALATED_MARGIN,
            predicted=pred,
        ))
    return records
