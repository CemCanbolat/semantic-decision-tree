"""Contrastive split proposal via the frontier LLM, with a leakage guard.

The prompt shows grouped positives and negatives (that is how the model
learns what to separate) but questions referencing the label terms are
rejected before any scoring happens.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from sdt.schemas import DecisionLabel, Example, Question, QuestionType

LABEL_TERMS = tuple(
    ["auto approv", "politely den", "escalate to", "escalated by",
     "out of scope", "needs info", "photo proof", "request photo"]
)

MAX_EXAMPLES_PER_SIDE = 8
MAX_TEXT_CHARS = 220
TARGET_CANDIDATES = 10


def _example_block(examples: List[Example]) -> str:
    lines = []
    for e in examples[:MAX_EXAMPLES_PER_SIDE]:
        text = e.text.replace(chr(10), " ")[:MAX_TEXT_CHARS]
        lines.append("- " + text)
    return chr(10).join(lines)


def build_proposal_prompt(pos: List[Example], neg: List[Example],
                          pos_label: DecisionLabel, neg_label: DecisionLabel) -> Tuple[str, str]:
    system = (
        "You write atomic yes/no screening questions for a support adjudication system. "
        "Questions must be about observable facts in the ticket text or the account/order JSON, "
        "phrased so a careful reader answers yes or no with high confidence. "
        "Never reference classification labels or decision outcomes. "
        "Phrase every question positively - ask about what IS present, never about what is absent or omitted. "
        "Prefer features that would generalize to unseen tickets; avoid naming specific products. "
        "One condition per question."
    )
    user = chr(10).join([
        "Below are two groups of support tickets.",
        "",
        "GROUP A (label: " + pos_label.value + "):",
        _example_block(pos),
        "",
        "GROUP B (label: " + neg_label.value + "):",
        _example_block(neg),
        "",
        "Propose up to " + str(TARGET_CANDIDATES) + " yes/no questions that are answered YES for most of GROUP A",
        "and NO for most of GROUP B. One question per line, no numbering, no explanations.",
        "Each question must be self-contained and grounded in what the tickets say.",
    ])
    return system, user


def leakage_check(text: str) -> bool:
    blob = text.lower().replace("-", " ")
    return any(term in blob for term in LABEL_TERMS)


def parse_questions(raw: str, provenance: str) -> List[Question]:
    out: List[Question] = []
    seen = set()
    for line in raw.splitlines():
        q = line.strip().lstrip("-*). ").lstrip("0123456789.*-) ").strip()
        if not q or len(q) < 8 or not q.endswith("?"):
            continue
        if leakage_check(q):
            continue
        if is_negated(q):
            continue
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(Question(text=q, qtype=QuestionType.SEMANTIC, provenance=provenance))
        if len(out) >= TARGET_CANDIDATES:
            break
    return out


def contrastive_pair(examples: List[Example]) -> Optional[Tuple[DecisionLabel, DecisionLabel]]:
    """The two most frequent labels at this node define the contrastive pair."""
    counts: Dict[DecisionLabel, int] = {}
    for e in examples:
        counts[e.gold] = counts.get(e.gold, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].value))
    if len(ranked) < 2:
        return None
    return ranked[0][0], ranked[1][0]


NEGATION_MARKERS = (
    "omit", "lacks ", "lack ", "fail to mention", "fails to mention",
    "no mention of", "without mentioning", "not mention", "does not contain",
    "doesn't contain", "never mention",
)


def is_negated(text: str) -> bool:
    blob = text.lower()
    return any(m in blob for m in NEGATION_MARKERS)


def build_ovr_prompt(pos: List[Example], pos_label: DecisionLabel,
                     neg: List[Example]) -> Tuple[str, str]:
    """One-vs-rest contrastive prompt for a minority class."""
    system = (
        "You write atomic yes/no screening questions for a support adjudication system. "
        "Questions must be about observable facts in the ticket text or the account/order JSON. "
        "Never reference classification labels or decision outcomes. "
        "Phrase every question positively - ask about what IS present, never about what is absent. "
        "Prefer features that would generalize to unseen tickets; avoid naming specific products."
    )
    user = chr(10).join([
        "Below are two groups of support tickets.",
        "",
        "GROUP A (label: " + pos_label.value + "):",
        _example_block(pos),
        "",
        "GROUP B (label: other):",
        _example_block(neg),
        "",
        "Propose up to " + str(TARGET_CANDIDATES) + " yes/no questions that are answered YES for most of GROUP A",
        "and NO for most of GROUP B. One question per line, no numbering, no explanations.",
    ])
    return system, user


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in re.split(r"[^A-Za-z0-9']+", text) if len(w) > 3]


def rare_fraction(question_text: str, corpus_texts: List[str], min_df: int = 3) -> float:
    """Fraction of the question's tokens that are rare in the corpus.

    Memorized questions name specific products that appear in few tickets;
    generalizable questions use common words. A document-frequency prior,
    not a hand-curated noun blocklist.
    """
    df: Dict[str, int] = {}
    for text in corpus_texts:
        for w in set(_tokens(text)):
            df[w] = df.get(w, 0) + 1
    toks = set(_tokens(question_text))
    if not toks:
        return 0.0
    rare = sum(1 for w in toks if df.get(w, 0) < min_df)
    return rare / len(toks)
