"""LLM-generated disambiguation criteria for noul questions.

The Decisions API accepts optional criteria (true_description /
false_description) that pin down exactly what yes and no mean - the
API-native cure for boundary-case ambiguity. Generated once per question
by the frontier model, offline; they ride along in the Question artifact
and change its cache identity (a question with criteria is a new scoring
unit, so old cached columns stay untouched for A/B).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from sdt.llm_client import FrontierClient
from sdt.schemas import Question, QuestionType


def build_criteria_prompt(question: Question) -> Tuple[str, str]:
    system = (
        "You write precise disambiguation criteria for yes/no screening "
        "questions scored by a decision model. Be concrete about boundary "
        "cases: the false description must cover the nearest confusable cases, "
        "not just the opposite."
    )
    user = chr(10).join([
        "Domain: retail customer support (orders, returns, exchanges, cancellations, account changes).",
        "Question: " + question.text,
        "",
        "Answer with exactly two lines:",
        "TRUE: one sentence describing precisely when the answer is yes",
        "FALSE: one sentence describing precisely when it is no",
        "Each line under 25 words. Do not restate the question.",
    ])
    return system, user


def parse_criteria(raw: str) -> Optional[dict]:
    true_line, false_line = None, None
    for line in raw.splitlines():
        blob = line.strip()
        low = blob.lower()
        if low.startswith("true:"):
            true_line = blob[5:].strip()
        elif low.startswith("false:"):
            false_line = blob[6:].strip()
    if not true_line or not false_line:
        return None
    return {"true_description": true_line, "false_description": false_line}


def add_criteria(frontier: FrontierClient, questions: List[Question],
                 skip_computed: bool = True) -> Tuple[List[Question], List[dict]]:
    """Returns (questions with criteria attached, per-question status log).
    Questions are mutated in place; failures leave criteria unset."""
    log = []
    for q in questions:
        if skip_computed and q.qtype is QuestionType.COMPUTED:
            continue
        if q.criteria:
            log.append({"question": q.text, "status": "already-had-criteria"})
            continue
        system, user = build_criteria_prompt(q)
        raw = frontier.complete(system, user)
        crit = parse_criteria(raw)
        if crit is None:
            log.append({"question": q.text, "status": "parse-failed"})
            continue
        q.criteria = crit
        log.append({"question": q.text, "status": "ok",
                    "true": crit["true_description"], "false": crit["false_description"]})
    return questions, log
