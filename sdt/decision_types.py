"""Multi-typed decision questions: noul (yes/no), choice (K-way), score (rubric).

Gemini Decision 1 answers three question shapes in ONE batch. We flatten
every shape into probability columns so the cache, combiner, and gates stay
type-agnostic:

  noul   -> one column: qid (probability of yes)
  choice -> K columns:  qid::option_name (probability per option)
  score  -> L columns:  qid::level_index (probability per rubric level)

A tree node over a choice question has K children; over a score question,
L children. The tree stops being binary.

NOTE on criteria: nouls here carry NO criteria (per experiment protocol).
For choice and score, the API structurally REQUIRES the option map / level
array inside its "criteria" field - that is where options are DEFINED, not
noul-style true/false disambiguation. We store them as first-class fields
(options, rubric) and serialize them into the request's criteria slot.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sdt.schemas import DecisionType
from sdt.schemas import Question, QuestionType, stable_hash


def validate_decision_shape(q: Question) -> None:
    dt = q.decision_type
    if dt is DecisionType.NOUL:
        if q.options or q.rubric:
            raise ValueError("noul question must not carry options/rubric")
    elif dt is DecisionType.CHOICE:
        if not q.options or not (2 <= len(q.options) <= 255):
            raise ValueError("choice question requires 2-255 options")
        if q.rubric:
            raise ValueError("choice question must not carry a rubric")
    elif dt is DecisionType.SCORE:
        if not q.rubric or len(q.rubric) < 2:
            raise ValueError("score question requires a rubric with >= 2 levels")
        if q.options:
            raise ValueError("score question must not carry options")
    else:
        raise ValueError("unknown decision type %r" % dt)


def decision_cache_key(q: Question) -> str:
    """Cache identity including the option map / rubric (shape is part of
    the question). Noul questions without criteria keep the legacy key."""
    base = q.text + "|v" + str(q.version)
    if q.decision_type is not DecisionType.NOUL:
        base += "|dt:" + q.decision_type.value
        if q.options:
            base += "|opt:" + ",".join(sorted(q.options.keys()))
        if q.rubric:
            base += "|rub:" + str(len(q.rubric))
    if q.criteria:
        base += "|c:" + __import__("json").dumps(q.criteria, sort_keys=True)
    return stable_hash(base)


def branch_names(q: Question) -> List[str]:
    if q.decision_type is DecisionType.CHOICE:
        return list(q.options.keys())
    if q.decision_type is DecisionType.SCORE:
        return [str(i) for i in range(len(q.rubric))]
    return ["yes", "no"]


def column_id(qid: str, branch: str) -> str:
    return qid + "::" + branch


def columns_of(q: Question) -> List[str]:
    if q.decision_type is DecisionType.NOUL:
        return [q.qid]
    return [column_id(q.qid, b) for b in branch_names(q)]


def attach(question: Question) -> Question:
    """Validate + re-key a multi-typed question (qid derives from text)."""
    validate_decision_shape(question)
    return question


def noul(text: str, provenance: str = "seed") -> Question:
    return Question(text=text, qtype=QuestionType.SEMANTIC, provenance=provenance)


def choice(text: str, options: Dict[str, str], provenance: str = "seed") -> Question:
    q = Question(text=text, qtype=QuestionType.SEMANTIC, provenance=provenance)
    q.decision_type = DecisionType.CHOICE
    q.options = dict(options)
    return attach(q)


def score(text: str, rubric: List[str], provenance: str = "seed") -> Question:
    q = Question(text=text, qtype=QuestionType.SEMANTIC, provenance=provenance)
    q.decision_type = DecisionType.SCORE
    q.rubric = list(rubric)
    return attach(q)
