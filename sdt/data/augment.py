"""Label-preserving augmentation: tone and distractor perturbations.

v1 is rule-based and deterministic (an emotional opener prepended, facts
untouched). The LLM-paraphrase variant from the plan lands here later behind
the same interface, with the hand audit for factual drift.
"""

from __future__ import annotations

import random
from typing import List

from sdt.schemas import DecisionLabel, Example

EMOTIONAL_OPENERS = [
    "I have been a loyal customer for years and this is unacceptable!",
    "I am extremely disappointed and expect this handled immediately.",
    "Honestly, this whole experience has been infuriating. Do better.",
]

_DISTRACTORS = [
    "By the way, I also browsed your winter catalog last week.",
    "Unrelated: please add me to the newsletter.",
    "Also, your website was slow yesterday.",
]


def emotional_variant(example: Example, opener: str) -> Example:
    """Prepend an emotional opener without touching any factual field."""
    return Example(
        example_id=example.example_id + "__emotional",
        text=opener + " " + example.text,
        context=dict(example.context),
        gold=example.gold,
        split=example.split,
        family=example.family,
        variant="emotional",
        provenance={**example.provenance, "augmented_from": example.example_id},
    )


def distract(example: Example, tail: str) -> Example:
    return Example(
        example_id=example.example_id + "__distract",
        text=example.text + " " + tail,
        context=dict(example.context),
        gold=example.gold,
        split=example.split,
        family=example.family,
        variant=example.variant if example.variant != "verbatim" else "distracted",
        provenance={**example.provenance, "augmented_from": example.example_id},
    )


def augment(examples: List[Example], seed: int = 7, emotional_families: tuple = ("return_delivered_clean", "return_pending_denied")) -> List[Example]:
    """Add one emotional variant per example in the given families; deterministic."""
    rng = random.Random(seed)
    out: List[Example] = list(examples)
    for ex in examples:
        if ex.family in emotional_families:
            out.append(emotional_variant(ex, rng.choice(EMOTIONAL_OPENERS)))
    return out
