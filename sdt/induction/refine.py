"""FP/FN-driven question refinement (TextGrad/LLEGO-style mutation)."""

from __future__ import annotations

from typing import List, Tuple

from sdt.schemas import Example

MAX_SHOW = 5


def build_refine_prompt(question_text: str,
                        false_positives: List[Example],
                        false_negatives: List[Example]) -> Tuple[str, str]:
    def block(exs: List[Example]) -> str:
        return chr(10).join("- " + e.text.replace(chr(10), " ")[:200] for e in exs[:MAX_SHOW]) or "- (none)"

    system = (
        "You improve yes/no screening questions. Keep the question atomic and factual. "
        "Rewrite it so the misclassified cases are answered correctly. "
        "Return up to 5 rewritten variants, one per line, no numbering."
    )
    user = chr(10).join([
        "Current question:",
        question_text,
        "",
        "Answered YES but should be NO:",
        block(false_positives),
        "",
        "Answered NO but should be YES:",
        block(false_negatives),
        "",
        "Rewrite the question to fix these errors without breaking the cases it already gets right.",
    ])
    return system, user
