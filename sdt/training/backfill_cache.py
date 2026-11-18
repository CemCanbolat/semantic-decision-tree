"""Backfill the score cache for a flattened feature set.

Induction scores each question only on the examples that reach its node.
Evaluating a flattened tree (or training a combiner) needs every question
against every example. This command fills exactly the missing pairs,
skipping anything already cached. Live-only: needs network + GEMINI_API_KEY.

    python3.11 -m sdt.training.backfill_cache \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json \
        --trees artifacts/trees/induced_v3.json artifacts/trees/induced_v1.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.schemas import Question, QuestionType, ScoreVector, Tree
from sdt.training.cache import ScoreCache


def collect_questions(tree_paths):
    seen, out = set(), []
    for path in tree_paths:
        _, questions = Tree.from_dict(__import__("json").loads(Path(path).read_text()))
        for q in questions:
            if q.qtype is QuestionType.SEMANTIC and q.text not in seen:
                seen.add(q.text)
                out.append(q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--trees", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    questions = collect_questions(args.trees)
    client = GeminiDecisionClient()

    missing_total = sum(len(cache.missing(q, [e.example_id for e in examples])) for q in questions)
    print("feature pool: %d semantic questions x %d examples | missing pairs: %d" % (
        len(questions), len(examples), missing_total))
    if missing_total == 0:
        print("cache complete; nothing to do")
        return

    started = time.time()
    done = 0
    for i, ex in enumerate(examples, 1):
        todo = [q for q in questions if cache.get(q, ex.example_id) is None]
        if not todo:
            continue
        sv = client.score_batch(ex.text, ex.context, todo)
        for q in todo:
            cache.put(q, ex.example_id, sv.scores[q.qid])
        done += 1
        if i % 20 == 0:
            cache.save()
            print("  backfilled %d/%d examples" % (i, len(examples)), file=sys.stderr)
    cache.save()
    print("done: %d examples scored in %.1fs (one batch call each); cache saved" % (
        done, time.time() - started))


if __name__ == "__main__":
    main()
