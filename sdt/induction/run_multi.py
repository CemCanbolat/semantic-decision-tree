"""Multi-seed induction: grow diverse trees on purpose, bank every question.

Iteration 5 showed single trees are lottery draws but their questions
compound. This runner induces N trees with different sampling seeds (the
frontier also samples at temperature), saves each artifact plus candidates
log, deposits every picked question into the shared bank, and leaves the
cache ready for bank selection.

    python3.11 -m sdt.induction.run_multi \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --n-trees 3 --frontier gemini --scorer gemini \
        --cache cache/scores.json --bank artifacts/question_bank.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.data.synth_tickets import generate
from sdt.eval.heuristic_jev import HeuristicJevClient
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.grow import TreeInducer
from sdt.induction.run_induction import GeminiFrontierClient, SCRIPTED_PROPOSAL
from sdt.llm_client import MockFrontierClient
from sdt.training.cache import ScoreCache
from sdt.training.question_bank import QuestionBank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--n-trees", type=int, default=3)
    ap.add_argument("--base-seed", type=int, default=11)
    ap.add_argument("--frontier", choices=["gemini", "scripted"], default="gemini")
    ap.add_argument("--scorer", choices=["gemini", "heuristic"], default="gemini")
    ap.add_argument("--cache", type=Path, default=Path("cache/scores.json"))
    ap.add_argument("--bank", type=Path, default=Path("artifacts/question_bank.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/trees"))
    args = ap.parse_args()

    examples = (load_examples(str(args.corpus)) if args.corpus
                else generate(n_per_family=4, seed=7))
    cache = ScoreCache(str(args.cache))
    bank = QuestionBank(str(args.bank))
    ts = time.strftime("%Y%m%d_%H%M%S")

    for i in range(args.n_trees):
        seed = args.base_seed + i
        frontier = (GeminiFrontierClient() if args.frontier == "gemini"
                    else MockFrontierClient(scripted=[SCRIPTED_PROPOSAL] * 64))
        scorer = (GeminiDecisionClient() if args.scorer == "gemini" else HeuristicJevClient())
        inducer = TreeInducer(frontier=frontier, scorer=scorer, cache=cache,
                              corpus_texts=[e.text for e in examples], seed=seed)
        tree_id = "ms%d_%s" % (i, ts)
        started = time.time()
        try:
            tree = inducer.induce(examples, tree_id=tree_id)
        finally:
            cache.save()
        questions = list(inducer._questions.values())
        added = bank.deposit(questions, run_id=tree_id)
        bank.save()
        out = args.out_dir / ("induced_%s.json" % tree_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tree.to_dict(inducer._questions), indent=2), encoding="utf-8")
        cand_path = out.with_suffix(".candidates.json")
        cand_path.write_text(json.dumps(inducer.stats.candidates, indent=2), encoding="utf-8")
        print("[multi %d/%d seed %d] %d questions %d leaves | bank +%d -> %d | %.0fs" % (
            i + 1, args.n_trees, seed, len(questions), len(tree.leaves()),
            added, len(bank), time.time() - started), flush=True)

    print("bank: %s | %d questions | runs: %s" % (
        args.bank, len(bank), ",".join(bank.summary()["runs"][-(args.n_trees + 2):])))


if __name__ == "__main__":
    main()
