"""CLI runner for the M1 baselines on a curated corpus.

Offline default uses HeuristicJevClient (a keyword stand-in, not a model of
Jev quality); reports are stamped with the scorer name so nobody mistakes a
plumbing run for a result. With credentials, plug in HttpJevClient and the
same command produces the real B2 numbers.

    python3.11 -m sdt.eval.run_baselines --corpus data/curated/tau2_retail/examples.jsonl \
        --out eval/reports/m1_tau2.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.data.synth_tickets import generate
from sdt.eval import metrics
from sdt.eval.baselines import run_b2_hard, run_b2_soft
from sdt.eval.heuristic_jev import HeuristicJevClient
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.schemas import SdtError


def summarize(name: str, records, examples, scorer: str) -> dict:
    return {
        "baseline": name,
        "scorer": scorer,
        "n": len(records),
        "accuracy": round(metrics.accuracy(records, examples), 4),
        "macro_f1": round(metrics.macro_f1(records, examples), 4),
        "coverage_at_0.97": metrics.coverage_at_precision(records, examples, 0.97),
        "confusion": metrics.confusion_matrix(records, examples),
        "outcome_counts": {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="examples.jsonl; omit to run on the seeded synthetic corpus")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scorer", choices=["heuristic", "gemini"], default="heuristic",
                    help="gemini requires GEMINI_API_KEY and network")
    args = ap.parse_args()

    if args.corpus:
        examples = load_examples(str(args.corpus))
        corpus_name = str(args.corpus)
    else:
        examples = generate(n_per_family=4, seed=7)
        corpus_name = "synthetic fixture corpus (n_per_family=4, seed=7)"

    if args.scorer == "gemini":
        client = GeminiDecisionClient()
        if not client.api_key:
            raise SystemExit("GEMINI_API_KEY is not set; see sdt/gemini_decision_client.py")
    else:
        client = HeuristicJevClient()
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": corpus_name,
        "warning": "scores produced by " + client.scorer_name + (" - plumbing validation only, not model quality" if args.scorer == "heuristic" else " - live Gemini Decision 1 run"),
        "baselines": [],
    }

    soft = run_b2_soft(client, examples)
    hard = run_b2_hard(client, examples)
    for name, records in (("B2_hand_tree_soft", soft), ("B2_hand_tree_hard", hard)):
        s = summarize(name, records, examples, client.scorer_name)
        counts: dict = {}
        for rec in records:
            counts[rec.outcome.value] = counts.get(rec.outcome.value, 0) + 1
        s["outcome_counts"] = counts
        report["baselines"].append(s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    for b in report["baselines"]:
        print(b["baseline"], "acc", b["accuracy"], "macro_f1", b["macro_f1"],
              "cov@0.97", b["coverage_at_0.97"])
    print("report ->", args.out)


if __name__ == "__main__":
    main()
