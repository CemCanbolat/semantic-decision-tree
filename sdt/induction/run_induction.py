"""CLI: induce a tree from a corpus, evaluate it, save the artifact.

Offline demo (scripted frontier + heuristic scorer, no network):

    python3.11 -m sdt.induction.run_induction --frontier scripted --scorer heuristic --out artifacts/trees/demo.json

Live (networked machine with GEMINI_API_KEY): proposals from a Gemini chat
model, candidate scoring from gemini-decision-1:

    python3.11 -m sdt.induction.run_induction --corpus data/curated/tau2_retail/examples.jsonl --frontier gemini --scorer gemini --out artifacts/trees/induced_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.data.synth_tickets import generate
from sdt.eval import metrics
from sdt.eval.baselines import run_b2_soft
from sdt.eval.heuristic_jev import HeuristicJevClient
from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.grow import TreeInducer
from sdt.llm_client import FrontierClient, MockFrontierClient
from sdt.schemas import Question, SdtError
from sdt.training.cache import ScoreCache

SCRIPTED_PROPOSAL = (
    "Does the user request a return or refund of items from the order?" + chr(10)
    + "Does the ticket mention the weather?" + chr(10)
    + "Is the message written in English?"
)


class GeminiFrontierClient(FrontierClient):
    """Minimal generateContent wrapper for proposal/refinement prompts."""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or os.environ.get("GEMINI_FRONTIER_MODEL", "gemini-flash-latest")

    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        if not self.api_key:
            raise SdtError("GEMINI_API_KEY not set for GeminiFrontierClient")
        body = json.dumps({
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/" + self.model + ":generateContent",
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        import time as _time
        last_err = None
        for attempt in range(4):
            if attempt:
                _time.sleep(min(2.0 * (2 ** (attempt - 1)), 16.0))
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as e:
                err_body = e.read()[:300].decode("utf-8", "replace")
                last_err = SdtError("frontier HTTP %d: %s" % (e.code, err_body))
                if e.code not in (429, 500, 502, 503, 504):
                    raise last_err
                continue
            except (urllib.error.URLError, OSError) as e:
                last_err = SdtError("frontier network error: %s" % e)
                continue
        raise last_err or SdtError("frontier failed after retries")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="examples.jsonl; omit for the seeded synthetic corpus")
    ap.add_argument("--frontier", choices=["scripted", "gemini"], default="scripted")
    ap.add_argument("--scorer", choices=["heuristic", "gemini"], default="heuristic")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("cache/scores.json"))
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args()

    examples = (load_examples(str(args.corpus)) if args.corpus
                else generate(n_per_family=4, seed=7))

    frontier = (GeminiFrontierClient() if args.frontier == "gemini"
                else MockFrontierClient(scripted=[SCRIPTED_PROPOSAL] * 64))
    scorer = (GeminiDecisionClient() if args.scorer == "gemini" else HeuristicJevClient())
    cache = ScoreCache(str(args.cache))

    started = time.time()
    inducer = TreeInducer(frontier=frontier, scorer=scorer, cache=cache,
                          max_depth=args.max_depth,
                          corpus_texts=[e.text for e in examples])
    try:
        tree = inducer.induce(examples, tree_id="induced_" + time.strftime("%Y%m%d_%H%M%S"))
    finally:
        cache.save()  # survive crashes: paid-for scores must never be lost

    questions = list(inducer._questions.values())
    records = run_b2_soft(scorer, examples, tree=tree, questions=questions)
    acc = metrics.accuracy(records, examples)
    f1 = metrics.macro_f1(records, examples)
    cov = metrics.coverage_at_precision(records, examples, 0.97)

    artifact = tree.to_dict(inducer._questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    candidates_path = args.out.with_suffix(".candidates.json")
    candidates_path.write_text(json.dumps(inducer.stats.candidates, indent=2), encoding="utf-8")

    print("induced in %.1fs: %d nodes, %d questions, %d leaves (pruned %d)" % (
        time.time() - started, len(tree.nodes), len(questions), len(tree.leaves()),
        inducer.stats.pruned_subtrees))
    print("frontier calls: %d | scorer calls: %d | cache hits: %d" % (
        inducer.stats.frontier_calls, inducer.stats.scorer_calls, inducer.stats.cache_hits))
    print("corpus: %d examples | accuracy %.4f | macro_f1 %.4f | cov@0.97 %s" % (
        len(examples), acc, f1, json.dumps(cov)))
    for row in inducer.stats.per_node:
        print("  depth %d q: %s (ig %.3f, cert %.3f)" % (
            row["depth"], row["question"], row["ig"], row["certainty"]))
    n_cand = len(inducer.stats.candidates)
    n_neg = sum(1 for c in inducer.stats.candidates if c["picked"] is False and c["val_score"] is not None and c["val_score"] <= c["baseline"])
    print("candidates considered: %d (rejected-by-validation-or-baseline: %d)" % (n_cand, n_neg))
    print("candidates log ->", candidates_path)
    print("tree ->", args.out)


if __name__ == "__main__":
    main()
