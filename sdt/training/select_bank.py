"""Greedy forward selection over the question bank by CV macro-F1.

The bank compounds only if selection prunes: start from the four exact
computed features, then greedily add the banked question that most improves
cross-validated macro-F1, stopping when nothing helps by more than epsilon.

Resumable and time-budgeted: each invocation evaluates as many remaining
candidates as fit in the budget, persists per-round scores, and exits.
Call repeatedly until state.done is true. Pure cache - zero API calls.

    python3.11 -m sdt.training.select_bank \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --state artifacts/bank_selection.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sdt.data.loader import load_examples
from sdt.induction.gain import class_weights, evaluate_question
from sdt.jev_client import COMPUTED_SEEDS
from sdt.schemas import DecisionLabel, Example
from sdt.training.cache import ScoreCache
from sdt.training.combiner import cross_validate
from sdt.training.question_bank import QuestionBank

TOP_CANDIDATES = 12
EPSILON = 0.003


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--time-budget", type=float, default=22.0)
    ap.add_argument("--iters", type=int, default=700)
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    cache = ScoreCache(str(args.cache))
    bank = QuestionBank(str(args.bank))
    banked = bank.semantic_questions()
    labels = sorted({e.gold for e in examples}, key=lambda l: l.value)
    y = [{l: i for i, l in enumerate(labels)}[e.gold] for e in examples]
    wmap = class_weights(examples)
    sample_w = [wmap[e.gold] for e in examples]

    state = {"included": [], "remaining": [], "round": 0, "scores": {},
             "history": [], "done": False, "baseline": None}
    if args.state.exists():
        state.update(json.loads(args.state.read_text(encoding="utf-8")))

    def cv_macro_f1(included_texts):
        from sdt.training.combiner import build_features
        qs = [q for q in banked if q.text in included_texts]
        X, _ = build_features(examples, qs, cache, include_computed=True)
        _, _, f1 = cross_validate(X, y, sample_w, labels, iters=args.iters)
        return f1

    started = time.time()

    if state["baseline"] is None:
        state["baseline"] = cv_macro_f1([])
        state["history"].append({"round": 0, "event": "baseline (computed only)",
                                 "macro_f1": round(state["baseline"], 4)})
        ranked = []
        for q in banked:
            if q.decision_type.value == "noul":
                ps = [cache.get(q, e.example_id) for e in examples]
                if any(p is None for p in ps):
                    continue
                info = evaluate_question(examples, q.qid, ps, weight_map=wmap)
                ranked.append((info.score, q.text))
            else:
                # multi-type admission: full column coverage + multi-way split score
                if any(cache.get_columns(q, e.example_id) is None for e in examples):
                    continue
                from sdt.experiments.soft_grower import split_score
                res = split_score([(e, 1.0) for e in examples], q, cache, wmap)
                if res is not None:
                    ranked.append((res[0], q.text))
        ranked.sort(reverse=True)
        state["remaining"] = [t for _, t in ranked[:TOP_CANDIDATES]]
        print("baseline (computed only) macro_f1 %.4f | %d candidates admitted" % (
            state["baseline"], len(state["remaining"])))

    while not state["done"] and state["remaining"]:
        if time.time() - started > args.time_budget:
            print("time budget reached; state persisted (round %d, %d remaining)" % (
                state["round"], len(state["remaining"])))
            break
        cand = next((t for t in state["remaining"]
                     if ("r%d::%s" % (state["round"], t)) not in state["scores"]), None)
        if cand is not None:
            key = "r%d::%s" % (state["round"], cand)
            trial = list(state["included"]) + [cand]
            state["scores"][key] = round(cv_macro_f1(trial), 4)
            print("  eval r%d %-58s macro_f1 %.4f" % (
                state["round"], cand[:58], state["scores"][key]), flush=True)
        # round complete?
        round_keys = ["r%d::%s" % (state["round"], t) for t in state["remaining"]]
        if all(k in state["scores"] for k in round_keys):
            best_key = max(round_keys, key=lambda k: state["scores"][k])
            best_f1 = state["scores"][best_key]
            prev = state["history"][-1]["macro_f1"] if state["history"] else state["baseline"]
            if best_f1 > prev + EPSILON:
                winner = best_key.split("::", 1)[1]
                state["included"].append(winner)
                state["remaining"] = [t for t in state["remaining"] if t != winner]
                state["history"].append({"round": state["round"] + 1, "event": "added",
                                         "question": winner[:80], "macro_f1": best_f1})
                print("ROUND %d WINNER: %-58s macro_f1 %.4f" % (
                    state["round"], winner[:58], best_f1), flush=True)
                state["round"] += 1
                state["scores"] = {}
            else:
                state["done"] = True
                print("no candidate improves by > %.3f; selection done at macro_f1 %.4f" % (
                    EPSILON, prev), flush=True)

    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if state["done"]:
        print("SELECTED %d questions:" % len(state["included"]))
        for t in state["included"]:
            print("  -", t[:90])
        print("macro_f1 trajectory:", [h["macro_f1"] for h in state["history"]])


if __name__ == "__main__":
    main()
