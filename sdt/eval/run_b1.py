"""B1: the single-prompt frontier classifier - the fair incumbent.

Same inputs as the pipeline (ticket text + context JSON), the full label
taxonomy, temperature 0, and a self-rated confidence so the
coverage-at-precision comparison is possible. Responses are cached per
example so interrupted runs resume. One call per ticket.

    python3.11 -m sdt.eval.run_b1 \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --out eval/reports/b1_retail.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

from sdt.data.loader import load_examples
from sdt.eval import metrics
from sdt.induction.run_induction import GeminiFrontierClient
from sdt.schemas import DecisionLabel, DecisionRecord, GateOutcome

SYSTEM = (
    "You are an expert support adjudicator. Classify the ticket into exactly "
    "one category based on what the customer is asking for and what the agent "
    "should do. Then rate your confidence that your label is correct."
)


def select_shots(examples, n, exclude_id):
    """Deterministic stratified shots: round-robin one per class from the
    train split, never the ticket being classified. Text-only excerpts -
    exactly the view the pipeline's induction prompts had."""
    by_class = {}
    for e in sorted(examples, key=lambda x: x.example_id):
        if e.split == "train" and e.example_id != exclude_id:
            by_class.setdefault(e.gold.value, []).append(e)
    order = sorted(by_class)
    shots = []
    round_idx = 0
    while len(shots) < n:
        added = False
        for c in order:
            pool = by_class[c]
            if round_idx < len(pool) and len(shots) < n:
                shots.append(pool[round_idx])
                added = True
        if not added:
            break
        round_idx += 1
    return shots


def build_prompt(ex, labels, shots=None) -> str:
    context = json.dumps(ex.context, ensure_ascii=False)[:3000]
    parts = ["Categories: " + ", ".join(labels)]
    if shots:
        parts.append("")
        parts.append("Reference examples (label: ticket excerpt):")
        for s in shots:
            excerpt = s.text.replace(chr(10), " ")[:250]
            parts.append("- %s: %s" % (s.gold.value, excerpt))
    parts += [
        "",
        "Ticket:",
        ex.text,
        "",
        "Context JSON:",
        context,
        "",
        'Respond with ONLY a JSON object: {"label": "<one category>", "confidence": <integer 0-100>}',
    ]
    return chr(10).join(parts)


def parse_answer(raw: str, valid) -> Optional[Dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    label = None
    # JSON answers: use the LAST "label" occurrence (the final answer, not a
    # label mentioned mid-reasoning). If braces exist, we are in JSON mode and
    # the prose fallback is disabled: truncated JSON is a failure, never a
    # scavenge.
    q = chr(34)
    pairs = re.findall(q + "label" + q + r"[ ]*:[ ]*" + q + "([^" + q + "]+)" + q, text)
    if pairs:
        cand = pairs[-1].strip().lower().replace(" ", "_").replace("-", "_")
        if cand in valid:
            label = cand
    elif "{" not in text:
        for v in valid:
            if re.search(v.replace("_", "[ _-]"), text.lower()):
                label = v
                break
    conf = 50
    confs = re.findall(q + "confidence" + q + r"[ ]*:[ ]*([0-9]+)", text)
    if confs:
        conf = max(0, min(100, int(confs[-1])))
    if label is None:
        return None
    return {"label": label, "confidence": conf}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None,
                    help="response cache (default: alongside the report)")
    ap.add_argument("--few-shot", type=int, default=0,
                    help="number of stratified labeled examples in the prompt (train split only)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel frontier calls (I/O-bound; 8-16 is safe)")
    args = ap.parse_args()

    examples = load_examples(str(args.corpus))
    labels = sorted({e.gold.value for e in examples})
    cache_path = args.cache or args.out.with_name(args.out.stem + "_cache.json")
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    client = GeminiFrontierClient()
    if not client.api_key:
        raise SystemExit("GEMINI_API_KEY not set")

    started = time.time()
    calls = 0
    missing = [ex for ex in examples if ex.example_id not in cache]

    shots = select_shots(examples, args.few_shot, exclude_id=None) if args.few_shot else None

    def one(ex):
        prompt_shots = [s for s in (shots or []) if s.example_id != ex.example_id]
        raw = client.complete(SYSTEM, build_prompt(ex, labels, prompt_shots),
                              max_tokens=1024, temperature=0.0)
        ans = parse_answer(raw, set(labels))
        if ans is None:
            raw2 = client.complete(
                SYSTEM, build_prompt(ex, labels, prompt_shots) + chr(10) + "Answer with the JSON object only.",
                max_tokens=512, temperature=0.0)
            ans = parse_answer(raw2, set(labels)) or {"label": "__none__", "confidence": 0}
        return ex.example_id, ans, raw[:400]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, ex) for ex in missing]
        for fut in as_completed(futures):
            eid, ans, raw_head = fut.result()
            ans['raw'] = raw_head
            cache[eid] = ans
            calls += 1
            done += 1
            if done % 50 == 0 or done == len(missing):
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                print("  b1 %d/%d done (%.0fs, %d workers)" % (
                    len(cache), len(examples), time.time() - started, args.workers), flush=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    records = []
    for ex in examples:
        ans = cache[ex.example_id]
        pred = DecisionLabel(ans["label"]) if ans["label"] in labels else None
        conf = ans["confidence"] / 100.0
        records.append(DecisionRecord(
            example_id=ex.example_id, tree_id="b1", scores={},
            class_posterior={pred.value: conf} if pred else {},
            margin=max(0.0, 2 * conf - 1),
            bottleneck_nodes=[],
            outcome=GateOutcome.AUTO_EXECUTED if pred else GateOutcome.ESCALATED_MARGIN,
            predicted=pred,
        ))
    acc = metrics.accuracy(records, examples)
    f1 = metrics.macro_f1(records, examples)
    cov = metrics.coverage_at_precision(records, examples, 0.97)
    cm = metrics.confusion_matrix(records, examples)
    per_class = {}
    for g, row in cm.items():
        total = sum(row.values())
        per_class[g] = "%d/%d" % (row.get(g, 0), total)

    report = {"baseline": "B1 single-prompt frontier classifier",
              "model": client.model, "n": len(examples), "api_calls": calls,
              "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
              "coverage_at_0.97": cov,
              "per_class_recall": per_class,
              "note": "confidence is self-rated (0-100), not calibrated probabilities",
              "wall_seconds": round(time.time() - started)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("B1: acc %.4f | macro_f1 %.4f | cov@0.97 %s | %d calls, %.0fs" % (
        acc, f1, json.dumps(cov), calls, time.time() - started))
    print("per-class:", json.dumps(per_class))
    print("report ->", args.out)


if __name__ == "__main__":
    main()
