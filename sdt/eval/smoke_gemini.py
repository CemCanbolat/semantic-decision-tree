"""Smoke-test the live Gemini Decisions API path.

Offline (no network needed): prints the exact request the client would send,
as JSON and as a curl command.

    python3.11 -m sdt.eval.smoke_gemini

With network + GEMINI_API_KEY: sends one real batch (all hand-tree semantic
questions against the first synthetic ticket) and prints the p-vector.

    export GEMINI_API_KEY=...
    python3.11 -m sdt.eval.smoke_gemini --send
"""

from __future__ import annotations

import argparse
import json
import sys

from sdt.data.synth_tickets import generate
from sdt.eval.baselines import build_hand_tree
from sdt.gemini_decision_client import GeminiDecisionClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--send", action="store_true", help="actually call the API (needs network + key)")
    args = ap.parse_args()

    tree, questions = build_hand_tree()
    example = generate(n_per_family=1, seed=7)[0]
    client = GeminiDecisionClient()
    request = client.build_request(example.text, example.context, questions)
    n_semantic = len(request["questions"])

    if not args.send:
        print("Request the client would send (%d semantic questions, computed resolved locally):" % n_semantic)
        print(json.dumps(request, indent=2))
        print()
        print("Equivalent curl:")
        print("curl -X POST " + client.endpoint + chr(32) +
              "-H 'Content-Type: application/json' -H 'x-goog-api-key: KEY' -d @request.json")
        return

    sv = client.score_batch(example.text, example.context, questions)
    by_text = {q.qid: q.text for q in questions}
    print("latency_ms", sv.latency_ms)
    for qid, p in sorted(sv.scores.items()):
        print("%6.3f  %s" % (p, by_text.get(qid, qid)))
    sys.exit(0 if sv.scores else 1)


if __name__ == "__main__":
    main()
