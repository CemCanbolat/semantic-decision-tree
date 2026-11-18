"""CLI: generate criteria for a tree artifact's questions (iteration 4).

    python3.11 -m sdt.induction.add_criteria \
        --tree artifacts/trees/induced_v3.json \
        --out artifacts/trees/induced_v3_criteria.json \
        --frontier gemini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdt.induction.criteria import add_criteria
from sdt.llm_client import MockFrontierClient
from sdt.schemas import Tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frontier", choices=["gemini", "mock"], default="gemini")
    args = ap.parse_args()

    tree, questions = Tree.from_dict(json.loads(args.tree.read_text()))
    semantic = [q for q in questions if q.text]

    if args.frontier == "gemini":
        from sdt.induction.run_induction import GeminiFrontierClient
        frontier = GeminiFrontierClient()
        if not frontier.api_key:
            raise SystemExit("GEMINI_API_KEY not set")
    else:
        scripted = []
        for q in semantic:
            scripted.append("TRUE: the condition holds for this ticket." + chr(10)
                            + "FALSE: the condition does not hold for this ticket.")
        frontier = MockFrontierClient(scripted=scripted)

    questions, log = add_criteria(frontier, questions)
    n_ok = sum(1 for r in log if r["status"] == "ok")
    print("criteria generated for %d/%d questions" % (n_ok, len(semantic)))
    for row in log:
        if row["status"] != "ok":
            print("  ", row["status"], ":", row["question"][:60])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tree.to_dict({q.qid: q for q in questions}), indent=2),
                        encoding="utf-8")
    log_path = args.out.with_suffix(".criteria_log.json")
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("tree ->", args.out)
    print("criteria log ->", log_path)


if __name__ == "__main__":
    main()
