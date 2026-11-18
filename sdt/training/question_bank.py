"""The question bank: the durable, compounding asset of this project.

Individual trees are lottery draws (iteration 5's lesson); the bank keeps
every question ever picked into a tree, deduplicated by text, with
provenance of which runs deposited it. The combiner selects over the whole
bank, so every induction - good or bad on its own - adds value.

Artifact format (JSON):
    {"questions": {text_hash: {"text", "criteria", "provenance", "runs", "first_seen"}}}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from sdt.jev_client import COMPUTED_SEEDS
from sdt.schemas import Question, QuestionType, Tree, stable_hash


class QuestionBank:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self._entries: Dict[str, dict] = {}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = data.get("questions", {})

    def deposit(self, questions: Iterable[Question], run_id: str) -> int:
        """Add questions (deduplicated by text). Returns number newly added."""
        added = 0
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for q in questions:
            if q.qtype is QuestionType.COMPUTED:
                continue  # computed seeds are structural, not banked
            key = stable_hash(q.text)
            entry = self._entries.get(key)
            if entry is None:
                self._entries[key] = {
                    "text": q.text,
                    "criteria": q.criteria,
                    "provenance": q.provenance,
                    "runs": [run_id],
                    "first_seen": now,
                    "decision_type": q.decision_type.value,
                    "options": q.options,
                    "rubric": q.rubric,
                }
                added += 1
            else:
                if run_id not in entry["runs"]:
                    entry["runs"].append(run_id)
                if q.criteria and not entry.get("criteria"):
                    entry["criteria"] = q.criteria  # criteria enrich, never overwrite
        return added

    def deposit_tree_artifact(self, path: Path, run_id: Optional[str] = None) -> int:
        run_id = run_id or path.stem
        _, questions = Tree.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return self.deposit(questions, run_id)

    def semantic_questions(self) -> List[Question]:
        out = []
        for entry in sorted(self._entries.values(), key=lambda e: e["text"]):
            q = Question(
                text=entry["text"],
                qtype=QuestionType.SEMANTIC,
                provenance="bank:" + ",".join(entry["runs"][:3]),
                criteria=entry.get("criteria"),
            )
            if entry.get("decision_type") and entry["decision_type"] != "noul":
                from sdt.schemas import DecisionType
                q.decision_type = DecisionType(entry["decision_type"])
                q.options = entry.get("options")
                q.rubric = entry.get("rubric")
            out.append(q)
        return out

    def __len__(self) -> int:
        return len(self._entries)

    def summary(self) -> dict:
        with_criteria = sum(1 for e in self._entries.values() if e.get("criteria"))
        runs = sorted({r for e in self._entries.values() for r in e["runs"]})
        return {"n_questions": len(self._entries),
                "with_criteria": with_criteria, "runs": runs}

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"questions": self._entries}, indent=2),
                                 encoding="utf-8")
