"""Score cache: (question cache_key, example_id) -> p.

The economic core of induction: rewording one question invalidates exactly
its column; everything else is reused. JSON-backed, stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from sdt.schemas import Question, ScoreVector


class ScoreCache:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        # {question_cache_key: {example_id: p}}
        self._store: Dict[str, Dict[str, float]] = {}
        if self.path and self.path.exists():
            self._store = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, question: Question, example_id: str) -> Optional[float]:
        return self._store.get(question.cache_key, {}).get(example_id)

    def put(self, question: Question, example_id: str, p: float) -> None:
        self._store.setdefault(question.cache_key, {})[example_id] = float(p)

    def missing(self, question: Question, example_ids: List[str]) -> List[str]:
        col = self._store.get(question.cache_key, {})
        return [eid for eid in example_ids if eid not in col]

    def record(self, question: Question, sv: ScoreVector) -> None:
        if sv.example_id:
            self.put(question, sv.example_id, sv.scores[question.qid])

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._store), encoding="utf-8")

    @property
    def n_entries(self) -> int:
        return sum(len(col) for col in self._store.values())

    # --- multi-type (choice/score) column storage ---

    def put_columns(self, question, example_id: str, branch_probs: Dict[str, float]) -> None:
        self._store.setdefault(question.cache_key + '|cols', {})[example_id] = dict(branch_probs)

    def get_columns(self, question, example_id: str):
        return self._store.get(question.cache_key + '|cols', {}).get(example_id)
