"""JSONL/JSON IO for dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from sdt.schemas import Example


def save_examples(examples: List[Example], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False))
            fh.write(chr(10))


def load_examples(path: str) -> List[Example]:
    out: List[Example] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Example.from_dict(json.loads(line)))
    return out


def save_json(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))
