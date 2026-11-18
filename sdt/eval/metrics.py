"""Evaluation metrics: accuracy, macro-F1, confusion, coverage@precision."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sdt.schemas import DecisionLabel, DecisionRecord, Example


def _correct_flags(records: List[DecisionRecord], examples: List[Example]) -> List[Tuple[float, bool]]:
    gold = {e.example_id: e.gold for e in examples}
    out = []
    for rec in records:
        if rec.predicted is None or rec.example_id not in gold:
            out.append((rec.margin, False))
        else:
            out.append((rec.margin, rec.predicted == gold[rec.example_id]))
    return out


def accuracy(records: List[DecisionRecord], examples: List[Example]) -> float:
    flags = _correct_flags(records, examples)
    if not flags:
        return 0.0
    return sum(1.0 for _, ok in flags if ok) / len(flags)


def confusion_matrix(records: List[DecisionRecord], examples: List[Example]) -> Dict[str, Dict[str, int]]:
    gold = {e.example_id: e.gold for e in examples}
    cm: Dict[str, Dict[str, int]] = {}
    for rec in records:
        g = gold.get(rec.example_id)
        if g is None:
            continue
        p = rec.predicted.value if rec.predicted else "none"
        cm.setdefault(g.value, {})
        cm[g.value][p] = cm[g.value].get(p, 0) + 1
    return cm


def macro_f1(records: List[DecisionRecord], examples: List[Example]) -> float:
    gold = {e.example_id: e.gold for e in examples}
    tp: Dict[str, int] = {}
    fp: Dict[str, int] = {}
    fn: Dict[str, int] = {}
    classes = set()
    for rec in records:
        if rec.example_id not in gold:
            continue
        g = gold[rec.example_id].value
        p = rec.predicted.value if rec.predicted else "none"
        classes.add(g)
        classes.add(p)
        if g == p:
            tp[g] = tp.get(g, 0) + 1
        else:
            fp[p] = fp.get(p, 0) + 1
            fn[g] = fn.get(g, 0) + 1
    f1s = []
    for c in sorted(classes):
        prec = tp.get(c, 0) / (tp.get(c, 0) + fp.get(c, 0)) if (tp.get(c, 0) + fp.get(c, 0)) else 0.0
        rec_ = tp.get(c, 0) / (tp.get(c, 0) + fn.get(c, 0)) if (tp.get(c, 0) + fn.get(c, 0)) else 0.0
        f1s.append(2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def coverage_curve(records: List[DecisionRecord], examples: List[Example],
                   thresholds: Optional[List[float]] = None) -> List[Dict[str, float]]:
    """Sweep margin threshold: auto-execute when margin >= threshold."""
    flags = _correct_flags(records, examples)
    n = len(flags)
    if thresholds is None:
        thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    curve = []
    for tau in thresholds:
        auto = [(m, ok) for m, ok in flags if m >= tau]
        if not auto:
            continue  # vacuous precision on an empty auto-set is not a feasible point
        cov = len(auto) / n if n else 0.0
        prec = sum(1.0 for _, ok in auto if ok) / len(auto)
        curve.append({"threshold": tau, "coverage": round(cov, 4), "precision": round(prec, 4)})
    return curve


def coverage_at_precision(records: List[DecisionRecord], examples: List[Example],
                          target: float = 0.97) -> Optional[Dict[str, float]]:
    curve = coverage_curve(records, examples)
    feasible = [row for row in curve if row["precision"] >= target]
    if not feasible:
        return None
    best = max(feasible, key=lambda row: row["coverage"])
    return {"target_precision": target, "coverage": best["coverage"],
            "precision": best["precision"], "threshold": best["threshold"]}
