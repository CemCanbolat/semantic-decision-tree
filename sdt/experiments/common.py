"""Shared helpers for the multi-type experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.induction.gain import class_weights, weighted_entropy
from sdt.schemas import DecisionLabel, Example, Question, ScoreVector, TreeNode, Tree


def strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def parse_json_object(raw: str):
    """Tolerant JSON extraction: fences, prose padding, trailing commas.

    Raises ValueError only if no complete JSON object/array is present."""
    text = strip_fences(raw).strip()
    # direct attempt
    try:
        return json.loads(text)
    except Exception:
        pass
    # bracket matching from the first opener
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == chr(92):
                esc = True
                continue
            if ch == chr(34):
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        try:
                            import re as _re
                            fixed = _re.sub(',[ ]*(}|' + chr(93) + ')', lambda m: m.group(1), candidate)
                            return json.loads(fixed)
                        except Exception:
                            break
        continue
    raise ValueError("no parseable JSON in frontier output: %.120r" % raw)


def complete_json(frontier, system: str, user: str, attempts: int = 3, fallback=None):
    """Frontier call with tolerant parsing, retries, and a final fallback.

    The unattended experiment suite must degrade, never die: a malformed
    response triggers a stricter retry; exhausted retries fall back."""
    last_err = None
    for i in range(attempts):
        suffix = "" if i == 0 else (chr(10) + "Return ONLY valid JSON. No prose, "
                                    "no markdown fences, escape all quotes.")
        try:
            return parse_json_object(frontier.complete(system, user + suffix))
        except Exception as e:
            last_err = e
    if fallback is not None:
        print("[warn] JSON parsing failed %d times (%s); using fallback" % (attempts, last_err))
        return fallback
    raise SystemExit("frontier JSON failed after %d attempts: %s" % (attempts, last_err))


def score_with_cache(client, examples, questions, cache, save_every=20):
    """Score every (question, example); choice/score land as columns, noul as floats."""
    done = 0
    for i, ex in enumerate(examples, 1):
        todo = []
        for q in questions:
            if q.decision_type.value == "noul":
                if cache.get(q, ex.example_id) is None:
                    todo.append(q)
            elif cache.get_columns(q, ex.example_id) is None:
                todo.append(q)
        if not todo:
            continue
        sv = client.score_batch(ex.text, ex.context, todo)
        for q in todo:
            if q.decision_type.value == "noul":
                cache.put(q, ex.example_id, sv.scores[q.qid])
            else:
                cols = {c.split("::", 1)[1]: p for c, p in sv.scores.items()
                        if c.startswith(q.qid + "::")}
                cache.put_columns(q, ex.example_id, cols)
        done += 1
        if i % save_every == 0:
            cache.save()
    cache.save()
    return done


def branch_assignment(q: Question, cache: ScoreCacheAlias, ex: Example) -> str:
    """Argmax branch for one example (from cached columns / noul probability)."""
    if q.decision_type.value == "noul":
        p = cache.get(q, ex.example_id)
        return "yes" if (p or 0.5) >= 0.5 else "no"
    cols = cache.get_columns(q, ex.example_id) or {}
    if not cols:
        return "missing"
    return max(cols.items(), key=lambda kv: kv[1])[0]


from sdt.training.cache import ScoreCache as ScoreCacheAlias  # noqa: E402


def majority(examples: List[Example]) -> DecisionLabel:
    counts: Dict[DecisionLabel, int] = {}
    for e in examples:
        counts[e.gold] = counts.get(e.gold, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0].value))[0]


def weighted_majority(examples: List[Example], alpha: float = 1.0) -> DecisionLabel:
    wmap = class_weights(examples)
    wmap = {k: v ** alpha for k, v in wmap.items()}
    scores: Dict[DecisionLabel, float] = {}
    for e in examples:
        scores[e.gold] = scores.get(e.gold, 0.0) + wmap.get(e.gold, 1.0)
    return max(scores.items(), key=lambda kv: (kv[1], kv[0].value))[0]


def build_multiway_tree(root_q: Question, examples: List[Example], cache,
                        tree_id: str, second_q: Question = None) -> Tuple[Tree, Dict]:
    """Root = multi-way question; per-branch leaves = weighted majority.
    If second_q given, each non-empty branch gets a second-level node."""
    from sdt.decision_types import branch_names
    branches = branch_names(root_q)
    groups: Dict[str, List[Example]] = {b: [] for b in branches}
    fallback_branch = max(branches, key=lambda b: len(branches))
    for ex in examples:
        b = branch_assignment(root_q, cache, ex)
        groups.setdefault(b, [])
        if b not in groups or not groups.get(b) and b not in branches:
            b = fallback_branch
        # unknown branch names (missing columns) fold into the fallback branch
        if b not in branches:
            groups[fallback_branch].extend(groups.pop(b))
        else:
            groups[b].append(ex)
    # repair: anything assigned to a non-branch key moves to the fallback branch
    for extra in [k for k in groups if k not in branches]:
        groups[fallback_branch].extend(groups.pop(extra))
    global_majority = majority(examples)
    nodes: Dict[str, TreeNode] = {}
    root_children: Dict[str, str] = {}
    for b in branches:
        members = groups[b]
        if second_q is not None and len(members) >= 2 * len(branch_names(second_q)):
            sub: Dict[str, str] = {}
            for lvl in branch_names(second_q):
                leaf_id = b.replace(" ", "_") + "_" + lvl
                lvl_members = [e for e in members
                               if branch_assignment(second_q, cache, e) == lvl]
                label = weighted_majority(lvl_members) if lvl_members else (weighted_majority(members) if members else global_majority)
                nodes[leaf_id] = TreeNode(leaf_id, leaf_label=label)
                sub[lvl] = leaf_id
            mid_id = b.replace(" ", "_") + "_mid"
            nodes[mid_id] = TreeNode(mid_id, question_qid=second_q.qid, branches=sub)
            root_children[b] = mid_id
        else:
            leaf_id = b.replace(" ", "_") + "_leaf"
            label = weighted_majority(members) if members else global_majority
            nodes[leaf_id] = TreeNode(leaf_id, leaf_label=label)
            root_children[b] = leaf_id
    nodes["root"] = TreeNode("root", question_qid=root_q.qid, branches=root_children)
    stats = {"branch_sizes": {b: len(groups[b]) for b in branches}}
    return Tree(tree_id, "root", nodes), stats
