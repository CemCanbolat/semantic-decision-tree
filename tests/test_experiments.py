import unittest

from sdt.decision_types import choice, noul, score, branch_names
from sdt.experiments.common import build_multiway_tree, weighted_majority
from sdt.experiments.exp_b_free import parse_free
from sdt.routing.soft_tree import route
from sdt.schemas import DecisionLabel, Example, ScoreVector
from sdt.training.cache import ScoreCache


def _ex(eid, gold, ctx):
    return Example(example_id=eid, text="t", context=ctx, gold=gold)


CTX_D = {"orders": {"#A": {"status": "delivered", "payment_history": []}}, "focus_order_id": "#A", "user": {}}


class TestParseFree(unittest.TestCase):
    def test_all_three_formats(self):
        raw = ("NOUL: Does the user request a refund?" + chr(10)
               + "CHOICE: Which action is requested? OPTIONS: return; exchange; cancel" + chr(10)
               + "SCORE: How clear is the request? LEVELS: vague; partial; explicit" + chr(10)
               + "junk line")
        qs = parse_free(raw, "t")
        kinds = sorted(q.decision_type.value for q in qs)
        self.assertEqual(kinds, ["choice", "noul", "score"])
        ch = [q for q in qs if q.decision_type.value == "choice"][0]
        self.assertEqual(len(branch_names(ch)), 3)
        sc = [q for q in qs if q.decision_type.value == "score"][0]
        self.assertEqual(len(branch_names(sc)), 3)

    def test_leaky_noul_rejected(self):
        qs = parse_free("NOUL: Is this an out of scope request?", "t")
        self.assertEqual(qs, [])


class TestMultiWay(unittest.TestCase):
    def test_choice_tree_build_and_route(self):
        q = choice("Which category?", {"refund": "money back", "other": "rest"})
        examples = ([_ex("a%d" % i, DecisionLabel.AUTO_APPROVE, CTX_D) for i in range(4)]
                    + [_ex("b%d" % i, DecisionLabel.POLITELY_DENY, CTX_D) for i in range(4)])
        cache = ScoreCache()
        for e in examples[:4]:
            cache.put_columns(q, e.example_id, {"refund": 0.9, "other": 0.1})
        for e in examples[4:]:
            cache.put_columns(q, e.example_id, {"refund": 0.2, "other": 0.8})
        tree, stats = build_multiway_tree(q, examples, cache, "t")
        self.assertEqual(stats["branch_sizes"], {"refund": 4, "other": 4})
        from sdt.jev_client import JevClient

        class S(JevClient):
            def __init__(self):
                self.current = None
            def score_batch(self, text, ctx, qs):
                cols = cache.get_columns(q, self.current)
                return ScoreVector(example_id=self.current,
                                   scores={q.qid + "::" + c: p for c, p in cols.items()})

        s = S()
        s.current = "a0"
        sv = s.score_batch("t", CTX_D, [q])
        r = route(tree, sv, {q.qid: q})
        self.assertGreater(r.class_posterior[DecisionLabel.AUTO_APPROVE], 0.8)

    def test_weighted_majority_basic(self):
        exs = ([_ex("a%d" % i, DecisionLabel.ESCALATE, CTX_D) for i in range(3)]
               + [_ex("b%d" % i, DecisionLabel.MODIFY_ORDER, CTX_D) for i in range(4)])
        self.assertIs(weighted_majority(exs), DecisionLabel.MODIFY_ORDER)


if __name__ == "__main__":
    unittest.main()
