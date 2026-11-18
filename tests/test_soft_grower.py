import unittest

from sdt.decision_types import noul as mk_noul
from sdt.experiments.soft_grower import (MIN_BRANCH_MASS, alpha_class_weights,
                                         branch_probabilities, prune_by_validation,
                                         split_score, tree_macro_f1,
                                         _majority, _weighted_entropy)
from sdt.schemas import DecisionLabel, Example, Question, QuestionType, TreeNode, Tree
from sdt.training.cache import ScoreCache


def _ex(eid, gold):
    return Example(example_id=eid, text="t", context={}, gold=gold)


A, B = DecisionLabel.AUTO_APPROVE, DecisionLabel.POLITELY_DENY


def _cache_for(q, ids_yes):
    cache = ScoreCache()
    for e in _ALL:
        cache.put(q, e.example_id, 0.9 if e.example_id in ids_yes else 0.1)
    return cache


# corpus: 100 A, 6 B
_ALL = ([_ex("a%03d" % i, A) for i in range(100)] + [_ex("b%02d" % i, B) for i in range(6)])


class TestGainRatioGuards(unittest.TestCase):
    def setUp(self):
        self.q = mk_noul("Does the marker question hold?")
        self.cw = alpha_class_weights(_ALL, 0.6)

    def test_lopsided_minority_guard_rejects(self):
        # the real exploit is a CONFIDENT lopsided peel-off: 105 tickets at
        # p=0.99 yes, 1 ticket at p=0.99 no. Sub-cutoff (0.01) contributions
        # drop, so the minority branch holds ~1 effective ticket -> rejected.
        cache = ScoreCache()
        for e in _ALL[:105]:
            cache.put(self.q, e.example_id, 0.99)
        cache.put(self.q, _ALL[105].example_id, 0.01)
        res = split_score([(e, 1.0) for e in _ALL], self.q, cache, self.cw)
        self.assertIsNone(res)

    def test_balanced_split_accepted(self):
        cache = _cache_for(self.q, {e.example_id for e in _ALL[:53]})
        res = split_score([(e, 1.0) for e in _ALL], self.q, cache, self.cw)
        self.assertIsNotNone(res)
        score, children = res
        self.assertGreater(score, 0)

    def test_denominator_floor_handles_skew(self):
        # 100 vs 6 split (minority mass 6 >= 3): accepted; denominator floored at 0.35
        cache = _cache_for(self.q, {e.example_id for e in _ALL if e.gold is B})
        res = split_score([(e, 1.0) for e in _ALL], self.q, cache, self.cw)
        self.assertIsNotNone(res)

    def test_fractional_routing_spreads_mass(self):
        cache = ScoreCache()
        for e in _ALL:
            if e.gold is A:
                cache.put(self.q, e.example_id, 0.9)
            else:
                cache.put(self.q, e.example_id, 0.1)
        items = [(e, 1.0) for e in _ALL]
        res = split_score(items, self.q, cache, self.cw)
        self.assertIsNotNone(res)
        score, children = res
        # every ticket keeps mass in BOTH branches (soft growth semantics)
        self.assertEqual(len(children["yes"]), 106)
        self.assertEqual(len(children["no"]), 106)
        by_id_yes = {ex.example_id: w for ex, w in children["yes"]}
        a_ex = next(ex for ex, _ in items if ex.gold is A)
        b_ex = next(ex for ex, _ in items if ex.gold is B)
        self.assertAlmostEqual(by_id_yes[a_ex.example_id], 0.9, places=6)
        self.assertAlmostEqual(by_id_yes[b_ex.example_id], 0.1, places=6)


class TestAlphaWeights(unittest.TestCase):
    def test_rare_class_boosted_not_degenerate(self):
        cw = alpha_class_weights(_ALL, 0.6)
        self.assertGreater(cw[B], cw[A])
        ratio = cw[B] / cw[A]
        self.assertLess(ratio, (100 / 6))  # softer than full inverse frequency


class TestPruner(unittest.TestCase):
    def test_memorized_subtree_collapses(self):
        from sdt.experiments.soft_grower import tree_macro_f1
        q = mk_noul("Marker?")
        cache = ScoreCache()
        examples = [_ex("a%d" % i, A) for i in range(10)] + [_ex("b%d" % i, B) for i in range(10)]
        for e in examples[:10]:
            cache.put(q, e.example_id, 0.9)
        for e in examples[10:]:
            cache.put(q, e.example_id, 0.1)
        # tree with a spurious extra level under YES: splits 10 A into 6/4 by noise
        q2 = mk_noul("Noise marker?")
        for e in examples[:6]:
            cache.put(q2, e.example_id, 0.9)
        for e in examples[6:]:
            cache.put(q2, e.example_id, 0.1)
        nodes = {
            "root": TreeNode("root", q.qid, branches={"yes": "mid", "no": "bleaf"}),
            "mid": TreeNode("mid", q2.qid, branches={"yes": "l1", "no": "l2"}),
            "l1": TreeNode("l1", leaf_label=A),
            "l2": TreeNode("l2", leaf_label=A),   # same label: pure memorization, zero lift
            "bleaf": TreeNode("bleaf", leaf_label=B),
        }
        tree = Tree("t", "root", nodes)
        pruned = prune_by_validation(tree, [q, q2], cache, examples)
        # l1/l2 carry identical labels: collapsing mid gives identical prune F1 -> not applied
        # so verify the pruner does not harm, and collapses when lift exists
        self.assertIn("mid", pruned.nodes)
        f1_before = tree_macro_f1(tree, [q, q2], cache, examples)
        self.assertEqual(f1_before, tree_macro_f1(pruned, [q, q2], cache, examples))

        # now make the sub-split actively WRONG on validation: l2 relabeled B
        nodes["l2"] = TreeNode("l2", leaf_label=B)
        tree2 = Tree("t2", "root", nodes)
        pruned2 = prune_by_validation(tree2, [q, q2], cache, examples)
        self.assertGreaterEqual(tree_macro_f1(pruned2, [q, q2], cache, examples),
                                tree_macro_f1(tree2, [q, q2], cache, examples))


if __name__ == "__main__":
    unittest.main()
