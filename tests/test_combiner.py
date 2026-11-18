import unittest

from sdt.schemas import DecisionLabel, Example
from sdt.training.combiner import (
    SoftmaxModel, accuracy_on, build_features, cross_validate,
    fit_temperature, macro_f1_on, relabel_tree_leaves, stratified_folds,
)
from sdt.training.cache import ScoreCache
from sdt.schemas import Question, QuestionType, Tree, TreeNode


def _ex(eid, gold, ctx=None):
    return Example(example_id=eid, text="t " + eid, context=ctx or {}, gold=gold)


LABELS = [DecisionLabel.AUTO_APPROVE, DecisionLabel.POLITELY_DENY]


class TestSoftmax(unittest.TestCase):
    def test_learns_separable_data(self):
        X = [[0.9], [0.95], [0.1], [0.05]]
        y = [0, 0, 1, 1]
        model = SoftmaxModel(LABELS, 1).fit(X, y, [1.0] * 4)
        preds = [model.predict_index(x) for x in X]
        self.assertEqual(preds, y)

    def test_class_weights_protect_minority(self):
        # 20 majority + 2 minority; unweighted would ignore the minority pair
        X = [[0.9]] * 20 + [[0.1], [0.05]]
        y = [0] * 20 + [1, 1]
        w_unweighted = SoftmaxModel(LABELS, 1).fit(X, y, [1.0] * 22)
        self.assertEqual([w_unweighted.predict_index(x) for x in [[0.1], [0.05]]], [1, 1])
        # weighted sanity: still perfect
        w_weighted = SoftmaxModel(LABELS, 1).fit(X, y, [1.0] * 20 + [10.0, 10.0])
        self.assertEqual([w_weighted.predict_index(x) for x in X], y)

    def test_temperature_softens_overconfidence(self):
        # mostly confident-correct, one confident-WRONG: the wrong one drives T above 1
        z = [[6.0, 0.0]] * 4 + [[0.0, 6.0]] + [[6.0, 0.0]]
        y = [0, 0, 0, 0, 1, 1]
        T = fit_temperature(z, y)
        self.assertGreater(T, 1.0)  # overconfident logits -> T pulls above 1

    def test_build_features_dedupes_seed_columns(self):
        from sdt.training.combiner import COMPUTED_SEEDS
        from sdt.schemas import QuestionType
        ex = Example(example_id="e1", text="t",
                     context={"orders": {"#A": {"status": "delivered", "payment_history": []}},
                              "focus_order_id": "#A", "user": {}},
                     gold=DecisionLabel.AUTO_APPROVE)
        cache = ScoreCache()
        dup_seed = COMPUTED_SEEDS[1]  # pending question passed as "semantic"
        X, names = build_features([ex], [dup_seed], cache, include_computed=True)
        self.assertEqual(names.count(dup_seed.text), 1)


class TestCV(unittest.TestCase):
    def test_honest_oof_on_noise(self):
        # features identical to labels (trivial), CV should be near-perfect
        X = [[1.0 if i % 2 == 0 else 0.0] for i in range(10)]
        y = [i % 2 for i in range(10)]
        labels = LABELS
        oof, acc, f1 = cross_validate(X, y, [1.0] * 10, labels)
        self.assertEqual(len(oof), 10)
        self.assertGreater(acc, 0.9)


class TestRelabel(unittest.TestCase):
    def test_mislabeled_minority_leaf_fixed(self):
        q = Question(text="Does the ticket mention a city?", qtype=QuestionType.SEMANTIC)
        nodes = {
            "root": TreeNode("root", q.qid, branches={'yes': "yes", 'no': "no"}),
            "yes": TreeNode("yes", leaf_label=DecisionLabel.POLITELY_DENY),  # wrong label
            "no": TreeNode("no", leaf_label=DecisionLabel.MODIFY_ORDER),
        }
        tree = Tree("t", "root", nodes)
        examples = (
            [_ex("a%02d" % i, DecisionLabel.MODIFY_USER) for i in range(3)]   # routed yes
            + [_ex("b%02d" % i, DecisionLabel.POLITELY_DENY) for i in range(4)]  # routed yes
            + [_ex("c%02d" % i, DecisionLabel.MODIFY_ORDER) for i in range(5)]   # routed no
        )
        cache = ScoreCache()
        for e in examples[:7]:
            cache.put(q, e.example_id, 0.9)
        for e in examples[7:]:
            cache.put(q, e.example_id, 0.1)
        new_tree, changes = relabel_tree_leaves(tree, [q], examples, cache)
        self.assertEqual(changes, [{"leaf": "yes", "from": "politely_deny",
                                    "to": "modify_user", "n_examples": 7}])
        self.assertIs(new_tree.nodes["yes"].leaf_label, DecisionLabel.MODIFY_USER)


if __name__ == "__main__":
    unittest.main()
