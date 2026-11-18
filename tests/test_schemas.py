import unittest

from sdt.schemas import (
    DecisionLabel,
    Example,
    Question,
    QuestionType,
    ScoreVector,
    SdtError,
    Tree,
    TreeNode,
)


class TestQuestion(unittest.TestCase):
    def test_stable_id_and_cache_key(self):
        q1 = Question(text="Is the order delivered?", qtype=QuestionType.COMPUTED, resolver="order_status_delivered")
        q2 = Question(text="Is the order delivered?", qtype=QuestionType.COMPUTED, resolver="order_status_delivered")
        self.assertEqual(q1.qid, q2.qid)
        q2b = Question(text="Is the order delivered?", qtype=QuestionType.COMPUTED, resolver="order_status_delivered", version=2)
        self.assertNotEqual(q1.cache_key, q2b.cache_key)

    def test_computed_requires_resolver(self):
        with self.assertRaises(SdtError):
            Question(text="x?", qtype=QuestionType.COMPUTED)

    def test_semantic_rejects_resolver(self):
        with self.assertRaises(SdtError):
            Question(text="x?", qtype=QuestionType.SEMANTIC, resolver="r")

    def test_roundtrip(self):
        q = Question(text="Refund requested?", qtype=QuestionType.SEMANTIC)
        self.assertEqual(Question.from_dict(q.to_dict()), q)


class TestExample(unittest.TestCase):
    def test_rejects_empty_text(self):
        with self.assertRaises(SdtError):
            Example(example_id="e1", text="  ", context={}, gold=DecisionLabel.ESCALATE)

    def test_roundtrip(self):
        ex = Example(example_id="e1", text="hi", context={"a": 1}, gold=DecisionLabel.AUTO_APPROVE)
        self.assertEqual(Example.from_dict(ex.to_dict()), ex)


class TestScoreVector(unittest.TestCase):
    def test_range_validation(self):
        with self.assertRaises(SdtError):
            ScoreVector(example_id="e", scores={"q": 1.5})
        ScoreVector(example_id="e", scores={"q": 0.0, "r": 1.0})


class TestTree(unittest.TestCase):
    def _nodes(self):
        return {
            "root": TreeNode("root", "q1", branches={'yes': "a", 'no': "b"}),
            "a": TreeNode("a", leaf_label=DecisionLabel.AUTO_APPROVE),
            "b": TreeNode("b", leaf_label=DecisionLabel.POLITELY_DENY),
        }

    def test_valid_tree(self):
        t = Tree("t", "root", self._nodes())
        self.assertEqual(sorted(t.leaves()), ["a", "b"])
        self.assertEqual(t.questions(), ["q1"])

    def test_dangling_child(self):
        nodes = self._nodes()
        nodes["root"] = TreeNode("root", "q1", branches={'yes': "ghost", 'no': "b"})
        with self.assertRaises(SdtError):
            Tree("t", "root", nodes)

    def test_leaf_with_children_rejected(self):
        nodes = self._nodes()
        nodes["a"] = TreeNode("a", leaf_label=DecisionLabel.AUTO_APPROVE, branches={"yes": "b"})
        with self.assertRaises(SdtError):
            Tree("t", "root", nodes)

    def test_cycle_rejected(self):
        nodes = {
            "root": TreeNode("root", "q1", branches={'yes': "mid", 'no': "mid"}),
            "mid": TreeNode("mid", "q2", branches={'yes': "root", 'no': "leaf"}),
            "leaf": TreeNode("leaf", leaf_label=DecisionLabel.ESCALATE),
        }
        with self.assertRaises(SdtError):
            Tree("t", "root", nodes)


class TestActionPrefixMap(unittest.TestCase):
    def test_return_prefix_maps_to_auto_approve(self):
        self.assertIs(DecisionLabel.from_action_prefix("return_delivered_order"), DecisionLabel.AUTO_APPROVE)

    def test_unknown_action_is_none(self):
        self.assertIsNone(DecisionLabel.from_action_prefix("get_order_details"))

    def test_modify_user_address_maps(self):
        self.assertIs(DecisionLabel.from_action_prefix("modify_user_address"),
                      DecisionLabel.MODIFY_USER)

    def test_cross_domain_labels_exist(self):
        self.assertIs(DecisionLabel("book_order"), DecisionLabel.BOOK_ORDER)
        self.assertIs(DecisionLabel("device_setting"), DecisionLabel.DEVICE_SETTING)
        self.assertIs(DecisionLabel("payment"), DecisionLabel.PAYMENT)


if __name__ == "__main__":
    unittest.main()
