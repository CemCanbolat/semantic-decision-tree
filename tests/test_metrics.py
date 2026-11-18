import unittest

from sdt.eval import metrics
from sdt.schemas import DecisionLabel, DecisionRecord, Example


def _ex(eid, gold):
    return Example(example_id=eid, text="t", context={}, gold=gold)


def _rec(eid, pred, margin):
    return DecisionRecord(
        example_id=eid, tree_id="t", scores={}, class_posterior={},
        margin=margin, bottleneck_nodes=[], outcome=None, predicted=pred,
    )


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.examples = [
            _ex("a", DecisionLabel.AUTO_APPROVE),
            _ex("b", DecisionLabel.AUTO_APPROVE),
            _ex("c", DecisionLabel.POLITELY_DENY),
            _ex("d", DecisionLabel.POLITELY_DENY),
        ]

    def test_accuracy(self):
        recs = [_rec("a", DecisionLabel.AUTO_APPROVE, 1.0),
                _rec("b", DecisionLabel.AUTO_APPROVE, 1.0),
                _rec("c", DecisionLabel.AUTO_APPROVE, 0.8),   # wrong
                _rec("d", DecisionLabel.POLITELY_DENY, 0.9)]
        self.assertEqual(metrics.accuracy(recs, self.examples), 0.75)

    def test_macro_f1(self):
        recs = [_rec("a", DecisionLabel.AUTO_APPROVE, 1.0),
                _rec("b", DecisionLabel.AUTO_APPROVE, 1.0),
                _rec("c", DecisionLabel.POLITELY_DENY, 1.0),
                _rec("d", DecisionLabel.POLITELY_DENY, 1.0)]
        self.assertEqual(metrics.macro_f1(recs, self.examples), 1.0)

    def test_none_prediction_counts_wrong(self):
        recs = [_rec("a", None, 0.0)] + [_rec(e, None, 0.0) for e in "bcd"]
        self.assertEqual(metrics.accuracy(recs, self.examples), 0.0)

    def test_coverage_at_precision_prefers_widest_feasible(self):
        recs = [_rec("a", DecisionLabel.AUTO_APPROVE, 0.95),   # correct, high margin
                _rec("b", DecisionLabel.AUTO_APPROVE, 0.95),   # correct, high margin
                _rec("c", DecisionLabel.AUTO_APPROVE, 0.3),    # wrong, low margin
                _rec("d", DecisionLabel.POLITELY_DENY, 0.3)]   # correct, low margin
        res = metrics.coverage_at_precision(recs, self.examples, 0.97)
        self.assertEqual(res["coverage"], 0.5)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["threshold"], 0.95)

    def test_coverage_curve_is_monotone_in_coverage(self):
        recs = [_rec("a", DecisionLabel.AUTO_APPROVE, 0.9),
                _rec("b", DecisionLabel.POLITELY_DENY, 0.5),
                _rec("c", DecisionLabel.POLITELY_DENY, 0.2),
                _rec("d", DecisionLabel.AUTO_APPROVE, 0.8)]
        curve = metrics.coverage_curve(recs, self.examples)
        covs = [row["coverage"] for row in curve]
        self.assertEqual(covs, sorted(covs))

    def test_empty_auto_set_is_not_a_feasible_point(self):
        recs = [_rec("a", DecisionLabel.AUTO_APPROVE, 0.5)]
        res = metrics.coverage_at_precision(recs, self.examples[:1], 0.97)
        self.assertEqual(res["coverage"], 1.0)  # the one correct record is included at tau <= 0.5
        curve = metrics.coverage_curve(recs, self.examples[:1])
        self.assertTrue(all(row["coverage"] > 0 for row in curve))  # vacuous rows excluded

    def test_confusion_counts_crossings(self):
        recs = [_rec("a", DecisionLabel.AUTO_APPROVE, 1.0),
                _rec("c", DecisionLabel.AUTO_APPROVE, 1.0)]
        cm = metrics.confusion_matrix(recs, self.examples[:1] + [self.examples[2]])
        self.assertEqual(cm["auto_approve"]["auto_approve"], 1)
        self.assertEqual(cm["politely_deny"]["auto_approve"], 1)


if __name__ == "__main__":
    unittest.main()
