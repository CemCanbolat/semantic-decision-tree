import math
import unittest

from sdt.routing.gates import decide
from sdt.routing.soft_tree import route
from sdt.schemas import DecisionLabel, ScoreVector, SdtError, Tree, TreeNode


def _tree():
    nodes = {
        "root": TreeNode("root", "q_root", branches={'yes': "left", 'no': "right"}),
        "left": TreeNode("left", "q_left", branches={'yes': "l_yes", 'no': "l_no"}),
        "l_yes": TreeNode("l_yes", leaf_label=DecisionLabel.AUTO_APPROVE),
        "l_no": TreeNode("l_no", leaf_label=DecisionLabel.POLITELY_DENY),
        "right": TreeNode("right", leaf_label=DecisionLabel.POLITELY_DENY),
    }
    return Tree("t", "root", nodes)


class TestSoftRouting(unittest.TestCase):
    def test_hand_computed_probabilities(self):
        r = route(_tree(), ScoreVector("e", {"q_root": 0.8, "q_left": 0.9}))
        probs = r.leaf_probs
        self.assertAlmostEqual(probs["l_yes"], 0.72, places=9)
        self.assertAlmostEqual(probs["l_no"], 0.08, places=9)
        self.assertAlmostEqual(probs["right"], 0.20, places=9)
        self.assertAlmostEqual(r.class_posterior[DecisionLabel.AUTO_APPROVE], 0.72, places=9)
        self.assertAlmostEqual(r.class_posterior[DecisionLabel.POLITELY_DENY], 0.28, places=9)

    def test_log_space_matches_direct_product_at_depth_3(self):
        nodes = {
            "n0": TreeNode("n0", "q0", branches={'yes': "n1", 'no': "d0"}),
            "n1": TreeNode("n1", "q1", branches={'yes': "n2", 'no': "d1"}),
            "n2": TreeNode("n2", "q2", branches={'yes': "d2", 'no': "d3"}),
            "d0": TreeNode("d0", leaf_label=DecisionLabel.AUTO_APPROVE),
            "d1": TreeNode("d1", leaf_label=DecisionLabel.EXCHANGE),
            "d2": TreeNode("d2", leaf_label=DecisionLabel.CANCEL_ORDER),
            "d3": TreeNode("d3", leaf_label=DecisionLabel.POLITELY_DENY),
        }
        t = Tree("deep", "n0", nodes)
        scores = ScoreVector("e", {"q0": 0.9, "q1": 0.7, "q2": 0.4})
        r = route(t, scores)
        self.assertAlmostEqual(r.leaf_probs["d2"], 0.9 * 0.7 * 0.4, places=9)
        self.assertAlmostEqual(r.leaf_probs["d3"], 0.9 * 0.7 * 0.6, places=9)
        self.assertAlmostEqual(r.leaf_probs["d1"], 0.9 * 0.3, places=9)
        self.assertAlmostEqual(r.leaf_probs["d0"], 0.1, places=9)
        total = sum(r.leaf_probs.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_uniform_scores_spread_mass(self):
        r = route(_tree(), ScoreVector("e", {"q_root": 0.5, "q_left": 0.5}))
        for p in r.leaf_probs.values():
            self.assertGreater(p, 0.0)

    def test_missing_score_raises(self):
        with self.assertRaises(SdtError):
            route(_tree(), ScoreVector("e", {"q_root": 0.8}))


class TestGates(unittest.TestCase):
    def test_confident_run_auto_executes(self):
        r = route(_tree(), ScoreVector("e", {"q_root": 0.99, "q_left": 0.98}))
        g = decide(r)
        self.assertEqual(g.outcome.value, "auto_executed")
        self.assertIs(g.top, DecisionLabel.AUTO_APPROVE)
        self.assertGreater(g.margin, 0.60)

    def test_low_margin_escalates(self):
        r = route(_tree(), ScoreVector("e", {"q_root": 0.5, "q_left": 0.99}))
        g = decide(r)
        self.assertEqual(g.outcome.value, "escalated_both")

    def test_bottleneck_detection(self):
        r = route(_tree(), ScoreVector("e", {"q_root": 0.9, "q_left": 0.55}))
        g = decide(r)
        self.assertIn("left", g.bottleneck_nodes)

    def test_no_bottleneck_when_certain(self):
        r = route(_tree(), ScoreVector("e", {"q_root": 0.95, "q_left": 0.95}))
        g = decide(r)
        self.assertEqual(g.bottleneck_nodes, [])


if __name__ == "__main__":
    unittest.main()
