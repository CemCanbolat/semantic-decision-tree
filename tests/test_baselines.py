import unittest

from sdt.data.synth_tickets import generate
from sdt.eval.baselines import build_b1_prompt, build_hand_tree, run_b1, run_b2_hard, run_b2_soft
from sdt.eval.heuristic_jev import HeuristicJevClient
from sdt.llm_client import MockFrontierClient
from sdt.schemas import DecisionLabel


class TestHandTree(unittest.TestCase):
    def setUp(self):
        self.tree, self.questions = build_hand_tree()

    def test_shape_within_budget(self):
        self.assertLessEqual(len(self.questions), 15)
        self.assertEqual(len(self.tree.nodes), 21)

    def test_all_tree_questions_scoreable(self):
        client = HeuristicJevClient()
        ex = generate(n_per_family=1)[0]
        sv = client.score_batch(ex.text, ex.context, self.questions)
        self.assertEqual(set(sv.scores.keys()), {q.qid for q in self.questions})


class TestB2OnSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = generate(n_per_family=4, seed=7)
        cls.client = HeuristicJevClient()

    def test_hard_routing_correct_except_photo_proof(self):
        recs = run_b2_hard(self.client, self.examples)
        wrong = [r.example_id for r, e in zip(recs, self.examples)
                 if r.predicted is not e.gold and e.gold is not DecisionLabel.REQUEST_PHOTO_PROOF]
        self.assertEqual(wrong, [])

    def test_photo_proof_family_lands_in_auto_approve(self):
        recs = run_b2_hard(self.client, self.examples)
        for r, e in zip(recs, self.examples):
            if e.gold is DecisionLabel.REQUEST_PHOTO_PROOF:
                self.assertIs(r.predicted, DecisionLabel.AUTO_APPROVE)

    def test_soft_records_carry_gate_outcomes(self):
        recs = run_b2_soft(self.client, self.examples)
        self.assertTrue(all(r.outcome is not None for r in recs))
        self.assertTrue(all(isinstance(r.class_posterior, dict) and r.class_posterior for r in recs))

    def test_soft_and_hard_agree_on_clear_synthetic(self):
        soft = {r.example_id: r.predicted for r in run_b2_soft(self.client, self.examples)}
        hard = {r.example_id: r.predicted for r in run_b2_hard(self.client, self.examples)}
        for eid in soft:
            if "photo_proof" in eid:
                continue
            self.assertEqual(soft[eid], hard[eid])


class TestB1(unittest.TestCase):
    def test_b1_with_mock_frontier(self):
        examples = generate(n_per_family=1, seed=3)
        scripted = [ex.gold.value for ex in examples]
        client = MockFrontierClient(scripted=scripted)
        recs = run_b1(client, examples)
        self.assertEqual([r.predicted for r in recs], [ex.gold for ex in examples])
        self.assertEqual(len(client.calls), len(examples))

    def test_b1_invalid_label_becomes_escalation(self):
        ex = generate(n_per_family=1, seed=3)[0]
        client = MockFrontierClient(scripted=["banana"])
        rec = run_b1(client, [ex])[0]
        self.assertIsNone(rec.predicted)
        self.assertEqual(rec.outcome.value, "escalated_margin")

    def test_prompt_contains_classes_and_context(self):
        ex = generate(n_per_family=1)[0]
        system, user = build_b1_prompt(ex)
        self.assertIn("auto_approve", system)
        self.assertIn("Ticket:", user)
        self.assertIn("orders", user)


if __name__ == "__main__":
    unittest.main()
