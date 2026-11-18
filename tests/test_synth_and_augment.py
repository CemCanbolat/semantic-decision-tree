import unittest

from sdt.data.augment import augment, emotional_variant
from sdt.data.synth_tickets import FAMILIES, generate
from sdt.schemas import DecisionLabel


class TestSynth(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate(n_per_family=2), generate(n_per_family=2))

    def test_all_families_present(self):
        ex = generate(n_per_family=1)
        self.assertEqual({e.family for e in ex}, set(FAMILIES.values()))

    def test_photo_proof_and_deny_covered(self):
        ex = generate(n_per_family=1)
        golds = {e.gold for e in ex}
        self.assertIn(DecisionLabel.REQUEST_PHOTO_PROOF, golds)
        self.assertIn(DecisionLabel.POLITELY_DENY, golds)

    def test_deny_family_context_is_pending(self):
        ex = [e for e in generate(n_per_family=1) if e.gold is DecisionLabel.POLITELY_DENY][0]
        oid = ex.context["focus_order_id"]
        self.assertEqual(ex.context["orders"][oid]["status"], "pending")

    def test_emotional_variant_exists_for_auto_approve(self):
        ex = generate(n_per_family=2)
        variants = {e.variant for e in ex if e.gold is DecisionLabel.AUTO_APPROVE}
        self.assertIn("emotional", variants)


class TestAugment(unittest.TestCase):
    def test_label_and_context_preserved(self):
        base = [e for e in generate(n_per_family=1) if e.family == "return_delivered_clean"][0]
        emo = emotional_variant(base, "Outrageous!")
        self.assertIs(emo.gold, base.gold)
        self.assertEqual(emo.context, base.context)
        self.assertTrue(emo.text.startswith("Outrageous!"))
        self.assertEqual(emo.variant, "emotional")

    def test_augment_is_deterministic_and_grows_set(self):
        base = generate(n_per_family=2)
        a1 = augment(base)
        a2 = augment(base)
        self.assertEqual(a1, a2)
        self.assertEqual(len(a1), len(base) + 4)  # one emotional per family per n_per_family


if __name__ == "__main__":
    unittest.main()
