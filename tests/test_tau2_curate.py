import json
import tempfile
import unittest
from pathlib import Path

from sdt.data.tau2_curate import classify_task, curate, render_text
from sdt.schemas import DecisionLabel

FIXTURE = Path(__file__).parent / "fixtures" / "tau2"


def _task(action_names, reason="I want to return my order", known="I am A B in 19122"):
    return {
        "id": "t",
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": ".",
                "domain": "retail",
                "reason_for_call": reason,
                "known_info": known,
                "unknown_info": None,
            },
        },
        "evaluation_criteria": {
            "actions": [{"action_id": "t_%d" % i, "name": n, "arguments": {}} for i, n in enumerate(action_names)],
            "reward_basis": ["DB"],
        },
    }


class TestClassify(unittest.TestCase):
    def test_return_write_is_auto_approve(self):
        cls = classify_task(_task(["find_user_id_by_name_zip", "return_delivered_order"]))
        self.assertIs(cls.label, DecisionLabel.AUTO_APPROVE)
        self.assertFalse(cls.ambiguous)

    def test_exchange_write(self):
        cls = classify_task(_task(["exchange_delivered_order_items"]))
        self.assertIs(cls.label, DecisionLabel.EXCHANGE)

    def test_transfer_is_escalate(self):
        cls = classify_task(_task(["transfer_to_human_agents"]))
        self.assertIs(cls.label, DecisionLabel.ESCALATE)

    def test_cancel_write(self):
        cls = classify_task(_task(["cancel_pending_order"]))
        self.assertIs(cls.label, DecisionLabel.CANCEL_ORDER)

    def test_no_write_refund_intent_is_ambiguous_deny(self):
        cls = classify_task(_task(["get_order_details"], reason="I want a refund for the broken mug"))
        self.assertIs(cls.label, DecisionLabel.POLITELY_DENY)
        self.assertTrue(cls.ambiguous)

    def test_no_write_info_only(self):
        cls = classify_task(_task(["get_product_details"], reason="How many tshirt options are there?"))
        self.assertIs(cls.label, DecisionLabel.OUT_OF_SCOPE)

    def test_multiple_mutating_categories_flagged(self):
        cls = classify_task(_task(["cancel_pending_order", "exchange_delivered_order_items"]))
        self.assertTrue(cls.ambiguous)
        self.assertEqual(len(cls.mutating_actions), 2)

    def test_modify_user_write_is_a_write(self):
        cls = classify_task(_task(["modify_user_address"]))
        self.assertIs(cls.label, DecisionLabel.MODIFY_USER)


class TestCurateFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.card = curate(FIXTURE, Path(self.tmp))
        from sdt.data.loader import load_examples

        self.examples = {e.example_id: e for e in load_examples(str(Path(self.tmp) / "examples.jsonl"))}

    def test_both_tasks_labeled_exchange(self):
        for tid in ("0", "1"):
            ex = self.examples["tau2_retail_" + tid]
            self.assertIs(ex.gold, DecisionLabel.EXCHANGE)
            self.assertEqual(ex.provenance["matched_action"], "exchange_delivered_order_items")

    def test_user_and_order_joined(self):
        ex = self.examples["tau2_retail_0"]
        self.assertEqual(ex.context["user"]["user_id"], "yusuf_rossi_9620")
        self.assertEqual(ex.context["focus_order_id"], "#W2378156")
        self.assertEqual(ex.context["orders"]["#W2378156"]["status"], "delivered")

    def test_splits_respected(self):
        self.assertEqual(self.examples["tau2_retail_0"].split, "train")
        self.assertEqual(self.examples["tau2_retail_1"].split, "test")

    def test_render_contains_reason_and_known(self):
        text = render_text(json.loads((FIXTURE / "tasks.json").read_text())[0])
        self.assertIn("Reason for contact:", text)
        self.assertIn("exchange the mechanical keyboard", text)
        self.assertIn("Known:", text)
        self.assertIn("Not remembering:", text)

    def test_data_card_and_audit_written(self):
        card = json.loads((Path(self.tmp) / "data_card.json").read_text())
        self.assertEqual(card["n_examples"], 2)
        self.assertEqual(card["label_counts"], {"exchange": 2})
        audit = (Path(self.tmp) / "audit.csv").read_text()
        self.assertIn("task_id", audit)
        self.assertIn("exchange_delivered_order_items", audit)

    def test_family_from_matched_action(self):
        self.assertEqual(self.examples["tau2_retail_0"].family, "tau2:exchange_delivered_order_items")


if __name__ == "__main__":
    unittest.main()
