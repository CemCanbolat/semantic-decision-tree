import unittest

from sdt.schemas import Question, QuestionType
from sdt.training.question_bank import QuestionBank


class TestBank(unittest.TestCase):
    def test_deposit_dedupes_and_records_runs(self):
        bank = QuestionBank()
        q = Question(text="Does the user request a refund?", qtype=QuestionType.SEMANTIC)
        self.assertEqual(bank.deposit([q], "run_a"), 1)
        self.assertEqual(bank.deposit([q], "run_b"), 0)  # dedupe
        qs = bank.semantic_questions()
        self.assertEqual(len(qs), 1)
        entry_runs = qs[0].provenance  # bank:a,b in provenance
        self.assertIn("run_a", str(bank.summary()["runs"]))

    def test_criteria_enrich_never_overwrite(self):
        bank = QuestionBank()
        plain = Question(text="Mention an address change?", qtype=QuestionType.SEMANTIC)
        crit = Question(text="Mention an address change?", qtype=QuestionType.SEMANTIC,
                        criteria={"true_description": "yes", "false_description": "no"})
        bank.deposit([plain], "run_a")
        bank.deposit([crit], "run_b")
        qs = bank.semantic_questions()
        self.assertIsNotNone(qs[0].criteria)

    def test_computed_never_banked(self):
        bank = QuestionBank()
        qc = Question(text="Is the referenced order status delivered?",
                      qtype=QuestionType.COMPUTED, resolver="order_status_delivered")
        self.assertEqual(bank.deposit([qc], "run_a"), 0)
        self.assertEqual(len(bank), 0)

    def test_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path
        bank = QuestionBank()
        q = Question(text="Does the text mention a return?", qtype=QuestionType.SEMANTIC)
        bank.deposit([q], "run_a")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.json"
            bank.path = path
            bank.save()
            loaded = QuestionBank(str(path))
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded.semantic_questions()[0].text, q.text)


if __name__ == "__main__":
    unittest.main()
