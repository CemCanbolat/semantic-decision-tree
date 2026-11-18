import unittest

from sdt.induction.add_criteria import main as add_criteria_main
from sdt.induction.criteria import add_criteria, build_criteria_prompt, parse_criteria
from sdt.llm_client import MockFrontierClient
from sdt.schemas import Question, QuestionType, Tree


class TestCriteria(unittest.TestCase):
    def test_parse_criteria(self):
        raw = "Some preamble" + chr(10) + "TRUE: The item is broken or faulty." + chr(10) + "FALSE: Preference or sizing complaint."
        crit = parse_criteria(raw)
        self.assertEqual(crit["true_description"], "The item is broken or faulty.")
        self.assertEqual(crit["false_description"], "Preference or sizing complaint.")

    def test_parse_failure_returns_none(self):
        self.assertIsNone(parse_criteria("yes sometimes" + chr(10) + "no other times"))

    def test_add_criteria_mutates_and_logs(self):
        qs = [Question(text="Was the item damaged?", qtype=QuestionType.SEMANTIC),
              Question(text="Is the order delivered?", qtype=QuestionType.COMPUTED,
                       resolver="order_status_delivered")]
        scripted = ["TRUE: broken on arrival." + chr(10) + "FALSE: wrong size."]
        out, log = add_criteria(MockFrontierClient(scripted=scripted), qs)
        self.assertIsNotNone(out[0].criteria)
        self.assertIsNone(out[1].criteria)  # computed skipped, no frontier call
        self.assertEqual(log[0]["status"], "ok")

    def test_prompt_mentions_boundary_cases(self):
        system, user = build_criteria_prompt(Question(text="Q?", qtype=QuestionType.SEMANTIC))
        self.assertIn("boundary", system)
        self.assertIn("TRUE:", user)


if __name__ == "__main__":
    unittest.main()
