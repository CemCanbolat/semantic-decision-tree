import json
import unittest
from typing import Dict, List, Tuple

from sdt.gemini_decision_client import GeminiDecisionClient
from sdt.schemas import Question, QuestionType, SdtError

Q_SEM_1 = Question(text="Does the user request a return or refund?", qtype=QuestionType.SEMANTIC)
Q_SEM_2 = Question(text="Is the customer threatening legal action?", qtype=QuestionType.SEMANTIC)
Q_COMPUTED = Question(text="Is the focus order delivered?", qtype=QuestionType.COMPUTED, resolver="order_status_delivered")

CTX = {"orders": {"#A1": {"status": "delivered", "payment_history": []}}, "focus_order_id": "#A1", "user": {}}


def _ok_body(questions):
    return json.dumps({
        "answers": {q.qid: {"type": "noul", "noul": 0.83, "confidence": 0.9} for q in questions},
        "usage": {"total_input_tokens": 100, "total_output_tokens": 0},
    }).encode("utf-8")


class FakeTransport:
    def __init__(self, responses: List[Tuple[int, bytes]]):
        self.responses = list(responses)
        self.requests: List[Tuple[str, Dict, bytes]] = []

    def __call__(self, url: str, headers: Dict, body: bytes):
        self.requests.append((url, headers, body))
        return self.responses.pop(0)


class TestRequestShape(unittest.TestCase):
    def test_semantic_rides_along_computed_stays_local(self):
        ft = FakeTransport([(200, _ok_body([Q_SEM_1]))])
        client = GeminiDecisionClient(api_key="k", transport=ft)
        sv = client.score_batch("return please", CTX, [Q_SEM_1, Q_COMPUTED])
        url, headers, body = ft.requests[0]
        req = json.loads(body.decode("utf-8"))
        self.assertEqual(req["model"], "gemini-decision-1")
        self.assertEqual(list(req["questions"].keys()), [Q_SEM_1.qid])
        self.assertEqual(req["questions"][Q_SEM_1.qid]["type"], "noul")
        self.assertEqual(req["questions"][Q_SEM_1.qid]["instructions"], Q_SEM_1.qid and Q_SEM_1.text)
        self.assertIsInstance(req["input"], list)
        self.assertEqual(req["input"][0]["type"], "text")
        self.assertIn("Customer ticket:", req["input"][0]["text"])
        self.assertIn("Account and order context JSON:", req["input"][1]["text"])
        self.assertEqual(headers["x-goog-api-key"], "k")
        self.assertEqual(sv.scores[Q_COMPUTED.qid], 1.0)  # resolved locally, never sent
        self.assertEqual(sv.scores[Q_SEM_1.qid], 0.83)   # parsed from response

    def test_computed_only_never_hits_network(self):
        ft = FakeTransport([])
        client = GeminiDecisionClient(api_key="k", transport=ft)
        sv = client.score_batch("t", CTX, [Q_COMPUTED])
        self.assertEqual(ft.requests, [])
        self.assertEqual(sv.scores[Q_COMPUTED.qid], 1.0)


class TestRetriesAndErrors(unittest.TestCase):
    def test_retry_on_429_then_success(self):
        ft = FakeTransport([(429, b"rate limited"), (429, b"rate limited"), (200, _ok_body([Q_SEM_1]))])
        client = GeminiDecisionClient(api_key="k", transport=ft, max_retries=3)
        sv = client.score_batch("t", CTX, [Q_SEM_1])
        self.assertEqual(len(ft.requests), 3)
        self.assertEqual(sv.scores[Q_SEM_1.qid], 0.83)

    def test_gives_up_after_max_retries(self):
        responses = [(503, b"unavailable")] * 4
        ft = FakeTransport(responses)
        client = GeminiDecisionClient(api_key="k", transport=ft, max_retries=3)
        with self.assertRaises(SdtError):
            client.score_batch("t", CTX, [Q_SEM_1])

    def test_4xx_does_not_retry(self):
        ft = FakeTransport([(400, b"bad request")])
        client = GeminiDecisionClient(api_key="k", transport=ft)
        with self.assertRaises(SdtError):
            client.score_batch("t", CTX, [Q_SEM_1])
        self.assertEqual(len(ft.requests), 1)

    def test_missing_answer_key_raises(self):
        ft = FakeTransport([(200, json.dumps({"answers": {}}).encode())])
        client = GeminiDecisionClient(api_key="k", transport=ft)
        with self.assertRaises(SdtError):
            client.score_batch("t", CTX, [Q_SEM_1])

    def test_missing_key_raises_with_guidance(self):
        client = GeminiDecisionClient(api_key=None, transport=FakeTransport([]))
        import os
        saved = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with self.assertRaises(SdtError) as ctx:
                client.score_batch("t", CTX, [Q_SEM_1])
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))
        finally:
            if saved:
                os.environ["GEMINI_API_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
