import unittest

from sdt.jev_client import HttpJevClient, MockJevClient
from sdt.schemas import Question, QuestionType, SdtError

CTX_DELIVERED = {
    "orders": {"#A1": {"status": "delivered", "payment_history": []}},
    "focus_order_id": "#A1",
    "user": {"payment_methods": {}},
}
CTX_PENDING = {
    "orders": {"#P1": {"status": "pending", "payment_history": []}},
    "focus_order_id": "#P1",
    "user": {"payment_methods": {}},
}

Q_DELIVERED = Question(text="Is the focus order delivered?", qtype=QuestionType.COMPUTED, resolver="order_status_delivered")
Q_REFUND = Question(text="Does the user request a refund?", qtype=QuestionType.SEMANTIC)


class TestMockJev(unittest.TestCase):
    def test_computed_resolved_exactly(self):
        self.assertEqual(MockJevClient().score_batch("t", CTX_DELIVERED, [Q_DELIVERED]).scores[Q_DELIVERED.qid], 1.0)
        self.assertEqual(MockJevClient().score_batch("t", CTX_PENDING, [Q_DELIVERED]).scores[Q_DELIVERED.qid], 0.0)

    def test_oracle_respected_and_deterministic(self):
        client = MockJevClient(oracle={Q_REFUND.cache_key: 0.91})
        sv = client.score_batch("t", CTX_DELIVERED, [Q_REFUND])
        self.assertEqual(sv.scores[Q_REFUND.qid], 0.91)
        sv2 = client.score_batch("t", CTX_DELIVERED, [Q_REFUND])
        self.assertEqual(sv.scores, sv2.scores)

    def test_unknown_resolver_raises(self):
        q = Question(text="x?", qtype=QuestionType.COMPUTED, resolver="nope")
        with self.assertRaises(SdtError):
            MockJevClient().score_batch("t", CTX_DELIVERED, [q])

    def test_http_client_fails_loudly(self):
        with self.assertRaises(SdtError):
            HttpJevClient().score_batch("t", CTX_DELIVERED, [Q_REFUND])

    def test_orderless_context_answers_false_not_crash(self):
        sv = MockJevClient().score_batch("t", {"orders": {}, "user": {}}, [Q_DELIVERED])
        self.assertEqual(sv.scores[Q_DELIVERED.qid], 0.0)

    def test_null_user_context_answers_false_not_crash(self):
        from sdt.schemas import Question, QuestionType
        q = Question(text="Does the account hold a gift card?",
                     qtype=QuestionType.COMPUTED, resolver="has_gift_card")
        sv = MockJevClient().score_batch("t", {"orders": {}, "user": None}, [q])
        self.assertEqual(sv.scores[q.qid], 0.0)


if __name__ == "__main__":
    unittest.main()
