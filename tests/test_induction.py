import unittest

from sdt.data.synth_tickets import generate
from sdt.eval.baselines import run_b2_soft
from sdt.eval.heuristic_jev import HeuristicJevClient
from sdt.induction.gain import entropy, evaluate_question, rank_questions
from sdt.induction.grow import TreeInducer
from sdt.induction.propose import build_proposal_prompt, leakage_check, parse_questions
from sdt.jev_client import JevClient
from sdt.llm_client import FrontierClient
from sdt.schemas import DecisionLabel, Example, Question, QuestionType
from sdt.training.cache import ScoreCache


def _ex(eid, gold, text="t"):
    return Example(example_id=eid, text=text, context={}, gold=gold)


class PairAwareFrontier(FrontierClient):
    """Returns a discriminating question for each contrastive pair it is shown."""

    PROPOSALS = {
        ("auto_approve", "cancel_order"): "Does the user request a return or refund of items from the order?",
        ("auto_approve", "politely_deny"): "Does the ticket indicate the order has already been delivered?",
        ("auto_approve", "exchange"): "Does the user request a return or refund of items from the order?",
        ("cancel_order", "exchange"): "Does the user request cancellation of a pending order?",
        ("cancel_order", "politely_deny"): "Does the user request cancellation of a pending order?",
        ("exchange", "politely_deny"): "Does the user request an exchange of items for a different variant of the same product?",
    }

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, max_tokens=1024):
        self.calls += 1
        pos = user.split("GROUP A (label: ")[1].split(")")[0]
        neg = user.split("GROUP B (label: ")[1].split(")")[0]
        q = self.PROPOSALS.get((pos, neg))
        if q is None:
            return "Does the ticket mention the weather?"
        return q + chr(10) + "Is the message written in English?"


FOUR_FAMILIES = (DecisionLabel.AUTO_APPROVE, DecisionLabel.POLITELY_DENY,
                 DecisionLabel.EXCHANGE, DecisionLabel.CANCEL_ORDER)


def _corpus():
    return [e for e in generate(n_per_family=4, seed=7) if e.gold in FOUR_FAMILIES]


class TestGain(unittest.TestCase):
    def test_perfect_split_scores_max(self):
        exs = ([_ex("a%d" % i, DecisionLabel.AUTO_APPROVE) for i in range(4)]
               + [_ex("b%d" % i, DecisionLabel.POLITELY_DENY) for i in range(4)])
        info = evaluate_question(exs, "q", [0.9] * 4 + [0.1] * 4)
        self.assertAlmostEqual(info.ig, 1.0, places=6)
        self.assertAlmostEqual(info.certainty, 0.8, places=6)
        self.assertGreater(info.score, 0.75)

    def test_useless_split_zero_ig(self):
        exs = [_ex("a", DecisionLabel.AUTO_APPROVE), _ex("b", DecisionLabel.AUTO_APPROVE)]
        info = evaluate_question(exs, "q", [0.9, 0.1])
        self.assertEqual(info.ig, 0.0)

    def test_uncertain_partition_penalized(self):
        exs = ([_ex("a%d" % i, DecisionLabel.AUTO_APPROVE) for i in range(4)]
               + [_ex("b%d" % i, DecisionLabel.POLITELY_DENY) for i in range(4)])
        sharp = evaluate_question(exs, "q1", [0.95] * 4 + [0.05] * 4)
        hazy = evaluate_question(exs, "q2", [0.6] * 4 + [0.4] * 4)
        self.assertGreater(sharp.score, hazy.score)  # same IG, different certainty

    def test_rank_respects_min_child(self):
        exs = ([_ex("a", DecisionLabel.AUTO_APPROVE)]
               + [_ex("b%d" % i, DecisionLabel.POLITELY_DENY) for i in range(4)])
        ranked = rank_questions(exs, {"q": [0.9] + [0.1] * 4}, min_child=2)
        self.assertEqual(ranked, [])


class TestPropose(unittest.TestCase):
    def test_leakage_guard(self):
        self.assertTrue(leakage_check("Should this ticket be auto-approved by the system?"))
        self.assertTrue(leakage_check("Is this an out of scope request?"))
        self.assertFalse(leakage_check("Does the user request a refund for the order?"))

    def test_parse_strips_numbering_and_dedupes(self):
        raw = ("1. Does the user request a refund?" + chr(10)
               + "- Does the user request a refund?" + chr(10)
               + "2) Does the order appear delivered?" + chr(10)
               + "junk line without question mark")
        qs = parse_questions(raw, "test")
        self.assertEqual(len(qs), 2)
        self.assertTrue(all(q.qtype is QuestionType.SEMANTIC for q in qs))

    def test_prompt_contains_groups_not_labels_in_questions(self):
        corpus = _corpus()
        pos = [e for e in corpus if e.gold is DecisionLabel.AUTO_APPROVE][:4]
        neg = [e for e in corpus if e.gold is DecisionLabel.POLITELY_DENY][:4]
        system, user = build_proposal_prompt(pos, neg, DecisionLabel.AUTO_APPROVE, DecisionLabel.POLITELY_DENY)
        self.assertIn("GROUP A", user)
        self.assertIn("GROUP B", user)
        self.assertIn("yes/no", system)


class TestInduction(unittest.TestCase):
    def test_induced_tree_separates_four_families(self):
        corpus = _corpus()
        inducer = TreeInducer(frontier=PairAwareFrontier(), scorer=HeuristicJevClient(), min_child=2)
        tree = inducer.induce(corpus, tree_id="test_induced")
        questions = list(inducer._questions.values())
        self.assertLessEqual(len(questions), 4)

        records = run_b2_soft(HeuristicJevClient(), corpus, tree=tree, questions=questions)
        correct = sum(1 for r, e in zip(records, corpus) if r.predicted is e.gold)
        self.assertEqual(correct, len(corpus))  # all 16 routed correctly

    def test_cache_eliminates_rescoring(self):
        corpus = _corpus()
        cache = ScoreCache()
        first = TreeInducer(frontier=PairAwareFrontier(), scorer=HeuristicJevClient(), cache=cache, min_child=2)
        first.induce(corpus)
        self.assertGreater(first.stats.scorer_calls, 0)
        second = TreeInducer(frontier=PairAwareFrontier(), scorer=HeuristicJevClient(), cache=cache, min_child=2)
        second.induce(corpus)
        self.assertEqual(second.stats.scorer_calls, 0)  # every (q, example) cached
        self.assertGreater(second.stats.cache_hits, 0)

    def test_question_budget_respected(self):
        corpus = _corpus()
        inducer = TreeInducer(frontier=PairAwareFrontier(), scorer=HeuristicJevClient(),
                              max_questions=1)
        tree = inducer.induce(corpus)
        self.assertEqual(len(tree.questions()), 1)

    def test_pure_node_becomes_leaf_without_frontier_call(self):
        corpus = [e for e in _corpus() if e.gold is DecisionLabel.AUTO_APPROVE]
        frontier = PairAwareFrontier()
        inducer = TreeInducer(frontier=frontier, scorer=HeuristicJevClient())
        tree = inducer.induce(corpus)
        self.assertEqual(frontier.calls, 0)
        self.assertEqual(len(tree.leaves()), 1)
        self.assertIs(tree.nodes[tree.root].leaf_label, DecisionLabel.AUTO_APPROVE)

    def test_prune_collapses_uniform_subtree(self):
        from sdt.schemas import Tree, TreeNode
        inducer = TreeInducer(frontier=PairAwareFrontier(), scorer=HeuristicJevClient())
        nodes = {
            "root": TreeNode("root", "q1", branches={'yes': "left", 'no': "right"}),
            "left": TreeNode("left", "q2", branches={'yes': "l1", 'no': "l2"}),
            "l1": TreeNode("l1", leaf_label=DecisionLabel.AUTO_APPROVE),
            "l2": TreeNode("l2", leaf_label=DecisionLabel.AUTO_APPROVE),  # uniform subtree
            "right": TreeNode("right", leaf_label=DecisionLabel.POLITELY_DENY),
        }
        tree = Tree("t", "root", nodes)
        pruned = inducer._prune(tree)
        self.assertGreaterEqual(pruned, 1)
        self.assertTrue(tree.nodes["left"].is_leaf)
        self.assertIs(tree.nodes["left"].leaf_label, DecisionLabel.AUTO_APPROVE)
        self.assertFalse(tree.nodes["root"].is_leaf)  # siblings differ, root stays


if __name__ == "__main__":
    unittest.main()


class CapturingFrontier(FrontierClient):
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user, max_tokens=1024):
        self.prompts.append(user)
        return self.replies.pop(0)


class FakeScorer(JevClient):
    """Canned p per (question cache key, example id)."""

    def __init__(self, table):
        self.table = table

    def score_batch(self, text, context, qs):
        from sdt.schemas import ScoreVector
        from sdt.jev_client import DEFAULT_RESOLVERS
        scores = {}
        for q in qs:
            if q.qtype is QuestionType.COMPUTED:
                scores[q.qid] = 1.0 if DEFAULT_RESOLVERS[q.resolver](context) else 0.0
            else:
                scores[q.qid] = self.table[(q.qid, text)]
        return ScoreVector(example_id="", scores=scores)


class TestIteration1Fixes(unittest.TestCase):
    def test_negated_candidates_rejected(self):
        raw = ("Does the text omit any mention of a return?" + chr(10)
               + "Does the ticket contain no mention of cancellation?" + chr(10)
               + "Does the user request a refund for the order?")
        qs = parse_questions(raw, "t")
        self.assertEqual([q.text for q in qs],
                         ["Does the user request a refund for the order?"])

    def test_minority_class_gets_ovr_prompt(self):
        from sdt.induction.grow import TreeInducer as TI
        corpus = ([_ex("a%02d" % i, DecisionLabel.AUTO_APPROVE, "refund text %d" % i) for i in range(20)]
                  + [_ex("b%02d" % i, DecisionLabel.POLITELY_DENY, "angry text %d" % i) for i in range(5)]
                  + [_ex("c%02d" % i, DecisionLabel.ESCALATE, "legal threat %d" % i) for i in range(2)])
        frontier = CapturingFrontier(["q about refund?" + chr(10) + "q about legal?"] * 4)
        inducer = TI(frontier=frontier, scorer=HeuristicJevClient())
        inducer._propose(corpus)
        ovr_prompts = [u for u in frontier.prompts if "GROUP B (label: other)" in u]
        self.assertTrue(ovr_prompts, "one-vs-rest prompt must fire for the 2-example minority")
        self.assertIn("GROUP A (label: escalate)", ovr_prompts[0])
        pair_prompts = [u for u in frontier.prompts if "GROUP B (label: politely_deny)" in u]
        self.assertTrue(pair_prompts)

    def test_class_weights_zero_credit_for_majority_only_split(self):
        from sdt.induction.gain import class_weights, evaluate_question
        corpus = ([_ex("a%02d" % i, DecisionLabel.AUTO_APPROVE) for i in range(20)]
                  + [_ex("b%02d" % i, DecisionLabel.ESCALATE) for i in range(4)])
        wmap = class_weights(corpus)
        total = sum(wmap[e.gold] for e in corpus)
        self.assertAlmostEqual(total, len(corpus), places=6)
        q_major = "qmajor"   # splits majority in half, rare class split evenly too
        ps_major = [0.9] * 10 + [0.1] * 10 + [0.9, 0.1, 0.9, 0.1]
        q_rare = "qrare"    # isolates the rare class perfectly
        ps_rare = [0.1] * 20 + [0.9, 0.9, 0.9, 0.9]
        s_major = evaluate_question(corpus, q_major, ps_major, wmap)
        s_rare = evaluate_question(corpus, q_rare, ps_rare, wmap)
        self.assertLess(s_major.ig, 0.1)   # majority-halving earns almost nothing weighted
        self.assertGreater(s_rare.ig, 0.9)

    def test_validation_rejects_memorizer(self):
        from sdt.induction.grow import TreeInducer as TI
        from sdt.schemas import Question, QuestionType

        q_gen = Question(text="Does the user request a refund for the order?", qtype=QuestionType.SEMANTIC)
        q_mem = Question(text="Does the user mention a tablet stand?", qtype=QuestionType.SEMANTIC)

        # ids: memorized train ids sort LAST (z*), reversed val ids sort FIRST (a*)
        mk = lambda eid, gold: _ex(eid, gold, eid)
        train = ([mk("za%dx" % i, DecisionLabel.AUTO_APPROVE) for i in range(6)]
                 + [mk("zb%dy" % i, DecisionLabel.POLITELY_DENY) for i in range(6)])
        val = ([mk("aa%dy" % i, DecisionLabel.AUTO_APPROVE) for i in range(2)]
               + [mk("ab%dx" % i, DecisionLabel.POLITELY_DENY) for i in range(2)])
        all_ex = sorted(train + val, key=lambda e: e.example_id)

        table = {}
        for e in all_ex:
            gold_yes = e.gold is DecisionLabel.AUTO_APPROVE
            table[(q_gen.qid, e.example_id)] = 0.9 if gold_yes else 0.1
            marker = e.example_id.endswith("x")
            table[(q_mem.qid, e.example_id)] = 0.9 if marker else 0.1
        scorer = FakeScorer(table)

        raw_replies = [q_gen.text + chr(10) + q_mem.text] * 4
        inducer = TI(frontier=CapturingFrontier(raw_replies), scorer=scorer, min_child=2)
        tree = inducer.induce(all_ex, tree_id="t_mem")

        root_qid = tree.nodes[tree.root].question_qid
        self.assertEqual(root_qid, q_gen.qid)  # generalizer wins despite memorizer's perfect train fit

        mem_rows = [c for c in inducer.stats.candidates if c["question"] == q_mem.text]
        self.assertTrue(mem_rows)
        self.assertLess(mem_rows[0]["val_score"], 0.5)  # memorizer: val 0.0 << baseline 0.5 -> vetoed
        gen_rows = [c for c in inducer.stats.candidates if c["question"] == q_gen.text]
        self.assertTrue(any(r["picked"] for r in gen_rows))
