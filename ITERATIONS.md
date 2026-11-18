# Iteration Log

Plain-language record of what we tried, what happened, and what we changed.
Newest iterations go at the bottom.

---

## Iteration 0 - First live run (2026-11-14)

**What we built:** a hand-written decision tree with 10 questions about
refunds, order status, and cancellations. Every question answered by
Gemini Decision 1 in one parallel batch. Also: our first automatically
induced tree (questions proposed by a chat model, scored by Decision 1).

**What happened:**

- The hand-written tree was bad: 17.5% accuracy. Two of my questions were
  poorly worded. One asked "is this within agent scope so no transfer is
  needed?" - the model answered "not sure" (0.39) even for a simple return,
  so everything fell into the escalate bucket. Another confused "has an
  order number" with "can be identified" - many customers identify by name
  and zip code instead.
- The induced tree was 3x better: 66.7% accuracy. It found great questions
  on its own, like one that spots exchange requests with near-perfect
  accuracy.
- Three problems appeared, all classic ML problems in new clothes:
  1. Class imbalance: rare outcomes (escalate, address change, denial) got
     zero questions proposed for them, so the tree never learned them.
  2. Overfitting: the tree memorized product names from the small dataset
     ("tablet, camera, air purifier") because candidates were judged only
     on the data that proposed them.
  3. Question phrasing: the most important question (the root) was phrased
     negatively ("does the text omit any mention of...") and the model
     answered it with the least confidence of any question in the tree.
- The safety gates worked exactly as designed: when the tree was confused,
  the system auto-executed almost nothing, and what it did execute was
  correct.

## Iteration 1 - Imbalance, overfitting, phrasing fixes (2026-11-14)

**What we changed:**

1. Imbalance: rare classes now get their own "this class versus everything
   else" question proposals, and question selection credits rare classes
   more (balanced weighting), so finding all 4 escalate examples counts
   almost as much as splitting 30 modify examples.
2. Overfitting: every node now holds out 30% of its examples as a
   validation check. A question must beat the do-nothing baseline on that
   held-out set to be used at all. Questions naming rare words (memorized
   product names) get their score discounted automatically - no hand-
   maintained blocklist. Every candidate, picked or not, is now logged.
3. Phrasing: negatively-phrased questions are rejected at parse time
   ("does the text omit..."), and the proposal prompt instructs positive
   phrasing, generalization over memorization, and one condition per
   question. This is free: a negative question is the same as its positive
   twin with the branches swapped, so nothing is lost.

**Expected effects (to verify in the next live run):**

- Rare classes get nonzero recall (target: escalate/modify_user from 0%
  toward ~50%; 4-6 examples each caps what is provable).
- Memorized product questions get rejected by validation or discounted by
  rarity.
- Higher macro-F1 (the metric that counts rare classes equally); accuracy
  may dip slightly if rare classes win questions from big classes.
- More frontier calls per induction (one-vs-rest prompts), same cheap
  scoring thanks to the cache.

**Status:** built and unit-tested offline (87 tests). Awaiting live run.

## Iteration 1 - Results (2026-11-14, live)

**What happened:** the fixes each worked, but together they overcorrected.

- The one-vs-rest prompts fired for every minority class, exactly as
  designed (11 address-change candidates, 8 denial candidates, 7 escalate
  candidates).
- The validation gate then killed the tree. On the branch holding most of
  the data, all 18 candidate questions scored slightly below the
  do-nothing baseline on the ~30 held-out examples, so the branch became a
  single leaf. Result: a 2-question, 3-leaf tree that mostly predicts the
  majority class. Accuracy fell to 35.1% (from 66.7%), macro-F1 to 0.16.

**The lesson:** asking a question to immediately improve final accuracy on
30 noisy examples is the wrong bar. Most good questions just reduce mixing
- their children still contain several classes, so their two-leaf accuracy
  barely moves. That is fine for a tree; it is fatal for a hard gate.
Iteration 1 confused "do not split harmfully" with "split must prove
terminal accuracy now." Classic overcorrection: iteration 0 overfit,
iteration 1 underfit.

## Iteration 2 - Validation as veto, gain as selector (2026-11-14)

**What we changed:** selection is back to class-weighted training gain
(with the rare-word discount). Validation no longer has to be beaten - it
only vetoes a candidate that is clearly harmful on held-out data (more
than 5 points below the do-nothing baseline). A memorized question still
dies (its validation score collapses); a good mixing-reducing question
survives.

**Expected effects:** a tree close to iteration 0's size and accuracy,
plus the minority questions and phrasing rules of iteration 1 - and every
candidate still logged for audit. The memorizer unit test still passes:
its validation score (0.0) vetoes it outright.

**Status:** built and unit-tested (87 tests). Awaiting live run.

## Iteration 2 - Results (2026-11-14, live)

**What happened:** recovered from the collapse - 64.9% accuracy (vs 35.1%
iterated-1, 66.7% iteration-0), 7 questions, 8 leaves. The mechanisms
worked individually:

- Politely-deny went from 1/5 to 5/5 - the first minority-class win, ever.
- Two of seven picked questions came from one-vs-rest prompts (the
  minority machinery is now shaping the tree, not just proposing into a
  void).
- Zero memorized product-name questions survived - the prompt guidance,
  rarity discount, and validation veto kept them out entirely (iteration-0
  had two in its final tree).
- The coverage curve got much healthier: peak precision 0.70 now holds at
  80% coverage (iteration-0 peaked 0.73 at only 10% coverage) - a far
  better auto-execute-most profile.
- Every question is positively phrased; the root is a clean exchange
  detector with certainty 0.83.

**Remaining gaps:**

- Escalate is still 0/4 and modify-user 0/6. Their questions get proposed
  (10 and 24 candidates respectively) and one even gets picked - but the
  leaf beneath is labeled with the wrong majority, because leaf labels use
  plain counts and 4-6 examples lose every count-based vote.
- Accuracy dipped 1.8 points below iteration-0 (auto-approve and cancel
  traded places with modify-order), so the iteration-0/2 comparison is a
  wash on accuracy while being clearly better on minority classes and
  memorization control.
- Still no 97%-precision operating point: calibration, not structure, is
  the bottleneck now.

**Next direction (iteration 3):** class-weighted leaf labels (cheap fix for
the mislabeled minority leaves), then the Phase-4 combiner - a temperature-
scaled classifier trained on the cached p-vectors, zero new API calls.
The tree found the features; the combiner learns the decision surface.

## Iteration 3 - Combiner on cached features (2026-11-14)

**Built:** a temperature-scaled softmax classifier (pure Python, class-
weighted, 5-fold cross-validated) that trains on the cached question
probabilities plus four exact computed features (order delivered, order
pending, refund count, gift card) - zero API calls for training.

**Sub-idea tested and refuted - weighted leaf relabeling.** The plan was to
relabel tree leaves by class-weighted majority so minority questions stop
losing count votes. Two findings killed it:

1. Balanced weights tie exactly in any majority vote (weight x count is the
   same constant for every class), so "weighted majority" is a no-op that
   falls to an alphabetical tie-break.
2. Raising the weight power above 1 to break ties hands EVERY leaf to the
   rarest class present, regardless of counts. Measured: accuracy collapsed
   from 0.64 to 0.12 with 8 leaf changes.

The lesson: leaf labels cannot be fixed by reweighting. Rare classes have
to earn influence through features that predict them across the whole
corpus - which is precisely the combiner's job.

**Blocked pending a cache backfill:** induction only scores each question
on the examples that reach its node, but a flattened feature set needs
every question against every example. 435 score pairs are missing for the
v3 feature set (1080 for v3+v1). The backfill command exists
(sdt.training.backfill_cache) and needs one networked run (~1 minute).

## Iteration 3 - Results (2026-11-14)

**All three stated predictions confirmed:**

1. Macro-F1 0.4913 (5-fold out-of-fold, honest) - the best of any system,
   beating even the trees' IN-SAMPLE numbers (v1: 0.456, v3: 0.393).
2. The untouched rare classes moved: escalate 1/4 and modify_user 1/6 -
   their first nonzero recalls in any iteration. Exchange hit 23/26 (89%),
   far beyond what any tree extracted from the same questions.
3. The first 97%-precision operating point ever: 10.5% of tickets
   auto-executable at 100% precision, holding 100% precision out to 10.5%
   coverage and ~75% precision at 30-50% coverage. Temperature 0.71
   (the model is slightly underconfident; sharpening helps).

**Feature ablation:** adding iteration-0's questions (20 features) raises
accuracy to 0.658 but lowers macro-F1 (0.488) and halves the 97%-precision
coverage (5.3%). Eleven features is the better operating point: more
features help majority classes and dilute rare-class precision.

**Trade-off observed:** politely_deny fell back to 1/5 (the v3 tree had
5/5) while escalate/modify_user/out_of_scope gained. Net macro-F1 still
well ahead - the combiner spreads influence by predictive value instead of
leaf geography.

**Deployment bundle now complete:** tree artifact (question batch) +
computed resolvers + combiner weights + temperature + gates = one decision
batch per ticket, microsecond combiner, escalate the rest.

## Iteration 4 - Noul criteria (2026-11-14)

**Idea:** the Decisions API accepts an optional criteria block per noul
question (true_description / false_description) pinning down exactly what
yes and no mean. We had never used it. The frontier model now writes
criteria for every question in the frozen v3 tree; the criteria'd
questions are re-scored under fresh cache identities (a question with
criteria is a new scoring unit), and the combiner is retrained - a clean
A/B: identical questions, with and without criteria.

**Falsifiable expectation:** sharper boundary definitions raise certainty
on confusable questions (cancel-vs-modify, exchange-vs-return), improving
macro-F1 and the coverage-at-97%-precision point over iteration 3
(0.4913 / 10.5%). If criteria are noise - generic restatements - the
numbers will not move.

**Status:** built and unit-tested (97 tests). Awaiting live run
(stage 4 of the pipeline).

## Iteration 4 - Results (2026-11-14)

**Split verdict, resolved by per-question selection.** Criteria are
high-quality (the frontier wrote real boundary disambiguation, e.g. the
exchange question's false-description explicitly covers "identical
replacement for defects" - the classic confusion) but they are a
per-question instrument, not a global one.

- Wholesale criteria: macro-F1 0.4913 -> 0.4321 (hurt), but coverage-at-
  97%-precision 10.5% -> 14.0% (helped).
- The damage traces to one question: "changes or modifications to an order
  or its payment" flipped hard assignments on 53 of 114 tickets (mean
  probability shift 0.41) - the frontier's definition of "modification"
  disagrees with the benchmark's action space.
- Per-question CV selection: only the city/geographic-location question
  (the modify_user rare-class proxy) clearly benefits from its criteria
  (+0.013 macro-F1); exchange, already-received, modify, and return all
  get worse; multi-order and pending are neutral.
- Best system overall: plain questions + city-question criteria + computed
  features - accuracy 0.6316, macro-F1 0.5040, coverage@97% precision
  11.4% at 100% precision. Best on every discrimination metric, slightly
  below wholesale-criteria on the coverage head.

**Principle learned:** criteria inject the frontier's ontology into the
scorer. Where the question's boundary is genuinely ambiguous (rare-class
proxies), that helps; where the model was already well calibrated
(exchange certainty 0.84), perturbing it hurts. Like every other choice
in this pipeline, criteria adoption is decided empirically per question,
by cross-validation, after the fact.

## Iteration 5 - Fresh induction with computed seeds (2026-11-14)

**Idea:** re-run induction from scratch. Two things differ from the v3 run:
the frontier samples at temperature 0.7, so a fresh run explores different
proposals (run-to-run variance is itself worth measuring); and the exact
computed features (order delivered, order pending, refund count, gift card)
now compete in every candidate pool on identical terms - gain, validation
veto, rarity discount. v3 spent two semantic questions re-deriving order
status that the JSON answers exactly; this run can discover that for free.

**Falsifiable expectations:**

1. The tree differs from v3 - different root, different questions, some
   overlap. If it were identical, the induction would be more deterministic
   than we believed.
2. Where order status discriminates, the computed seeds displace semantic
   status questions (exact 0/1 beats cert ~0.7).
3. Combiner macro-F1 lands within 0.03 of the 0.504 champion unless the
   fresh question set is materially better or worse.
4. Coverage at 97% precision stays in the 10-15% band.

**Status:** built and unit-tested (97 tests). Awaiting live run
(stage 5 of the pipeline).

## Iteration 5 - Results (2026-11-14)

**Pre-registered scorecard: two confirmed, two falsified.**

1. CONFIRMED - the tree is genuinely different: 4 questions, 5 leaves, a
   spec-change root instead of v3's exchange root.
2. CONFIRMED - the computed pending seed won a node (information gain 0.80,
   certainty 1.000), displacing semantic status detection exactly as
   predicted.
3. FALSIFIED - the fresh tree's combiner scored macro-F1 0.4092 with no
   97%-precision operating point, far outside the predicted 0.47-0.53 band.
4. FALSIFIED - no coverage point at all (band was 10-15%).

**Diagnosis:** run-to-run variance is large. The temperature-0.7 sampler
produced only 3 semantic questions (v3 found 7) and skipped the
return-detector that carried v3. A single induction is a lottery draw.

**The surprise that reframes the result:** the union of both runs' question
banks set a project record - macro-F1 0.5632 (champion was 0.5040), accuracy
0.6316, coverage at 97% precision 7.9%. The individually-weaker run
contributed three questions v3 never found, above all a direct
address-change question (certainty 0.948) that lifted modify_user recall
from 1/6 to 4/6. Every rare class is now nonzero: escalate 1/4,
politely_deny 3/5, out_of_scope 2/2, exchange 24/26.

**Strategic lesson:** individual trees are lottery draws; the question bank
is the compounding asset. Every induction run, good or bad by itself, adds
features to the bank - which is the semantic-forest idea from the original
plan, discovered empirically. Future iterations should induce diverse trees
on purpose and combine through the bank, and re-run selection on the whole
bank rather than a single tree's questions.

Also fixed: computed seeds passed as tree questions no longer duplicate the
seed columns in the feature matrix.

## Iteration 6 - Question bank + multi-seed induction (2026-11-14)

**Built:** the question bank as a first-class artifact - every question ever
picked into any tree, deduplicated by text, carrying provenance and any
criteria earned (criteria enrich, never overwrite). A multi-seed runner
induces N trees with different contrastive sampling seeds and deposits all
their questions into the shared bank; a bank combiner then selects over the
whole pool at once.

**Offline bootstrap result (v1 + v3 + v4 banked, 23 features, all cached):**
accuracy 0.6491 - the best yet - but macro-F1 0.5340, BELOW the v3+v4
union's 0.5632. Escalate and politely_deny fell back to zero. The pattern
is consistent with iteration 3's ablation: every added feature helps
majority classes and dilutes rare-class precision.

**Lesson:** a bank without selection is a swamp. Accumulation compounds
only if selection prunes - the next piece is greedy forward selection by
cross-validated macro-F1 (or L1), choosing per question whether it earns
a slot, mirroring how criteria adoption was decided per question in
iteration 4.

**Status:** bank + multi-seed + bank combiner built and unit-tested
(102 tests). Live multi-seed run staged as pipeline stage 6.

## Iteration 6 - Results (2026-11-14)

**Multi-seed run:** three seed-varied trees (224s + 211s + 262s), genuinely
diverse roots (modify-attribute, address-change, spec-change). The bank
grew 19 -> 29 questions. Computed seeds won nodes in all three trees
(pending twice, gift card once) - five seed wins across two runs now.

**Bank-all combiner (33 features):** accuracy 0.6491 (tied with the smaller
bank), macro-F1 degraded further to 0.5178, escalate and politely_deny at
zero. Confirmed at scale: a bank without selection is a swamp.

**Greedy forward selection by CV macro-F1** (computed seeds mandatory, then
one question at a time, stop when nothing improves by 0.003) picked five
semantic questions and stopped:

  0.1917 (computed only) -> 0.3770 -> 0.4722 -> 0.5161 -> 0.5548 -> 0.5736

**The selected system (9 features) is the new champion and solves the
rare-class problem:** macro-F1 0.5736, accuracy 0.6579 (best honest
number), escalate 3/4 (from zero in four previous iterations), modify_user
6/6 perfect, out_of_scope 2/2, exchange 23/26. The two delivery-status
phrasings were BOTH selected - the same fact asked two ways is worth more
than either alone (an ensemble-of-phrasings effect).

**The trade-off made visible:** the high-precision automation head shrank -
coverage at 97% precision fell to 0.9% (temperature 0.62), while the fat
33-feature bank holds 6.1%. Macro-F1 selection builds the best balanced
adjudicator; feature-rich banks build the best auto-execution gate. Those
are different operating points, and the right next question is whether one
selector can optimize both - e.g. selection by macro-F1 with a
coverage-at-precision constraint.

Model artifact: eval/reports/selected_model.json (9 features, temperature,
weights - the complete deployment bundle).

## Iteration 7 - Multi-type decisions: choice, score, and free-type (2026-11-14)

**Protocol (per the experimenter):** explore Gemini Decision 1's choice and
score question types alongside nouls; NO noul criteria anywhere (choice
option maps and score level arrays are API-required structure, not
criteria, and are documented as such). Three experiments:

- **A - constrained:** exactly ONE choice question (options = the eight
  outcome classes with frontier-written descriptions) plus ONE score
  question (frontier-designed ordered rubric). The tree is genuinely
  non-binary: the choice node branches per class, the score node grades
  per level. Evaluates tree soft-routing and a flat combiner over all
  probability columns.
- **B - free-type induction:** the proposal prompt explains all three
  formats and the frontier picks per question. A new free-type inducer
  grows mixed trees (noul 2-way, choice K-way, score L-way) under the
  champion discipline: class-weighted gain, validation veto, rarity
  discount. Question: do LLMs write better questions when given the full
  menu?
- **C (self-designed) - paraphrase ensemble:** iteration 6 kept BOTH
  delivery-status phrasings. Formalize it: the frontier writes 2
  paraphrases of each of the 5 selected questions; combiner trains on
  5 originals vs 15 total. Tests whether the ensemble-of-phrasings effect
  generalizes.

Core support built: multi-typed Question (decision_type, options, rubric)
with type-aware cache identity; multi-way TreeNode (branches map) with
legacy binary migration; generalized log-space routing over K-way splits;
choice/score serialization and column parsing in the Decisions client;
column storage in the cache; multi-type bank roundtrip. 106 tests green.

**Falsifiable expectations:** A's choice tree alone should beat the 17.5%
hand tree and approach majority-class (the choice question is close to a
direct classifier); its combiner should land mid-pack because 2 questions
carry limited signal. B should discover score questions for graded
dimensions (urgency/clarity) if they exist in this domain. C should add
1-3 points of macro-F1 over 0.5736 if phrasing diversity generalizes, and
do nothing if the iteration-6 effect was a one-off.

**Status:** built and unit-tested. Awaiting live run (stages 7-9).

## Iteration 7 - Suite review after first failure (2026-11-14)

**Failure:** experiment A died because the frontier returned malformed
JSON for the rubric (unterminated string - likely an unescaped quote),
and the parser had a single point of failure.

**Full-suite review and hardening (principle: degrade, never die):**

- common.py: tolerant JSON extraction (bracket matching through prose and
  fences, trailing-comma repair), complete_json with 3 retries (later
  attempts demand strict JSON) and caller-provided fallbacks.
- Experiment A: class descriptions fall back to raw class names; the
  rubric falls back to a fixed three-level clarity rubric; an unusable
  rubric no longer aborts the run.
- build_multiway_tree: examples whose choice columns come back empty fold
  into the largest branch instead of raising KeyError.
- Experiments B and C were already line-tolerant (per-candidate try/except)
  and use the retrying decision client; verified again.
- Pipeline guards re-run any experiment whose report is absent, so the
  failed stage A will retry automatically on the next bash run.

106 tests green including new parser/retry/fallback cases.

## Iteration 7 - Second failure, second lesson (2026-11-14)

**Experiment A SUCCEEDED after the hardening.** Results:

- One choice question (8 class options, frontier-written descriptions)
  as a direct 8-way tree: accuracy 38.6%, macro-F1 0.358 - beats the
  17.5% hand tree and 29.8% majority, below induced trees. The raw
  choice distribution is a fair but not great direct classifier; it
  under-predicts auto_approve (10 vs 27 gold) and misses out_of_scope
  entirely.
- The combiner on just TWO questions (choice columns + 3-level score
  columns + 4 computed = 15 features): accuracy 58.8%, macro-F1 0.5006.
  Two questions reach 87% of the 9-feature champion's macro-F1 - choice
  questions are extremely efficient feature generators.

**Experiment B failed on iteration 3's old lesson, rediscovered:** the
free-type inducer scores each question only on examples reaching its node,
but the flattened combiner needs every question x every example. The
original pipeline had a backfill stage; the experiment CLI did not.
Fixed by backfilling all tree questions across the corpus before building
features (46 missing pairs, one extra batch call).

**Status:** A complete; B and C await re-run.

## Iteration 7 - Final results (2026-11-14)

**B - free-type induction (the headline):** given the full menu, the
frontier reached for a CHOICE question at the ROOT - "What is the primary
adjustment sought regarding order items?" (gain 0.786) - then 3 nouls
beneath it. Zero score questions. That tree is the best pure tree this
project has built: accuracy 70.2% (previous best 66.7%), macro-F1 0.528,
and it has a genuine 97%-precision operating point at 10.5% coverage.
Notably its TREE beats its own flat combiner (52.6%/0.450) - with a K-way
root, the routing structure itself carries discrimination the L2-softmax
combiner cannot extract from the columns at this sample size.

**C - paraphrase ensemble: falsified.** 10 paraphrases of the 5 champion
questions added nothing (macro-F1 0.5894 vs 0.5903 originals-only,
accuracy identical). The iteration-6 "both delivery phrasings selected"
effect does not generalize. Serendipitous finding: the champion baseline
re-measured at 0.5903 macro-F1 under this run's settings - the recorded
0.5736 was slightly under-trained gradient descent, not a real difference.
True champion value is ~0.59.

**A - constrained (completed earlier):** choice columns are extremely
efficient features (2 questions -> 0.50 macro-F1) but the raw choice
distribution is a mediocre direct classifier.

**Cross-experiment verdicts:**

- Choice questions earn their place: as root (B) and as features (A).
- Score questions went unused by a free chooser and contributed little
  when constrained - nothing in this adjudication task is naturally
  ordinal. Verdict: not for this domain, revisit for graded domains.
- Tree-vs-combiner is not settled: the bank-combiner won iterations 3-6,
  but the choice-rooted tree wins in B. The obvious next synthesis is a
  choice-rooted tree whose branches feed bank-selected combiners.

## Iteration 8 - Deep soft trees: the four-part rebuild (2026-11-14)

**Motivation:** the iteration-7 free-type tree stopped at depth 2 and its
70.2% was in-sample - the structure felt inadequate, honestly judged.
Four changes (evaluated from a frontier model's suggestions, with two
corrections of our own):

1. Class-weighted entropy with SOFT exponent alpha = 0.6 (full alpha = 1
   is degenerate - proven in the iteration-3 relabel refutation).
2. Gain Ratio with BOTH guards against the C4.5 lopsided exploit:
   denominator floored at max(H_split, 0.35) AND minimum minority-branch
   effective mass of 3.0 - plus a small (K-1) complexity penalty. The
   minority guard measures RAW fractional ticket mass, not class-weighted
   mass: alpha weighting shrinks rare-class mass by design, so guarding
   weighted mass would reject exactly the rare-class splits alpha exists
   to enable (caught by a test failure during the build).
3. Fractional routing during growth (weights below 0.05 dropped): growth
   finally matches soft inference semantics. A subtle consequence surfaced
   by test: fuzzy mass (p ~ 0.9) legitimately feeds thin branches, so the
   lopsided guard bites only CONFIDENT peel-offs (p ~ 0.99, sub-cutoff
   contributions dropped) - which is the correct semantics.
4. Grow deep with relaxed brakes, then prune bottom-up - LEAKAGE-FREE:
   5-fold nested CV where growth sees only a 70% grow-split of each
   fold's training data, pruning decisions only the 30% prune-split, and
   the test fold only scores the finished tree. Pruning criterion is
   validation macro-F1 lift. A final deployment tree is grown+pruned on
   all data.

**Falsifiable expectations:** deeper trees (mean depth > 2, more internal
nodes than B's 4); honest OOF macro-F1 competitive with the 0.59 champion;
rare classes not worse than iteration 6's selected system; and the
gain-ratio guards visible in the log as rejected lopsided candidates.

**Status:** built and unit-tested (112 tests). Awaiting live run
(stage 10 of the pipeline).

## Iteration 8 - First run failure (2026-11-15)

Fold 0 crashed in pruning: growth scores each question only on the grow
split, but the pruner routes the prune split (and later, test folds)
through the tree - the path-coverage versus full-matrix gap in its third
costume. Fixed by backfilling every tree question across all examples
that any stage will route (grow split + prune split + test fold, and the
full corpus for the final deployment tree). The failed run's grow-split
scores were already cached, so the re-run reuses them.

## Iteration 8 - Results (2026-11-15): falsified, decisively

**Every pre-registered expectation failed:**

1. Trees ARE deeper (15-19 nodes, depth 3-5) - but depth did not pay.
2. OOF macro-F1 0.3154 vs the 0.59 champion. OOF accuracy 0.4912.
   Fold variance is enormous: [0.47, 0.26, 0.22, 0.28, 0.34].
3. Rare classes collapsed: escalate 0/4, politely_deny 0/5,
   auto_approve 9/27.

**Why, read from the growth logs:**

- Data starvation compounds: each fold grows on ~64 examples; fractional
  routing spreads mass further; deep nodes chase noise. "Does the account
  hold a gift card?" was picked at four different nodes - a weak computed
  feature winning by default where nothing else clears the guards.
- Pruning on a ~27-example split cannot separate signal from memorization;
  the macro-F1 lift criterion is noise at that size.
- The comparison itself is the finding: iteration 8 is the first TREE
  evaluated fully out-of-fold. Iteration 7's 70.2% was in-sample; its
  true OOF value would also be far lower. Trees-as-final-classifiers lose
  to the flat combiner at this data scale - decisively.

**The architectural conclusion (confirmation by refutation):** trees are
for FEATURE DISCOVERY, and shallow is fine for that; the combiner is the
decision surface and effectively IS the depth, learning interactions over
the whole p-vector instead of fragmenting examples down branches. This
was iteration 3's thesis; iteration 8 tested the alternative and lost.

**Salvage:** the run banked 8 new questions (7 noul + 1 choice) and its
scores are cached. The bank now holds ~37 questions; a re-selection over
the enlarged bank (after a backfill) is the cheap path to possibly beat
0.59.

## Iteration 9 - The upgrade kit without CV fragmentation (2026-11-15)

**Rationale (experimenter's call):** iteration 8's mechanisms may be good;
the nested CV was the bold move - each fold grew on ~64 examples and
starved them. Same grower, full 114-example corpus, no data-driven
pruning (only structural collapse of uniform-label subtrees), judged the
established way: tree numbers labeled in-sample, combiner CV as the
honest measure.

**Falsifiable expectations:** the tree grows deeper than B's depth-2 on
full data (mean depth > 2); the combiner over its questions lands at or
above B's 0.45, with the real test being movement toward the 0.59
champion. Rare classes should not be zero in the combiner.

**Status:** built and unit-tested (112 tests). Awaiting live run
(stage 11).

## Iteration 9 - Results (2026-11-15): mechanisms vindicated

**The experimenter's hypothesis is confirmed.** On the full corpus, the
iteration-8 kit beats experiment B on every number:

- tree (in-sample): accuracy 72.8% vs B's 70.2%, macro-F1 0.581 - the
  best tree this project has grown
- combiner CV (honest): macro-F1 0.5105 vs B's 0.4496 - a solid step,
  still below the 0.5903 bank-selection champion

**Expectation scorecard:** mean depth > 2 was FALSIFIED (mean 1.47,
max 4) - but the tree grew WIDER, not deeper: 26 nodes / 19 leaves vs
B's 10 / 6. Read with the growth log, this is the gain-ratio guards
working as designed: deep splits happen where mass supports them
(delivery-status root won by the computed seed; a 0.807 modify-intent
split at depth 2), while low-mass nodes get filler splits (the gift-card
seed was picked five times) that uniform-collapse then removes. The
combiner expectation half-passed: 0.51 > 0.45, short of 0.59.

**Standing conclusion, now triple-confirmed:** single-tree question sets
lose to bank selection. The bank is at 49 questions; the next move is
re-running greedy selection over the whole pool (offline, cache-covered -
selection admits only fully-scored questions, so partially-covered
crash-run strays drop out automatically).

## Iteration 10 - Bank re-selection over 49 questions (2026-11-15): new champion

**Result: macro-F1 0.6453 (champion was 0.5903), accuracy 0.6930, and
coverage at 97% precision 23.7% at 1.00 - more than double the previous
best head.** Every rare class healthy: escalate 3/4, modify_user 5/6,
out_of_scope 2/2, exchange 23/26.

The winning set is six semantic questions + four computed seeds (10
features): modify-attribute, both delivery-status phrasings,
address-change, pending, and the round-2 winner - "Does the customer's
request include canceling, returning, or removing an order or item" -
which came from ITERATION 9's run. The compounding thesis is fully
vindicated: iteration 9's combiner scored only 0.51 on its own, but its
question swung the selection trajectory from 0.574 to 0.645. No
individual run needs to win for the bank to gain.

Selection trajectory: 0.19 -> 0.38 -> 0.47 -> 0.52 -> 0.61 -> 0.62 ->
0.6453, stopping when nothing improved by 0.003. Temperature 0.62.
Selector scope note: choice/score questions are column-stored and outside
this noul-only selector; six multi-type questions were excluded (their
combiner scores, 0.45-0.51, trailed anyway).

Artifacts: eval/reports/selected_model_v2.json (deployment bundle),
eval/reports/iteration10_bank_selection.json.

## Iteration 11 - Cross-domain: airline and telecom (2026-11-15)

**Question:** does the whole loop transfer beyond retail, and what does
real data volume change?

**Curated corpora:**

- airline: 50 tasks (30 train / 20 test). Labels: politely_deny 19 (the
  policy refuses most requests in the reference actions - no-write
  trajectories dominate), modify_order 14, book_order 8, cancel 4,
  out_of_scope 4, escalate 1. Same dict-keyed join shape as retail.
- telecom base: 114 tasks across 8 fine-grained device actions
  (toggle_roaming 64, escalate 20, reseat_sim 10, ...).
- telecom full: 2,285 tasks across 11 actions - the data volume the
  project has been starving for - but heavily imbalanced (roaming 73%,
  several actions at 1-4 examples even at full scale).

Telecom taxonomy lesson: the coarse mapping (everything -> device_setting)
collapsed the domain to near-monoclass; the honest decision space is the
action itself, so telecom uses fine-grained labels. Also fixed a split-
filter bug (task ids live in multiple splits; membership is now set-based).

**Protocol per domain:** iterate-9 grower (full corpus, no CV
fragmentation) on a per-domain bank and cache, then greedy bank selection
(no time-budget limit on the user machine), then full selected-system
evaluation. Retail computed seeds stay in the feature matrix (constant
zero off-retail; harmless).

**Falsifiable expectations:** airline's tiny corpus (50) will fight the
grower - expect a shallow tree and noisy selection; telecom base should
behave like retail; telecom full is the real test of the data-volume
thesis - if volume helps, its selected system should show materially
better coverage-at-precision than anything at n=114, despite imbalance.

**Status:** curation done and validated offline; stages 12-14 wired.
Telecom full is the long stage (~2,285 decision batches per pass).

## Iteration 11 - Results (2026-11-15): the honest cross-domain verdict

**Mechanical transfer: YES.** One pipeline, zero code changes, three
domains curated, grown, selected, evaluated. The protocol holds.

**Competitive transfer: splits three ways.**

- AIRLINE (n=50): the grower works - combiner CV 0.4706 macro-F1 across 6
  classes, respectable at that size. But SELECTION BROKE: greedy CV over a
  50-example corpus picked a subset scoring 0.26 (politely_deny 0/19).
  Lesson: greedy selection needs a minimum-n guard, or a fallback rule -
  never ship a selected set that underperforms the grown set.
- TELECOM BASE (n=114): behaves like retail - grow combiner 0.316,
  selection slightly worse (0.296). Rare actions (1-4 examples) all zero,
  as at retail. Coverage head 56% at 1.00 precision.
- TELECOM FULL (n=2,285): the data-volume thesis CONFIRMED for the
  coverage head and not yet for the tail. Coverage at 97% precision:
  75.0% at 99.2% actual precision - three quarters of tickets
  auto-executable near-perfectly, by far the project's best triage number.
  But macro-F1 is only 0.218: 73% roaming dominance plus 1-4-example
  classes (reseat_sim 0/256 despite 256 examples - the two selected
  roaming-travel questions cannot separate it, and the single-run bank
  held only 3 questions to select from).

**Two structural lessons:**

1. Retail's 43-question bank took TEN RUNS to build; these domains got one
   run each (3-question banks). Selection has nothing to select from -
   the bank-accumulation thesis is reinforced by its absence here.
2. Coverage and macro-F1 now clearly measure different deployment values:
   telecom-full's majority classes ARE the auto-executable set (huge
   head, poor tail); retail was the opposite (balanced classes, moderate
   head). A production system reports both.

**Iteration 12 candidates:** selection guardrail (fallback to grown set
when selection underperforms), multi-seed induction per domain to build
real banks, and hierarchical handling for extreme imbalance.

## Hold-out confirmation (2026-11-15): where each number really stands

Asked the right question: are we evaluating on test data? Answer, now with
evidence:

- Combiner numbers were always out-of-fold CV (weights never scored on
  their own training folds), but question SELECTION used the full corpus,
  and tau2's own train/test split was never respected by the pipeline.

- Strict held-out run (champion 6 questions, softmax trained on the 74
  train examples ONLY, temperature from train logits only, scored on the
  40 untouched test examples): accuracy 77.5%, macro-F1 0.7555 - HIGHER
  than both the CV estimate (0.6453) and the train-fit reference (0.7206).
  The L2-regularized 10-feature model does not overfit; if anything the
  test split is slightly easier. The headline macro-F1 stands.

- The coverage head is the number that shrinks under strict protocol:
  5% at 1.00 precision on test (2 of 40) vs 45.9% on train and 23.7% CV.
  With n=40, one wrong confident ticket moves precision by 2.5 points;
  the coverage estimate needs larger test sets before it is trustworthy.

- Remaining caveat, stated plainly: the six questions were still selected
  with the test examples' CV in the loop. A fully clean protocol selects
  on the 74 train only - the one confirmation still unrun.

## Iteration 12 - Fixes 1-4 on retail and telecom full (2026-11-15)

**Fixes applied (proposed externally, evaluated honestly):**

1. RuleFit interaction features (pairwise products of semantic columns;
   tree leaf posteriors where a tree exists). Complements omitted by
   design: p_i*(1-p_j) is linearly spanned by p_i and p_i*p_j.
2. Mild class weights (N/N_c)^0.35 for FITTING only; temperature and
   gates stay on unweighted probabilities.
3. Choice/score option columns admitted to greedy selection (previously
   noul-only; multi-type questions scored by their multi-way split).
4. Semantic dedup (token-Jaccard > 0.75 vs accepted questions), class cap
   250 in proposal sampling, and a one-pass OVR targeting reseat_sim_card.

**Retail offline results (fixes 1-2, cache-only, baseline = champion):**

- baseline (raw, alpha=1): acc 0.6930 / macro-F1 0.6453 / cov 23.7% @1.00
- fix1 (interactions):     acc 0.7105 / macro-F1 0.6490 / cov 13.2% @1.00
  -> strict win on both accuracy and macro-F1.
- fix2 (mild weights):     acc 0.7018 / macro-F1 0.5441 / cov 41.2% @0.979
  -> a dial, not a fix: trades macro-F1 for a much larger coverage head.
- fix1+2:                  acc 0.7456 / macro-F1 0.6253 / cov 7.0% @1.00
  -> best accuracy; macro and head both shrink.

Lesson: interactions are a strict improvement; weight exponent is a
deployment dial (rare-class balance vs auto-execute head), and retail's
champion alpha=1 remains the macro-F1 optimum there. Telecom full - with
its 73% imbalance - is where the mild exponent should pay (stage 15).

**Status:** fixes 1-3 verified offline on retail; fix 4 + full telecom
evaluation staged as pipeline stage 15 (grow with dedup/caps, OVR pass
for reseat_sim_card, then both weight variants with interactions and
tree-leaf features).

## Iteration 12 - Results (2026-11-15): the fix package works

| telecom full | macro-F1 | accuracy | cov@97% | reseat_sim |
|---|---|---|---|---|
| iteration 11 | 0.2183 | 78.6% | 75.0% @ 99.2% | 0/256 |
| iter12, alpha=1 | 0.2755 | 82.1% | 78.2% @ 97.0% | 60/256 |
| iter12, mild   | 0.2498 | 87.1% | 75.6% @ 99.1% | 248/256 |

**reseat_sim_card moved off zero for the first time - the OVR pass plus
dedup plus interactions did exactly what they were designed to do.** The
"No Service" question is the unlock; the two near-duplicate phrasings
that slipped past the 0.75 Jaccard guard turned out to be an asset,
since their pairwise product is a strong reseat feature.

**The alpha exponent is confirmed as a deployment dial, not a fix:**
mild weights lift reseat to 248/256 but zero out grant_app_permission
and toggle_data entirely (their few-hundred examples lose to reseat's
256 under milder weighting), netting LOWER macro-F1 than alpha=1. The
alpha=1 variant wins macro-F1; mild wins accuracy and the
highest-precision head. Both beat iteration 11 on every metric.

Remaining zeros are the true tail: 1-4-example classes (airplane_mode,
wifi_calling, set_network, disconnect_vpn) - unreachable at any
weighting without more examples of them, by design of the benchmark.

All four external fixes validated: interactions (strict win, retail and
telecom), mild weights (a dial - confirmed), choice-column admission
(available; the grower chose nouls this run), dedup+cap+OVR (the reseat
unlock). Every headline number improved.

## B1 - The fair incumbent, finally run (2026-11-15)

The single-prompt frontier classifier - the comparison the plan called
for in M1 and the first question any reader asks. Design for fairness:
same inputs as the pipeline (ticket + context JSON), full label
taxonomy, temperature 0, strict-JSON output with self-rated confidence
(so the coverage comparison is possible), one call per ticket,
per-example response cache with resume. Staged for retail, airline, and
telecom full (pipeline stage 16).

Expected: somewhere between expA's choice-question 38.6% and our
systems. The decomposition thesis needs B1 to lose on macro-F1 at
minimum; if B1 wins anywhere, that result gets reported with the same
prominence.

## B1 - Harness bug caught, quarantined, fixed (2026-11-15)

The first B1 run was MEASUREMENT-BROKEN, not model-broken: 76% of telecom
responses (30-39% on retail/airline) failed parsing and were scored as
wrong-with-zero-confidence, producing an absurd 2.9% accuracy on a
73%-majority domain. Cache audit found the mechanism: thinking models
spend the 256-token output budget on reasoning and truncate mid-JSON,
and the prose-fallback regex then scavenged label words from the
reasoning text (hence phantom refuel_data answers).

Fixes: output budget 1024 (retry 512); JSON answers parse on the LAST
label occurrence (final answer, not mid-reasoning mentions); when braces
exist the prose fallback is disabled entirely - truncated JSON is a
failure, never a scavenge; raw responses now cached for audit. All
stale B1 artifacts quarantined as .broken; retail and airline B1 will
re-run under the fixed harness too - the comparison standard is
symmetric, and a broken win would be as dishonest as a rigged loss.

Lesson recorded: a baseline that scores below chance is a harness bug
report, not a result.

## B1 - Final clean results (2026-11-15): the sharing exhibit

| domain | B1 single prompt | pipeline (best) | gap |
|---|---|---|---|
| retail | 59.7% / 0.557 F1 | 69.3% / 0.645 (CV); 77.5% / 0.756 (holdout) | +10/+9; +18/+20 |
| airline | 48.0% / 0.328 | 62.0% / 0.471 | +14/+14 |
| telecom full | 9.2% / 0.023 | 87.1% / 0.250-0.276 | +78 pts |

**Three failure modes, one per domain, each the thesis in miniature:**

1. RETAIL: B1 is a serious baseline (0.557) and beats the pipeline on
   mid-frequency classes (modify_order 27/34 vs our 18/34) - but
   collapses exactly on the policy-sensitive ones the founding
   hypothesis named: auto_approve 4/27 (pipeline 19/27), politely_deny
   1/5. Decomposition's advantage concentrates precisely where
   predicted, not uniformly.
2. AIRLINE: clean across-the-board pipeline win.
3. TELECOM: zero-shot latches onto one plausible-sounding action
   (refuel_data, 2184/2285 answers) at SELF-RATED CONFIDENCE 95 - the
   correct label lives in context fields, not text sentiment. Also the
   calibration exhibit: self-confidence 95 on systematically wrong
   answers gives no usable coverage head (null at 97% precision).

Raw-response audit trail in the b1 caches. The comparison is now
symmetric, fixed-harness, and reproducible.

## B1.5 - Few-shot incumbent (2026-11-15)

Closes the one disclosed asymmetry: B1 was zero-shot while the pipeline
consumed the corpus during induction. B1.5 gives the single prompt the
same kind of labeled evidence the inducer saw - 8 stratified examples
(one per class, train split only, never the classified ticket itself,
text-only excerpts like the induction prompts). Same model, temperature
0, same parsing/caching/audit. Staged for retail (the domain where the
zero-shot gap was narrowest and the question is real). Pre-registered
expectation: few-shot closes part of the gap, especially mid-frequency
classes; the policy-sensitive classes (auto_approve, politely_deny)
remain the pipeline's edge. If B1.5 matches or beats the pipeline's CV
macro-F1, that is reported as the headline instead.

## B1.5 result + one-shot ablation (2026-11-15)

**B1.5 few-shot retail: 56.1% / 0.5432 macro-F1 - BELOW zero-shot
(59.7% / 0.5566).** Giving the incumbent the same labeled evidence did
not close the gap; auto_approve stayed 3/27. The decomposition, not
evidence access, is what wins. Three-tier retail exhibit complete:
zero-shot 0.557, few-shot 0.543, pipeline 0.645 (CV) / 0.756 (holdout).

**One-shot generation ablation (user-designed):** the frontier sees every
class with labeled examples in ONE pass and writes the whole question
set at once; those questions run through the identical downstream
machinery (decision-model scoring, interaction combiner, AND a greedy
tree restricted to the one-shot pool). Isolates exactly what the
iterative loop contributes. Pre-registered: if one-shot + combiner
matches the champion 0.645, the iterative tree machinery is unnecessary;
if it lands near B1 levels, iteration is the core value; between them,
the marginal contribution is quantified. Staged as stage 18.

## One-shot ablation - run 1 finding (2026-11-15)

First run: the model wrote ONE catch-all choice question despite "up to
12" - an elegance-over-coverage failure mode that is itself a result:
one-shot generation gravitates toward mega-questions. Its numbers
(0.242 flat / 0.315 tree) land below even B1 zero-shot (0.557) and far
below the champion (0.645): with a single broad question, decomposition
never happens. Iteration is the core value. Prompt hardened to demand
exactly 10 atomic questions; re-run to test the STRONG version of
one-shot generation.

## One-shot ablation - final (2026-11-15)

The experimenter rejected the coached re-run ("EXACTLY 10 questions"
would bias the model toward our hypothesis) - correct call. The NATURAL
one-shot result stands as the ablation: given every class at once and
"up to 12 questions," the model wrote ONE catch-all choice question
(elegance over coverage), scoring 0.242 flat / 0.315 tree - below even
B1 zero-shot (0.557) and far below the champion (0.645/0.756). Verdict:
one-shot generation does not decompose; the iterative contrastive loop
is the core value. Report rebuilt offline from cache with the natural-
prompt note.
