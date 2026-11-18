# Benchmark Sheet (follow-up post material)

All numbers from this repo's reports; every figure has a JSON artifact and
an audit trail. Pipeline = flattened question batch + calibrated combiner.

## Table 1 - The incumbent ladder (macro-F1 / accuracy)

| System | Retail (n=114) | Telecom full (n=2,285) |
|---|---|---|
| Single prompt, one-shot generation of questions | 0.315 / 43.9% | - |
| Single-prompt classifier, few-shot (8 labeled examples) | 0.543 / 56.1% | - |
| Single-prompt classifier, zero-shot | 0.557 / 59.7% | 0.023 / 9.2% |
| **This pipeline** | **0.645 CV / 0.756 holdout** | **0.276 / 82.1%** (balanced weights) |
| **This pipeline** (accuracy-tuned weights) | 0.649 / 71.1% (with interactions) | **0.250 / 87.1%** |

Notes: one-shot generation = a frontier LLM writes the whole question set
in a single pass given all classes; it produced one catch-all question
(0.24 flat). Hand-written tree baseline (agent-authored from the policy):
17.5% - included for honesty, it had no iterative refinement.

## Table 2 - Where each system fails (retail per-class recall)

| Class | Single prompt | Pipeline |
|---|---|---|
| exchange (easy, 26) | 25/26 | 23/26 |
| auto_approve (refund grants, 27) | 4/27 | **19/27** |
| politely_deny (5) | 1/5 | 2/5 |
| escalate (4) | 2/4 | 3/4 |
| modify_order (34) | **27/34** | 18/34 |

The single prompt wins the mid-frequency classes; the pipeline wins
exactly the policy-sensitive ones. Decomposition's advantage is not
uniform - it concentrates where granting vs denying turns on facts.

## Table 3 - The telecom failure mode (why 9.2%)

The zero-shot prompt answered "refuel_data" for 96% of tickets at
self-rated confidence 95. The correct label lives in context fields
(line status, device settings), not in the ticket text. Result: no
usable confidence threshold exists (coverage null at 97% precision).
The pipeline's induced questions probe those fields directly.

## Table 4 - Safe automation (coverage at high precision)

| Domain | Auto-executed | Precision | Latency |
|---|---|---|---|
| Retail (champion) | 23.7% | 100% | ~250ms |
| Telecom (accuracy-tuned) | 75.6% | 99.1% | ~250ms |

Cross-validated; escalation routes the remainder to a reasoning agent.

## Table 5 - The fix package (telecom, iteration 11 -> 12)

| | macro-F1 | accuracy | reseat_sim recall |
|---|---|---|---|
| Before (plain induction) | 0.218 | 78.6% | 0/256 |
| + dedup + OVR pass + interactions + weight dial | 0.276 | 87.1% | **248/256** |

reseat_sim_card was unlocked by one targeted question ("No Service"
status) plus its interaction with the not-traveling signal.

## Table 6 - Cost

| | Cost | Time |
|---|---|---|
| Retail champion scoring, whole corpus | ~$0.01 | ~30s |
| Telecom discovery (2 passes, 2,285 tickets) | <$0.50 | ~40 min |
| Single-prompt baseline, telecom | ~$0.85 | ~21 min (parallel) |

Per-ticket: ~250ms decision batch vs ~5s generative call.

## Provenance

Every number above comes from eval/reports/*.json in the repo, produced
by the pipeline in ITERATIONS.md (18 iterations, pre-registered
expectations, failures included). Baselines ran under a fixed harness;
a harness bug that undercounted the baseline was caught, disclosed, and
re-run before publishing (see ITERATIONS.md, B1 section).
