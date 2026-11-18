# Semantic Decision Tree

Decision trees for unstructured data, built on Google's decision model
(gemini-decision-1). A frontier LLM mines atomic yes/no questions offline
using tree splitting and information gain; at inference every question is
scored in ONE ~250ms batch and a calibrated combiner plus confidence
gates decide: auto-execute or escalate. Zero fine-tuning anywhere.

## Results (tau2-bench; every number reproducible from this repo, no API key needed)

| | Retail (n=114) | Telecom (n=2,285) |
|---|---|---|
| One-shot LLM question generation | 0.315 macro-F1 | - |
| Single-prompt classifier (few-shot / zero-shot) | 0.543 / 0.557 | 0.023 |
| **This pipeline** | **0.645 CV / 0.756 holdout** | **0.276 / 87.1% acc** |
| Auto-executed at high precision | 23.7% @ 100% | 75.6% @ 99.1% |

The single prompt wins easy classes but collapses exactly where policy
adjudication turns: refund grants 4/27 (pipeline: 19/27). On telecom it
answers one plausible-sounding action for 96% of tickets at self-rated
confidence 95 - the correct label lives in context fields, which is what
the induced questions learn to probe. Full tables: eval/reports/benchmark_sheet.md.

## How it works

Offline (question mining, ~$0.50/domain):
1. A frontier LLM sees tickets the tree still confuses and proposes
   yes/no (noul), multiple-choice, and rubric questions
2. Every candidate is scored by gemini-decision-1; class-weighted,
   gain-ratio-guarded information gain decides the split
3. Every question any run ever picked lands in a question bank; greedy
   cross-validated selection picks the final batch

Online (per ticket):
1. One decision-model call scores all questions (~250ms)
2. Exact facts (order status, refund count) resolve locally from JSON
3. A temperature-calibrated linear combiner over question probabilities
   and their pairwise interactions produces the class + confidence
4. Margin gate: auto-execute or escalate the ambiguous remainder

## Quickstart

Reproduce every published number offline (score caches are committed):

    python3.11 -m unittest discover -s tests
    python3.11 -m sdt.training.run_selected \
        --corpus data/curated/tau2_retail/examples.jsonl \
        --cache cache/scores.json --bank artifacts/question_bank.json \
        --state artifacts/bank_selection_v2.json \
        --out /tmp/champion.json

Full live pipeline (needs GEMINI_API_KEY; .env.example for shape):

    # clone tau2-bench per third_party/README.md, then:
    bash scripts/run_live_pipeline.sh

## Repo map

- sdt/ - the pipeline (induction, routing, combiner, experiments)
- tests/ - 113 unit tests
- ITERATIONS.md - the full research log: 18 iterations, pre-registered
  expectations, failures included
- eval/reports/ - every result artifact + benchmark_sheet.md
- cache/ - committed decision-model scores (keyless reproduction)
- artifacts/ - question banks, trees, trained models
- data/curated/ - tau2-bench-derived corpora (MIT; see data/raw manifest)

Attribution: benchmark tasks from sierra-research/tau2-bench (MIT).
Decisions API behaviors verified against the 2026-11-11 launch.
