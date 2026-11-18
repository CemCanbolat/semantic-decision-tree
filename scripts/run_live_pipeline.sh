#!/usr/bin/env bash
# One-shot live pipeline: smoke -> live baseline -> live induction.
# Idempotent: completed stages are skipped on re-run.
# Progress appends to eval/reports/live_progress.log (shared with the agent sandbox).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=python3.11
command -v python3.11 >/dev/null 2>&1 || PY=python3

LOG=eval/reports/live_progress.log
mkdir -p eval/reports artifacts/trees cache
# show progress live in the terminal AND append to the shared log
exec > >(tee -a "$LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "[$(ts)] live pipeline started (pid $$)"
# pick up the key from .env if this shell lacks it
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f .env ]; then
  set -a; . ./.env; set +a
  echo "[$(ts)] key loaded from .env"
fi
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "[$(ts)] ERROR: GEMINI_API_KEY is not set in this shell"
  exit 1
fi
echo "[$(ts)] key present (length not logged)"

if [ ! -f eval/reports/m1_tau2_live.json ]; then
  echo "[$(ts)] stage 1/2: smoke test"
  "$PY" -m sdt.eval.smoke_gemini --send || { echo "[$(ts)] smoke failed"; exit 1; }
  echo "[$(ts)] stage 1/2: live baseline (114 tickets)"
  "$PY" -m sdt.eval.run_baselines --scorer gemini \
    --corpus data/curated/tau2_retail/examples.jsonl \
    --out eval/reports/m1_tau2_live.json || { echo "[$(ts)] baseline failed"; exit 1; }
  echo "[$(ts)] stage 1/2: baseline report written"
else
  echo "[$(ts)] stage 1/2: baseline report already present, skipping"
fi

if [ ! -f artifacts/trees/induced_v3.json ]; then
  echo "[$(ts)] stage 2/2: live induction"
  "$PY" -m sdt.induction.run_induction \
    --corpus data/curated/tau2_retail/examples.jsonl \
    --frontier gemini --scorer gemini \
    --out artifacts/trees/induced_v3.json || { echo "[$(ts)] induction failed"; exit 1; }
  echo "[$(ts)] stage 2/2: induced tree written"
else
  echo "[$(ts)] stage 2/2: induced tree already present, skipping"
fi

# guard on CONTENT, not existence: an offline partial report must not skip the stage
if ! grep -q "best_feature_set" eval/reports/iteration3_combiner.json 2>/dev/null; then
  echo "[$(ts)] stage 3/3: cache backfill + combiner"
  "$PY" -m sdt.training.backfill_cache     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json     --trees artifacts/trees/induced_v3.json artifacts/trees/induced_v1.json     || { echo "[$(ts)] backfill failed"; exit 1; }
  "$PY" -m sdt.training.run_combiner     --corpus data/curated/tau2_retail/examples.jsonl     --tree artifacts/trees/induced_v3.json     --extra-tree artifacts/trees/induced_v1.json     --cache cache/scores.json     --out eval/reports/iteration3_combiner.json     || { echo "[$(ts)] combiner failed"; exit 1; }
  echo "[$(ts)] stage 3/3: combiner report written"
else
  echo "[$(ts)] stage 3/3: combiner report already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/iteration4_criteria.json 2>/dev/null; then
  echo "[$(ts)] stage 4/4: criteria pass (generate, re-score, A/B combiner)"
  "$PY" -m sdt.induction.add_criteria     --tree artifacts/trees/induced_v3.json     --out artifacts/trees/induced_v3_criteria.json     --frontier gemini     || { echo "[$(ts)] criteria generation failed"; exit 1; }
  "$PY" -m sdt.training.backfill_cache     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json     --trees artifacts/trees/induced_v3_criteria.json     || { echo "[$(ts)] criteria backfill failed"; exit 1; }
  "$PY" -m sdt.training.run_combiner     --corpus data/curated/tau2_retail/examples.jsonl     --tree artifacts/trees/induced_v3_criteria.json     --cache cache/scores.json     --out eval/reports/iteration4_criteria.json     || { echo "[$(ts)] criteria combiner failed"; exit 1; }
  echo "[$(ts)] stage 4/4: criteria report written"
else
  echo "[$(ts)] stage 4/4: criteria report already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/iteration5_fresh.json 2>/dev/null; then
  echo "[$(ts)] stage 5/5: fresh induction v4 (computed seeds in pool) + backfill + combiner"
  "$PY" -m sdt.induction.run_induction     --corpus data/curated/tau2_retail/examples.jsonl     --frontier gemini --scorer gemini     --out artifacts/trees/induced_v4.json     --cache cache/scores.json     || { echo "[$(ts)] v4 induction failed"; exit 1; }
  "$PY" -m sdt.training.backfill_cache     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json     --trees artifacts/trees/induced_v4.json     || { echo "[$(ts)] v4 backfill failed"; exit 1; }
  "$PY" -m sdt.training.run_combiner     --corpus data/curated/tau2_retail/examples.jsonl     --tree artifacts/trees/induced_v4.json     --cache cache/scores.json     --out eval/reports/iteration5_fresh.json     || { echo "[$(ts)] v4 combiner failed"; exit 1; }
  echo "[$(ts)] stage 5/5: fresh-run report written"
else
  echo "[$(ts)] stage 5/5: fresh-run report already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/iteration6_bank.json 2>/dev/null; then
  echo "[$(ts)] stage 6/6: multi-seed induction (3 trees) + bank combiner"
  "$PY" -m sdt.induction.run_multi     --corpus data/curated/tau2_retail/examples.jsonl     --n-trees 3 --frontier gemini --scorer gemini     --cache cache/scores.json --bank artifacts/question_bank.json     || { echo "[$(ts)] multi-seed induction failed"; exit 1; }
  "$PY" -m sdt.training.run_bank     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json --bank artifacts/question_bank.json     --out eval/reports/iteration6_bank.json --backfill     || { echo "[$(ts)] bank combiner failed"; exit 1; }
  echo "[$(ts)] stage 6/6: bank report written"
else
  echo "[$(ts)] stage 6/6: bank report already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/expA_choice_score.json 2>/dev/null; then
  echo "[$(ts)] stage 7/9: experiment A - one choice + one score, non-binary tree"
  "$PY" -m sdt.experiments.exp_a_constrained     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json --bank artifacts/question_bank.json     --out eval/reports/expA_choice_score.json     || { echo "[$(ts)] experiment A failed"; exit 1; }
else
  echo "[$(ts)] stage 7/9: experiment A already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/expB_free_types.json 2>/dev/null; then
  echo "[$(ts)] stage 8/9: experiment B - free-type induction (noul/choice/score)"
  "$PY" -m sdt.experiments.exp_b_free     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json --bank artifacts/question_bank.json     --out eval/reports/expB_free_types.json     || { echo "[$(ts)] experiment B failed"; exit 1; }
else
  echo "[$(ts)] stage 8/9: experiment B already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/expC_paraphrase.json 2>/dev/null; then
  echo "[$(ts)] stage 9/9: experiment C - paraphrase ensemble"
  "$PY" -m sdt.experiments.exp_c_paraphrase     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json     --selection artifacts/bank_selection.json     --out eval/reports/expC_paraphrase.json     || { echo "[$(ts)] experiment C failed"; exit 1; }
else
  echo "[$(ts)] stage 9/9: experiment C already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/iteration8_deep_soft.json 2>/dev/null; then
  echo "[$(ts)] stage 10/10: iteration 8 - deep soft trees, gain-ratio guarded, leakage-free pruning"
  "$PY" -m sdt.experiments.run_iteration8     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json --bank artifacts/question_bank.json     --out eval/reports/iteration8_deep_soft.json     || { echo "[$(ts)] iteration 8 failed"; exit 1; }
else
  echo "[$(ts)] stage 10/10: iteration 8 already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/iteration9_full_corpus.json 2>/dev/null; then
  echo "[$(ts)] stage 11/11: iteration 9 - full upgrade kit on the full corpus, no CV fragmentation"
  "$PY" -m sdt.experiments.run_iteration9     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json --bank artifacts/question_bank.json     --out eval/reports/iteration9_full_corpus.json     || { echo "[$(ts)] iteration 9 failed"; exit 1; }
else
  echo "[$(ts)] stage 11/11: iteration 9 already present, skipping"
fi

if ! grep -q "cv_macro_f1" eval/reports/crossdomain_airline.json 2>/dev/null; then
  echo "[$(ts)] stage 12/14: airline - grow + bank + selection"
  mkdir -p artifacts/trees
  "$PY" -m sdt.data.tau2_curate --tau2-dir tau2_clone/tau2-bench/data/tau2/domains/airline --out-dir data/curated/tau2_airline/ --domain airline 
  "$PY" -m sdt.experiments.run_iteration9 --corpus data/curated/tau2_airline/examples.jsonl --cache cache/scores_airline.json --bank artifacts/question_bank_airline.json --out eval/reports/crossdomain_airline_grow.json     || { echo "[$(ts)] airline grow failed"; exit 1; }
  rm -f artifacts/bank_selection_airline.json
  "$PY" -m sdt.training.select_bank --corpus data/curated/tau2_airline/examples.jsonl --cache cache/scores_airline.json --bank artifacts/question_bank_airline.json --state artifacts/bank_selection_airline.json --time-budget 3600     || { echo "[$(ts)] airline selection failed"; exit 1; }
  "$PY" -m sdt.training.run_selected --corpus data/curated/tau2_airline/examples.jsonl --cache cache/scores_airline.json --bank artifacts/question_bank_airline.json --state artifacts/bank_selection_airline.json --out eval/reports/crossdomain_airline.json     || { echo "[$(ts)] airline eval failed"; exit 1; }
else
  echo "[$(ts)] stage 12/14: airline already present, skipping"
fi

if ! grep -q "cv_macro_f1" eval/reports/crossdomain_telecom_base.json 2>/dev/null; then
  echo "[$(ts)] stage 13/14: telecom_base - grow + bank + selection"
  mkdir -p artifacts/trees
  "$PY" -m sdt.data.tau2_curate --tau2-dir tau2_clone/tau2-bench/data/tau2/domains/telecom --out-dir data/curated/tau2_telecom_base/ --domain telecom --splits base 
  "$PY" -m sdt.experiments.run_iteration9 --corpus data/curated/tau2_telecom_base/examples.jsonl --cache cache/scores_telecom.json --bank artifacts/question_bank_telecom.json --out eval/reports/crossdomain_telecom_base_grow.json     || { echo "[$(ts)] telecom_base grow failed"; exit 1; }
  rm -f artifacts/bank_selection_telecom.json
  "$PY" -m sdt.training.select_bank --corpus data/curated/tau2_telecom_base/examples.jsonl --cache cache/scores_telecom.json --bank artifacts/question_bank_telecom.json --state artifacts/bank_selection_telecom.json --time-budget 3600     || { echo "[$(ts)] telecom_base selection failed"; exit 1; }
  "$PY" -m sdt.training.run_selected --corpus data/curated/tau2_telecom_base/examples.jsonl --cache cache/scores_telecom.json --bank artifacts/question_bank_telecom.json --state artifacts/bank_selection_telecom.json --out eval/reports/crossdomain_telecom_base.json     || { echo "[$(ts)] telecom_base eval failed"; exit 1; }
else
  echo "[$(ts)] stage 13/14: telecom_base already present, skipping"
fi

if ! grep -q "cv_macro_f1" eval/reports/crossdomain_telecom_full.json 2>/dev/null; then
  echo "[$(ts)] stage 14/14: telecom_full - grow + bank + selection"
  mkdir -p artifacts/trees
  "$PY" -m sdt.data.tau2_curate --tau2-dir tau2_clone/tau2-bench/data/tau2/domains/telecom --out-dir data/curated/tau2_telecom_full/ --domain telecom --splits full 
  "$PY" -m sdt.experiments.run_iteration9 --corpus data/curated/tau2_telecom_full/examples.jsonl --cache cache/scores_telecom_full.json --bank artifacts/question_bank_telecom_full.json --out eval/reports/crossdomain_telecom_full_grow.json     || { echo "[$(ts)] telecom_full grow failed"; exit 1; }
  rm -f artifacts/bank_selection_telecom_full.json
  "$PY" -m sdt.training.select_bank --corpus data/curated/tau2_telecom_full/examples.jsonl --cache cache/scores_telecom_full.json --bank artifacts/question_bank_telecom_full.json --state artifacts/bank_selection_telecom_full.json --time-budget 3600     || { echo "[$(ts)] telecom_full selection failed"; exit 1; }
  "$PY" -m sdt.training.run_selected --corpus data/curated/tau2_telecom_full/examples.jsonl --cache cache/scores_telecom_full.json --bank artifacts/question_bank_telecom_full.json --state artifacts/bank_selection_telecom_full.json --out eval/reports/crossdomain_telecom_full.json     || { echo "[$(ts)] telecom_full eval failed"; exit 1; }
else
  echo "[$(ts)] stage 14/14: telecom_full already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/iteration12_telecom_full.json 2>/dev/null; then
  echo "[$(ts)] stage 15/15: iteration 12 on telecom full - dedup, caps, OVR, interactions, weight variants"
  "$PY" -m sdt.experiments.run_iteration12     --corpus data/curated/tau2_telecom_full/examples.jsonl     --cache cache/scores_telecom_full.json     --bank artifacts/question_bank_telecom_full.json     --tree artifacts/trees/induced_iter9.json     --ovr-target reseat_sim_card     --out eval/reports/iteration12_telecom_full.json     || { echo "[$(ts)] iteration 12 failed"; exit 1; }
else
  echo "[$(ts)] stage 15/15: iteration 12 already present, skipping"
fi

for DOM in retail airline telecom_full; do
  case "$DOM" in
    retail) CORPUS=data/curated/tau2_retail/examples.jsonl ;;
    airline) CORPUS=data/curated/tau2_airline/examples.jsonl ;;
    telecom_full) CORPUS=data/curated/tau2_telecom_full/examples.jsonl ;;
  esac
  if ! grep -q "accuracy" "eval/reports/b1_${DOM}.json" 2>/dev/null; then
    echo "[$(ts)] stage 16: B1 single-prompt baseline - ${DOM}"
    "$PY" -m sdt.eval.run_b1 --corpus "$CORPUS" --out "eval/reports/b1_${DOM}.json" \
      || { echo "[$(ts)] B1 ${DOM} failed"; exit 1; }
  else
    echo "[$(ts)] stage 16: B1 ${DOM} already present, skipping"
  fi
done

if ! grep -q "accuracy" eval/reports/b1_fewshot_retail.json 2>/dev/null; then
  echo "[$(ts)] stage 17: B1.5 few-shot baseline (8 stratified train shots) - retail"
  "$PY" -m sdt.eval.run_b1     --corpus data/curated/tau2_retail/examples.jsonl     --out eval/reports/b1_fewshot_retail.json     --cache eval/reports/b1_fewshot_retail_cache.json     --few-shot 8     || { echo "[$(ts)] B1.5 failed"; exit 1; }
else
  echo "[$(ts)] stage 17: B1.5 already present, skipping"
fi

if ! grep -q "best_feature_set" eval/reports/exp_oneshot_retail.json 2>/dev/null; then
  echo "[$(ts)] stage 18: one-shot generation ablation - does iterative induction add value?"
  "$PY" -m sdt.experiments.exp_oneshot     --corpus data/curated/tau2_retail/examples.jsonl     --cache cache/scores.json     --bank artifacts/question_bank.json     --out eval/reports/exp_oneshot_retail.json     || { echo "[$(ts)] oneshot failed"; exit 1; }
else
  echo "[$(ts)] stage 18: oneshot already present, skipping"
fi

echo "[$(ts)] live pipeline COMPLETE"
