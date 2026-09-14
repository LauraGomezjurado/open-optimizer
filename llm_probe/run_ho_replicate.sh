#!/usr/bin/env bash
# Replicate the cumulant Pareto effect on seeds 1,2 (seed 0 showed small but
# consistent recombinability gain that effrank could not credit). Only the arms
# that matter: adamw baseline + the three cumulant lambdas. Save checkpoints.
set -uo pipefail
cd /home/flytekit/workspace/llm_probe
PY=/opt/micromamba/envs/runtime/bin/python3
STEPS=2000
for S in 1 2; do
  LOG=results/ho_rep_seed${S}.log
  echo "=== replicate seed=$S ===" | tee $LOG
  run () { tag="$1"; shift; echo ">>> $tag :: $*" | tee -a $LOG
    $PY train.py --seed $S --steps $STEPS --save_ckpt --tag "$tag" "$@" 2>&1 | tail -2 | tee -a $LOG; echo "" | tee -a $LOG; }
  run "hobase"        --arm flat_ar --optimizer adamw
  run "ho_cumul0p01"  --arm flat_ar --optimizer adamw --anticollapse cumulant --acl_lambda 0.01
  run "ho_cumul0p1"   --arm flat_ar --optimizer adamw --anticollapse cumulant --acl_lambda 0.1
  run "ho_cumul1p0"   --arm flat_ar --optimizer adamw --anticollapse cumulant --acl_lambda 1.0
  echo "=== DONE seed $S ===" | tee -a $LOG
done
echo "ALL REPLICATE DONE"
