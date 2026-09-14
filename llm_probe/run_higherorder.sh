#!/usr/bin/env bash
# Higher-order independence lever vs established baselines.
# Question: does negentropy/cumulant (independence, past decorrelation) raise the
# FER axis (effrank) at matched capability (ar_ce) — and beyond Muon's spectral bias?
# Anti-circularity: R applied at mid layer (acl_layer=n_layer//2); we ALSO save
# checkpoints so recombinability (a DIFFERENT property) can validate later.
#
# NOTE: every config gets a UNIQUE --tag so output JSONs never collide.
set -uo pipefail
cd /home/flytekit/workspace/llm_probe
PY=/opt/micromamba/envs/runtime/bin/python3
S=${1:-0}
STEPS=${2:-2000}
LOG=results/ho_run_seed${S}.log
echo "=== higher-order sweep seed=$S steps=$STEPS ===" | tee $LOG

run () {  # $1 = unique tag, rest = args
  tag="$1"; shift
  echo ">>> $tag :: $*" | tee -a $LOG
  $PY train.py --seed $S --steps $STEPS --save_ckpt --tag "$tag" "$@" 2>&1 | tail -3 | tee -a $LOG
  echo "" | tee -a $LOG
}

# references (distinct optimizer keeps files separate; tag also unique)
run "hobase"           --arm flat_ar --optimizer adamw
run "hobase"           --arm flat_ar --optimizer muon
# higher-order lever on AdamW, lambda sweep — UNIQUE tag per config
run "ho_negent0p01"    --arm flat_ar --optimizer adamw --anticollapse negentropy --acl_lambda 0.01
run "ho_negent0p1"     --arm flat_ar --optimizer adamw --anticollapse negentropy --acl_lambda 0.1
run "ho_negent1p0"     --arm flat_ar --optimizer adamw --anticollapse negentropy --acl_lambda 1.0
run "ho_cumul0p01"     --arm flat_ar --optimizer adamw --anticollapse cumulant   --acl_lambda 0.01
run "ho_cumul0p1"      --arm flat_ar --optimizer adamw --anticollapse cumulant   --acl_lambda 0.1
run "ho_cumul1p0"      --arm flat_ar --optimizer adamw --anticollapse cumulant   --acl_lambda 1.0
# stack on Muon: does independence add on top of the spectral bias?
run "ho_negent0p1"     --arm flat_ar --optimizer muon  --anticollapse negentropy --acl_lambda 0.1
run "ho_cumul0p1"      --arm flat_ar --optimizer muon  --anticollapse cumulant   --acl_lambda 0.1
echo "=== DONE seed $S ===" | tee -a $LOG
