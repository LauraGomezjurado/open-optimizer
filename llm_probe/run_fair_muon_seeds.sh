#!/usr/bin/env bash
# Phase C: Muon at the tuned LR (from Phase A ladder) across 3 seeds, matched to
# the AdamW seed band. Pass the winning LR as $1.
set -uo pipefail
cd /home/flytekit/workspace/llm_probe
PY=/opt/micromamba/envs/runtime/bin/python3
LR=${1:?pass tuned Muon LR, e.g. 0.02}
STEPS=${2:-2000}
LOG=results/fair_muon_seeds.log
echo "=== fair Muon seeds lr=$LR steps=$STEPS ===" | tee $LOG
for S in 0 1 2; do
  echo ">>> fair_muon_s${S} lr=$LR" | tee -a $LOG
  $PY train.py --arm flat_ar --optimizer muon --lr $LR --seed $S --steps $STEPS \
    --save_ckpt --tag "fair_muon_s${S}" 2>&1 \
    | grep -E "params|step .*(0|1999)|done|checkpoint" | tail -4 | tee -a $LOG
  echo "" | tee -a $LOG
done
echo "=== PHASE C DONE ===" | tee -a $LOG
