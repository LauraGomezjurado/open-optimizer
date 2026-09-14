#!/usr/bin/env bash
# GATE 1: dense effrank-vs-loss curve for AdamW and Muon via multi-LR sweep.
# Each run logs (ar_ce, mlp_effrank, weight_norm) every 200 steps along training.
# Different LRs land at different final losses -> together they trace the curve.
# If Muon's effrank-vs-loss curve sits ABOVE AdamW's across the whole overlap AND
# it's not explained by weight_norm, the +40% is a real geometry effect.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
for lr in 0.0001 0.0003 0.0006 0.001; do
  "$PY" train.py --arm flat_ar --optimizer adamw --lr $lr --steps 3000 --seed 0 --tag g1adamw_lr$lr 2>&1 | grep -viE "Token indices|Warning" | tail -1
done
for lr in 0.005 0.01 0.02 0.04; do
  "$PY" train.py --arm flat_ar --optimizer muon --lr $lr --steps 3000 --seed 0 --tag g1muon_lr$lr 2>&1 | grep -viE "Token indices|Warning" | tail -1
done
echo GATE1_DONE
