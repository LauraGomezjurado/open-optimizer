#!/usr/bin/env bash
# PATH 1: whitened-polar (Sigma_x) vs Muon vs AdamW, matched-loss, 2 seeds, ckpts.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
for s in 0 1; do
  "$PY" train.py --arm flat_ar --optimizer whitened --lr 0.02 --steps 4000 --seed $s --tag wh --save_ckpt 2>&1 | grep -viE "Token indices|Warning" | tail -1
done
echo WHITENED_DONE
