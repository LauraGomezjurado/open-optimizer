#!/usr/bin/env bash
# PATH A retry: fixed whitened-polar (renorm + up_only) LR-sweep, seed 0, 3000 steps.
# up_only + renorm are the new defaults in WhitenedPolar. Sweep LR to find matched loss.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
for lr in 0.005 0.01 0.02 0.04; do
  "$PY" train.py --arm flat_ar --optimizer whitened --lr $lr --steps 3000 --seed 0 --tag wh2_lr$lr 2>&1 | grep -viE "Token indices|Warning" | tail -1
done
echo WHITENED2_DONE
