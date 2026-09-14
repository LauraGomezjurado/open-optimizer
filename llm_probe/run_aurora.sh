#!/usr/bin/env bash
# Aurora vs Muon vs AdamW at matched loss, 30M, with checkpoints for recombinability.
# adamw/muon recomb ckpts already exist (flat_ar_seed{0,1}_{,muon_}recomb). Add aurora.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
for s in 0 1; do
  "$PY" train.py --arm flat_ar --optimizer aurora --lr 0.02 --steps 4000 --seed $s --tag recomb --save_ckpt 2>&1 | grep -viE "Token indices|Warning" | tail -1
done
echo AURORA_DONE
