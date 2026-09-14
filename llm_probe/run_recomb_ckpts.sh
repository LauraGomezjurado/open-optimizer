#!/usr/bin/env bash
# Train matched-loss AdamW vs Muon at 30M WITH checkpoints, for recombinability.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
for s in 0 1; do
  "$PY" train.py --arm flat_ar --optimizer adamw --lr 0.0003 --steps 4000 --seed $s --tag recomb --save_ckpt 2>&1 | grep -viE "Token indices|Warning" | tail -1
  "$PY" train.py --arm flat_ar --optimizer muon  --lr 0.02   --steps 4000 --seed $s --tag recomb --save_ckpt 2>&1 | grep -viE "Token indices|Warning" | tail -1
done
echo RECOMB_CKPTS_DONE
