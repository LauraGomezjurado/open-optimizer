#!/usr/bin/env bash
# Anti-collapse: does explicit richness (loss - lambda*R) buy recombinability?
# All 30M flat_ar, save ckpts (for recombinability validation). Lambda sweep.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
# baselines already exist (recomb ckpts): adamw & muon. Here: adamw+anticollapse.
for lam in 0.01 0.1; do
  "$PY" train.py --arm flat_ar --optimizer adamw --lr 0.0003 --steps 4000 --seed 0 \
    --anticollapse offdiag --acl_lambda $lam --tag acl${lam} --save_ckpt 2>&1 \
    | grep -viE "Token indices|Warning" | tail -1
done
echo ANTICOLLAPSE_DONE
