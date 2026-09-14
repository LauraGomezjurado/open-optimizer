#!/usr/bin/env bash
# The direct ~124M point (d=768 L=12), matched-loss AdamW vs Muon, 4000 steps.
# Matches arXiv:2606.09658's main GPT-2 setting for a scale-credible comparison.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
"$PY" train.py --arm flat_ar --optimizer adamw --lr 0.0003 --d_model 768 --n_layer 12 --steps 4000 --seed 0 --tag big124 2>&1 | grep -viE "Token indices|Warning" | tail -1
"$PY" train.py --arm flat_ar --optimizer muon  --lr 0.02   --d_model 768 --n_layer 12 --steps 4000 --seed 0 --tag big124 2>&1 | grep -viE "Token indices|Warning" | tail -1
echo BIG124_DONE
