#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=/opt/micromamba/envs/runtime/bin/python3
"$PY" train.py --arm flat_ar --d_model 128 --tag narrow --save_ckpt --seed 0 --steps 2500 2>&1 | grep -viE "Token indices|FutureWarning|warnings.warn"
"$PY" train.py --arm flat_ar --d_model 512 --tag wide   --save_ckpt --seed 0 --steps 2500 2>&1 | grep -viE "Token indices|FutureWarning|warnings.warn"
"$PY" validate_metric.py --narrow results/ckpt/flat_ar_seed0_narrow.pt --wide results/ckpt/flat_ar_seed0_wide.pt 2>&1 | grep -viE "Token indices|unauthenticated|HF_TOKEN"
echo GATE_DONE
