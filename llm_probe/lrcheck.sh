#!/usr/bin/env bash
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
"$PY" train.py --arm flat_ar --optimizer adamw --lr 0.0003 --steps 4000 --seed 0 2>&1 | grep -viE "Token indices|Warning" | tail -1
"$PY" train.py --arm flat_ar --optimizer muon --lr 0.02 --steps 4000 --seed 0 2>&1 | grep -viE "Token indices|Warning" | tail -1
"$PY" train.py --arm flat_ar --optimizer muon --lr 0.01 --steps 4000 --seed 0 --tag lr01 2>&1 | grep -viE "Token indices|Warning" | tail -1
echo LRCHECK_DONE
