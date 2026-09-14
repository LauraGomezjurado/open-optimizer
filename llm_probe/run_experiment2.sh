#!/usr/bin/env bash
# Experiment 2 (both next steps at once), run on the pod:
#   (A) STRONGER objective contrast: add staged_strong arm; train all 4 arms x 3
#       seeds, saving checkpoints.
#   (B) SAE GATE: train an SAE on the residual stream of the matched-loss arms
#       (flat_ar, staged, staged_strong — the causal-native ones) and compare
#       monosemanticity. mdlm excluded from SAE (different task, ar_ce huge).
set -uo pipefail
cd "$(dirname "$0")"
PY=/opt/micromamba/envs/runtime/bin/python3

echo "=========== (A) training arms w/ checkpoints ==========="
rm -f results/*.json
for s in 0 1 2; do
  for arm in flat_ar staged staged_strong mdlm; do
    echo ">>> arm=$arm seed=$s"
    "$PY" train.py --arm "$arm" --seed "$s" --save_ckpt 2>&1 \
      | grep -viE "Token indices|FutureWarning|warnings.warn|Repo card|Generating|Resolving data"
  done
done
echo "ARMS_DONE"

echo "=========== (B) SAE gate on matched-loss arms (seed 0) ==========="
for arm in flat_ar staged staged_strong; do
  "$PY" sae.py --ckpt results/ckpt/${arm}_seed0.pt --layer 4 2>&1 \
    | grep -viE "Token indices|FutureWarning|warnings.warn"
done
echo "SAE_DONE"
echo "ALL_DONE"
