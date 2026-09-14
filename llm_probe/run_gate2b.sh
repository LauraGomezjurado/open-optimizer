#!/usr/bin/env bash
# GATE 2b: confirmation of the leverage finding with the artifact fixes:
# log_softmax probe (shape not confidence) + both seeds. Compare L-isotropy ordering.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
for s in 0 1; do
  echo "--- seed $s ---"
  $PY gate_leverage.py --k 32 --ckpts \
    results/ckpt/flat_ar_seed${s}_recomb.pt \
    results/ckpt/flat_ar_seed${s}_muon_recomb.pt \
    results/ckpt/flat_ar_seed${s}_aurora_recomb.pt \
    --out results/gate2b_seed${s}.json 2>&1 | grep -viE "Token indices|Warning"
done
echo GATE2B_DONE
