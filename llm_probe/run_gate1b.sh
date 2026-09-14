#!/usr/bin/env bash
# GATE 1b: weight-norm-matched effrank check. Muon ran at 3-7x larger ||W|| than
# AdamW. To match: raise Muon's weight decay (shrinks ||W||) and/or lower AdamW's,
# sweeping wd so final ||W|| coincides, then compare effrank at matched loss AND
# matched norm. If the effrank gap survives => geometry effect; if it collapses => scale.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
# AdamW baseline (wd=0.1, as before) and Muon with escalating wd to pull ||W|| down.
"$PY" train.py --arm flat_ar --optimizer adamw --lr 0.0003 --wd 0.1  --steps 3000 --seed 0 --tag g1b_adamw 2>&1 | grep -viE "Token indices|Warning" | tail -1
for wd in 0.1 0.3 0.8 2.0; do
  "$PY" train.py --arm flat_ar --optimizer muon --lr 0.02 --wd $wd --steps 3000 --seed 0 --tag g1b_muon_wd$wd 2>&1 | grep -viE "Token indices|Warning" | tail -1
done
echo GATE1B_DONE
