#!/usr/bin/env bash
# Mechanism scan: is the Muon-vs-AdamW effrank gap driven by WIDTH or DEPTH?
# For each config, train AdamW and Muon (LRs that land near matched loss) and
# record effrank. The SLOPE of gap-vs-width and gap-vs-depth is the theory signal.
# 1 seed (30M point already has 3-seed confirmation); 2500 steps for wall-clock.
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
run () {  # args: tag d_model n_layer
  local tag=$1 d=$2 L=$3
  "$PY" train.py --arm flat_ar --optimizer adamw --lr 0.0003 --d_model $d --n_layer $L --steps 2500 --seed 0 --tag scan_${tag} 2>&1 | grep -viE "Token indices|Warning" | tail -1
  "$PY" train.py --arm flat_ar --optimizer muon  --lr 0.02   --d_model $d --n_layer $L --steps 2500 --seed 0 --tag scan_${tag} 2>&1 | grep -viE "Token indices|Warning" | tail -1
  echo "== done $tag (d=$d L=$L) =="
}
# All scan configs are SMALL/FAST (d<=512, L<=12 at d=384 => ~<=40M, ~2-4 min each).
# The expensive 124M (d=768 L=12) point is run SEPARATELY (run_big124.sh) since it
# is ~50 min/optimizer; we don't want it blocking the mechanism scan.
echo "### WIDTH scan (depth fixed=6) ###"
run w256 256 6
run w384 384 6
run w512 512 6
echo "### DEPTH scan (width fixed=384) ###"
run d2 384 2
run d4 384 4
run d8 384 8
run d12 384 12
echo SCAN_DONE
