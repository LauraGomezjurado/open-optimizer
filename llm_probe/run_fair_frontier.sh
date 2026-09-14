#!/usr/bin/env bash
# Fair Muon-vs-AdamW frontier: the comparison the "+40% recombinability" headline
# actually needs. Fixes vs the old sweep:
#   1. Muon at its OWN tuned LR (the old sweep ran Muon at AdamW's 3e-4 = crippled).
#      We first ladder Muon LR on seed 0, then run the winner at 3 seeds.
#   2. Multi-seed BOTH optimizers (matched-loss comparison needs a seed band).
#   3. Save every checkpoint so neutral_frontier.py can judge coherence externally.
# AdamW seed 0/1/2 @ lr3e-4 already exist as flat_ar_seed{0,1,2}_hobase.pt; we
# re-emit them here under fair_* tags so all comparison ckpts share one naming.
set -uo pipefail
cd /home/flytekit/workspace/llm_probe
PY=/opt/micromamba/envs/runtime/bin/python3
STEPS=${1:-2000}
LOG=results/fair_frontier.log
echo "=== fair frontier steps=$STEPS ===" | tee $LOG

run () { tag="$1"; shift; echo ">>> $tag :: $*" | tee -a $LOG
  $PY train.py --arm flat_ar --steps $STEPS --save_ckpt --tag "$tag" "$@" 2>&1 \
    | grep -E "params|step .*(0|1999)|done|checkpoint" | tail -4 | tee -a $LOG
  echo "" | tee -a $LOG; }

# --- Phase A: Muon LR ladder on seed 0 (pick tuned LR by final ar_ce) ---
for LR in 0.005 0.01 0.02 0.04; do
  run "fair_muon_lr${LR}" --optimizer muon --lr $LR --seed 0
done

# --- Phase B: AdamW reference at 3 seeds (lr 3e-4, its tuned value) ---
for S in 0 1 2; do
  run "fair_adamw_s${S}" --optimizer adamw --lr 3e-4 --seed $S
done
echo "=== PHASE A+B DONE. Inspect ladder, then run run_fair_muon_seeds.sh <LR> ===" | tee -a $LOG
