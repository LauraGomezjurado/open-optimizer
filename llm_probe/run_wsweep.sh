#!/usr/bin/env bash
# Focused whitened-polar design sweep, LR=0.01, 2000 steps, seed 0 (screening).
# Baselines to beat: Muon ef~0.49/ce~2.71, Aurora ef~0.50/ce~2.72 at 4000 steps
# (note: 2000 steps so baselines are lower here too; we also run muon/aurora@2000 for fair ref).
cd "$(dirname "$0")"; PY=/opt/micromamba/envs/runtime/bin/python3
run(){ "$PY" train.py --arm flat_ar --steps 2000 --seed 0 --lr 0.01 "$@" 2>&1 | grep -viE "Token indices|Warning" | tail -1; }
# fair 2000-step references
run --optimizer muon   --tag s_muon
run --optimizer aurora --tag s_aurora
# alpha sweep (place=up, muon base)
run --optimizer whitened --w_alpha 0.25 --w_place up --tag s_a25
run --optimizer whitened --w_alpha 0.5  --w_place up --tag s_a50
run --optimizer whitened --w_alpha 1.0  --w_place up --tag s_a100
# placement
run --optimizer whitened --w_alpha 0.5  --w_place both --tag s_both
# phi-gap postcorr (place=up, alpha=0.5)
run --optimizer whitened --w_alpha 0.5 --w_place up --w_postcorr 0.5 --tag s_pc50
run --optimizer whitened --w_alpha 0.5 --w_place up --w_postcorr 1.0 --tag s_pc100
# aurora base + whiten
run --optimizer whitened --w_alpha 0.5 --w_place up --w_base aurora --tag s_aurw
# aurora base + whiten + postcorr (the full stack)
run --optimizer whitened --w_alpha 0.5 --w_place up --w_base aurora --w_postcorr 0.5 --tag s_full
echo WSWEEP_DONE
