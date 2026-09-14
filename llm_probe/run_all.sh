#!/usr/bin/env bash
# Run all arms x seeds on the pod. Invoked as: bash run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=/opt/micromamba/envs/runtime/bin/python3
rm -f results/*.json                       # fresh
for s in 0 1 2; do
  for arm in flat_ar staged mdlm; do
    echo ">>> arm=$arm seed=$s"
    "$PY" train.py --arm "$arm" --seed "$s" 2>&1 \
      | grep -viE "Token indices|FutureWarning|warnings.warn|Repo card|Generating|Resolving data"
  done
done
echo ALL_ARMS_DONE
