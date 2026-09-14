#!/usr/bin/env bash
# Prep the llm_probe on a GPU pod. This script COPIES code + installs deps only.
# It does NOT launch training (pod safety rule #3: launches need a fresh
# go-ahead each time). After it finishes it PRINTS the commands to run.
#
# Usage:  bash llm_probe/deploy_to_pod.sh <FULL_POD_NAME>
set -euo pipefail

POD="${1:?usage: deploy_to_pod.sh <POD_NAME>}"
NS=lila
EXPECT_EMAIL="lgomezjurado@lila.ai"

echo ">> verifying pod ownership (safety rule #1)…"
OWNER=$(kubectl get pod -n "$NS" "$POD" \
  -o jsonpath='{.metadata.annotations.lila\.ai/user-email}')
if [[ "$OWNER" != "$EXPECT_EMAIL" ]]; then
  echo "STOP: pod $POD is owned by '$OWNER', not $EXPECT_EMAIL. Wrong pod." >&2
  exit 1
fi
echo "   ok: owned by $OWNER"

echo ">> checking memory limit (safety rule #2)…"
kubectl get pod -n "$NS" "$POD" \
  -o jsonpath='{.spec.containers[0].resources.limits.memory}'; echo

DEST=/home/flytekit/workspace/llm_probe   # synced/persisted path (rule #5)
echo ">> copying llm_probe -> $POD:$DEST"
kubectl exec -n "$NS" "$POD" -- mkdir -p "$DEST"
for f in model.py factoredness.py objectives.py data.py train.py compare.py \
         optim.py anticollapse.py mechanism_depth.py gate_norm_isolation.py \
         recombinability.py DESIGN.md; do
  kubectl cp -n "$NS" "llm_probe/$f" "$POD:$DEST/$f"
done

echo ">> installing deps into the pod's runtime env"
kubectl exec -n "$NS" "$POD" -- bash -lc '
  PY=/opt/micromamba/envs/runtime/bin/python3
  $PY -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())" \
    && $PY -m pip install -q "datasets" "transformers" "numpy<2.3" || true
'

cat <<EOF

============================================================
PREP COMPLETE. Nothing has been trained.
To RUN (needs your fresh go-ahead), exec into the pod and:

  cd $DEST
  PY=/opt/micromamba/envs/runtime/bin/python3

  # 0) metric-validation gate FIRST (cheap): does the proxy separate
  #    early vs late checkpoint? (a known contrast) — inspect the per-step
  #    log in results/flat_ar.json after a short run.
  \$PY train.py --arm flat_ar --steps 300      # smoke / gate

  # 1) full arms (seeds 0,1,2 each for error bars)
  for s in 0 1 2; do
    \$PY train.py --arm flat_ar --seed \$s
    \$PY train.py --arm staged  --seed \$s
    \$PY train.py --arm mdlm    --seed \$s
  done

  # 2) headline table
  \$PY compare.py

  # 3) copy results OUT before stopping the pod (rule #4):
  #    kubectl cp -n $NS $POD:$DEST/results ./llm_probe/results
============================================================
EOF
