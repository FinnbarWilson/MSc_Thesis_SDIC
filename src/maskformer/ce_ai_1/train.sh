#!/usr/bin/env bash
# Train MaskFormer on ColliderML, on ce-ai-1.
#
#   nohup ./train.sh pu0   > ../../../external/train_pu0.log   2>&1 &
#   nohup ./train.sh pu200 > ../../../external/train_pu200.log 2>&1 &
#
# ONE SCRIPT AND ONE CONFIG PER RUN. This replaced train_pu0.sh and train_pu200.sh, and the
# overlay stack they drove, on 2026-08-12. Overlays meant the objective a run actually used
# depended on the ORDER of several --config flags, and because `tasks` is a YAML list any overlay
# touching it replaced the whole list rather than merging into it -- which is how the pu0 run of
# 2026-08-11 ended up on a different mask objective from the one its base config documented.
# configs/pu0.yaml and configs/pu200.yaml are now each self-contained.
#
# Overrides for smoke tests and resumes, unchanged:
#   NUM_TRAIN=2000 MAX_EPOCHS=1 ./train.sh pu0        # smoke
#   CKPT=logs/<run>/ckpts/last.ckpt ./train.sh pu0    # resume
#
# SIZE max_epochs BEFORE COMMITTING TO A LONG pu200 RUN. configs/pu200.yaml carries pu0's schedule
# and says so: launch, watch ~300 steps, set max_epochs from the measured rate, relaunch.
# OneCycleLR is sized from total optimiser steps and cannot be resized mid-run, so a run that
# overruns its walltime never reaches its decay phase and its final checkpoint sits at a high
# learning rate.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

DATASET="${1:-}"
case "$DATASET" in
    pu0|pu200) ;;
    *) echo "usage: $0 [pu0|pu200]   (got '${DATASET:-}')" >&2; exit 2 ;;
esac

cd "$EXP_DIR"

# Preflight, pu200 only: refuse to start if the training window overlaps a CLUE store window. The
# stores for the head-to-head come from events [7000,7050) and [7500,8000); training into them
# would make the comparison a test on training data. src/io/event_store.py asserts this at scoring
# time, but that is hours later -- fail here instead. pu0 has 100,000 events and trains on
# [0, 20000) with val/test above it, so nothing currently overlaps there.
if [ "$DATASET" = "pu200" ]; then
    "$PYTHON" - "${NUM_TRAIN:-6000}" <<'PY' || exit 2
import sys
n = int(sys.argv[1])
if n > 6750:
    sys.exit(f"ABORT: num_train={n} runs past the test window into the CLUE store windows "
             f"[7000,7050) and [7500,8000). Keep num_train <= 6000, or move the store windows "
             f"in configs/pu200.yaml and config/experiment.yaml together.")
print(f"window check OK: train [0,{n}) is disjoint from the store windows")
PY
fi

EXTRA_ARGS=()
[ -n "${NUM_TRAIN:-}" ]  && EXTRA_ARGS+=(--data.num_train "$NUM_TRAIN")
[ -n "${MAX_EPOCHS:-}" ] && EXTRA_ARGS+=(--trainer.max_epochs "$MAX_EPOCHS")
[ -n "${CKPT:-}" ]       && EXTRA_ARGS+=(--ckpt_path "$CKPT")

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "python    : $PYTHON"
echo "config    : configs/${DATASET}.yaml"
echo "extra     : ${EXTRA_ARGS[*]:-(none)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "WARNING: COMET_API_KEY unset -- Comet logging will fail. See ce_ai_1/env.sh."
fi

exec "$PYTHON" main.py fit \
    --config "configs/${DATASET}.yaml" \
    --data.pin_memory false \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
