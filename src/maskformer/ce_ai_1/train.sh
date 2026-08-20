#!/usr/bin/env bash
# Train MaskFormer on ColliderML, on ce-ai-1.
#
#   nohup ./train.sh pu0   > ../../../external/train_pu0.log   2>&1 &
#   nohup ./train.sh pu200 > ../../../external/train_pu200.log 2>&1 &
#
# One script and one self-contained config per run, rather than an overlay stack: `tasks` is a
# YAML list, so any overlay touching it replaced the whole list rather than merging into it, and
# the objective a run used depended on the order of its --config flags.
#
# Overrides for smoke tests and resumes:
#   NUM_TRAIN=2000 MAX_EPOCHS=1 ./train.sh pu0        # smoke
#   CKPT=logs/<run>/ckpts/last.ckpt ./train.sh pu0    # resume
#
# Size max_epochs before committing to a long run: launch, watch ~300 steps, set it from the
# measured rate, relaunch. OneCycleLR is sized from total optimiser steps and cannot be resized
# mid-run, so a run that overruns never reaches its decay phase and its final checkpoint sits at
# a high learning rate.
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
# time, but that is hours later, so fail here instead. pu0 has 100,000 events and trains on
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
# The config's data directories are this machine's. DATA_DIR points them elsewhere.
if [ -n "${DATA_DIR:-}" ]; then
    SHARDS="$DATA_DIR/ttbar_${DATASET}/"
    EXTRA_ARGS+=(--data.train_dir "$SHARDS" --data.val_dir "$SHARDS" --data.test_dir "$SHARDS")
fi

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "python    : $PYTHON"
echo "config    : configs/${DATASET}.yaml"
echo "extra     : ${EXTRA_ARGS[*]:-(none)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "WARNING: COMET_API_KEY unset; Comet logging will fail. See ce_ai_1/env.sh."
fi

exec "$PYTHON" main.py fit \
    --config "configs/${DATASET}.yaml" \
    --data.pin_memory false \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
