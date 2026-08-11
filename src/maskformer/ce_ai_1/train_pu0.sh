#!/usr/bin/env bash
# Train MaskFormer on ColliderML ttbar pu0, on ce-ai-1.
#
#   nohup ./train_pu0.sh > ../../../external/train_pu0_dice.log 2>&1 &
#
# The pu0 counterpart of train_pu200.sh, and deliberately a separate file rather than a flag on
# it: that script carries a pu200 preflight (a window check against the CLUE store windows at
# [7000,7050) and [7500,8000)) whose event numbers mean nothing here, and defaults to the barrel
# overlay. Sharing them would mean one script whose every line needs an "if pu0" caveat.
#
# WHAT THIS RUNS BY DEFAULT
#
#   configs/calo_clustering.yaml   the pu0 experiment, unchanged
#   configs/overlay_pu0_dice.yaml  the ce-ai-1 data paths, and the dice-dominant mask objective
#
# Override with OVERLAYS to run something else, same interface as train_pu200.sh:
#   OVERLAYS="overlay_pu0_dice.yaml overlay_long_schedule.yaml" ./train_pu0.sh
#   NUM_TRAIN=2000 MAX_EPOCHS=1 ./train_pu0.sh        # smoke
#   CKPT=logs/<run>/ckpts/last.ckpt ./train_pu0.sh    # resume
#
# NO WINDOW PREFLIGHT HERE, and that is a gap rather than a decision. pu200 has one because its
# CLUE store windows are carved out of the same 10,000 downloaded events. pu0 has 100,000 events
# on disk and calo_clustering.yaml trains on [0, 20000) with val/test above it, so nothing
# currently overlaps -- but if pu0 stores are ever dumped, add the same check that
# train_pu200.sh has before trusting this.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

cd "$EXP_DIR"

CONFIG_ARGS=(--config configs/calo_clustering.yaml)
for overlay in ${OVERLAYS:-overlay_pu0_dice.yaml}; do
    CONFIG_ARGS+=(--config "configs/${overlay}")
done

EXTRA_ARGS=()
[ -n "${NUM_TRAIN:-}" ]  && EXTRA_ARGS+=(--data.num_train "$NUM_TRAIN")
[ -n "${MAX_EPOCHS:-}" ] && EXTRA_ARGS+=(--trainer.max_epochs "$MAX_EPOCHS")
[ -n "${CKPT:-}" ]       && EXTRA_ARGS+=(--ckpt_path "$CKPT")

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "python    : $PYTHON"
echo "configs   : ${CONFIG_ARGS[*]}"
echo "extra     : ${EXTRA_ARGS[*]:-(none)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "WARNING: COMET_API_KEY unset -- Comet logging will fail. See ce_ai_1/env.sh."
fi

exec "$PYTHON" main.py fit \
    "${CONFIG_ARGS[@]}" \
    --data.pin_memory false \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
