#!/usr/bin/env bash
# Train MaskFormer on ColliderML ttbar pu200, on ce-ai-1. Replaces slurm/calo_clustering.sh.
#
#   nohup ./train_pu200.sh > ~/train_pu200.log 2>&1 &
#
# There is no walltime cap here and no queue, so unlike on DIAS the run does not have to be split
# into resumable jobs -- overlay_pu200_barrel.yaml sizes a single ~21 h schedule that completes its
# OneCycle decay in one go. Run it under nohup (or tmux) so it survives losing the ssh session.
#
# Overrides, same interface as the old slurm script:
#   NUM_TRAIN=4000 MAX_EPOCHS=2 ./train_pu200.sh
#   CKPT=logs/<run>/ckpts/last.ckpt ./train_pu200.sh     # resume
#
# STACKING OVERLAYS. The mask-head variants layer ON TOP of the pu200 barrel config, so they need
# two overlay files rather than one. OVERLAYS takes a space-separated list, applied in order, and
# later files win on conflicts:
#   OVERLAYS="overlay_pu200_barrel.yaml overlay_v1_coverage.yaml" ./train_pu200.sh
#   OVERLAYS="overlay_pu200_barrel.yaml overlay_v2_recall.yaml"      ./train_pu200.sh
#   OVERLAYS="overlay_pu200_barrel.yaml overlay_v3_propagation.yaml" ./train_pu200.sh
# OVERLAY (singular) still works and is unchanged, so nothing that used it before needs editing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

cd "$EXP_DIR"

EXTRA_ARGS=()
[ -n "${NUM_TRAIN:-}" ]  && EXTRA_ARGS+=(--data.num_train "$NUM_TRAIN")
[ -n "${MAX_EPOCHS:-}" ] && EXTRA_ARGS+=(--trainer.max_epochs "$MAX_EPOCHS")
[ -n "${CKPT:-}" ]       && EXTRA_ARGS+=(--ckpt_path "$CKPT")

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "python    : $PYTHON"
echo "exp dir   : $EXP_DIR"
echo "extra     : ${EXTRA_ARGS[*]:-(none)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "WARNING: COMET_API_KEY unset -- Comet logging will fail. See ce_ai_1/env.sh."
fi

# Preflight: refuse to start if the training window overlaps a store window. The stores dumped for
# the CLUE comparison come from events [7000,7050) and [7500,8000); training into them would make
# the head-to-head a test on training data. src/io/event_store.py asserts this at scoring time, but
# that is hours later -- fail here instead.
"$PYTHON" - "${NUM_TRAIN:-6000}" <<'PY'
import sys
n = int(sys.argv[1])
if n > 6750:
    sys.exit(f"ABORT: num_train={n} runs past the test window into the CLUE store windows "
             f"[7000,7050) and [7500,8000). Keep num_train <= 6000, or move the store windows "
             f"in configs/overlay_pu200_barrel.yaml and config/experiment.yaml together.")
print(f"window check OK: train [0,{n}) is disjoint from the store windows")
PY

# Build the --config chain. OVERLAYS wins if set; otherwise fall back to the single OVERLAY, whose
# default is unchanged, so a bare ./train_pu200.sh still runs exactly what it always did.
CONFIG_ARGS=(--config configs/calo_clustering.yaml)
for overlay in ${OVERLAYS:-${OVERLAY:-overlay_pu200_barrel.yaml}}; do
    if [ ! -f "configs/$overlay" ]; then
        echo "ABORT: no such overlay: $EXP_DIR/configs/$overlay"
        echo "       (env.sh re-syncs configs from the repository on every run -- if you just added"
        echo "        it, check it is in src/maskformer/hepattn_colliderml/configs/)"
        exit 1
    fi
    CONFIG_ARGS+=(--config "configs/$overlay")
    echo "overlay   : $overlay"
done

exec "$PYTHON" main.py fit \
    "${CONFIG_ARGS[@]}" \
    --data.pin_memory false \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
