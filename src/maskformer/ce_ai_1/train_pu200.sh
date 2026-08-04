#!/usr/bin/env bash
# Train MaskFormer on ColliderML ttbar pu200, on ce-ai-1. Replaces slurm/calo_clustering.sh.
#
#   nohup ./train_pu200.sh > ~/train_pu200.log 2>&1 &
#
# There is no walltime cap here and no queue, so unlike on DIAS the run does not have to be split
# into resumable jobs -- overlay_pu200.yaml sizes a single ~21 h schedule that completes its
# OneCycle decay in one go. Run it under nohup (or tmux) so it survives losing the ssh session.
#
# Overrides, same interface as the old slurm script:
#   NUM_TRAIN=4000 MAX_EPOCHS=2 ./train_pu200.sh
#   CKPT=logs/<run>/ckpts/last.ckpt ./train_pu200.sh     # resume
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
             f"in configs/overlay_pu200.yaml and config/experiment.yaml together.")
print(f"window check OK: train [0,{n}) is disjoint from the store windows")
PY

exec "$PYTHON" main.py fit \
    --config configs/calo_clustering.yaml \
    --config configs/overlay_pu200.yaml \
    --data.pin_memory false \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
