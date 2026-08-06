#!/bin/bash -l
#SBATCH --job-name=calo_probe
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
# One epoch is ~2.9 h; 5 h covers startup, the dump, and a slower arm (the incidence head adds a
# dense [particles x hits] target and a kl_div term, so arm 2 is the one likely to run long).
#SBATCH --time=05:00:00
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_probe_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_probe_%j.err

# Train ONE probe arm for one epoch, then dump an event store from it, in a single job.
#
#   OVERLAY=overlay_probe_maskattn.yaml sbatch src/maskformer/dias/probe_arm.sh
#
# Train and dump live in the same job on purpose: the arms run unattended overnight, and a separate
# dump job per arm would need a dependency on a checkpoint path that does not exist until the
# training finishes. Doing both here means the only thing downstream needs is "did this job end".
#
# The 100-event window [20250, 20350) matches external/eventstore_ep000, which is the BASELINE: the
# epoch-0 checkpoint of run 48247, i.e. the same one-epoch budget under the unmodified config. Same
# events, same cuts, so the arms and the baseline are directly comparable. dias/compare_probes.py
# reads all four.
#
# WHY ONE EPOCH IS ENOUGH. Measured on run 48247's own checkpoints, cells recovered by the matched
# cluster at E > 20 GeV went 3.9 (epoch 0) -> 4.7 (epoch 1) -> 5.5 (epoch 6) against a true 38, and
# eff@0.5 went 0.122 -> 0.134 -> 0.136. The pathology is fully formed after one epoch and five more
# bought +0.002, so a longer arm cannot show anything this cannot -- and the thing being compared is
# the SHAPE of the size-vs-energy curve, not a converged number.
#
# Overrides: NUM_TRAIN, EVENTS (dump size), START, SYNC, ALLOW_FAULTY_GPU.

set -euo pipefail

# shellcheck disable=SC1091
. /home/xucapfwi/MSc_Thesis_SDIC/src/maskformer/dias/env.sh

: "${OVERLAY:?Set OVERLAY=overlay_probe_<name>.yaml}"
ARM="$(basename "$OVERLAY" .yaml)"
OUT="$REPO/external/probes/$ARM"
START="${START:-20250}"
EVENTS="${EVENTS:-100}"
WORKERS="${WORKERS:-24}"

echo "==================================================================="
echo "PROBE ARM : $ARM"
echo "Node      : $(hostname)"
echo "Started   : $(date)"
echo "Output    : $OUT"
echo "==================================================================="

preflight_paths
[ -f "$MIRROR/configs/$OVERLAY" ] || { echo "ABORT: no overlay $MIRROR/configs/$OVERLAY"; exit 1; }
mkdir -p "$OUT"
select_gpu
sync_mirror
load_comet_key

cd "$EXP_DIR"

echo
echo "--- [1/2] training one epoch ---------------------------------------"
run_in_container "$ENV_PYTHON" main.py fit \
  --config "$CONFIG" \
  --config "$MIRROR/configs/$OVERLAY" \
  --trainer.default_root_dir "$OUT" \
  --data.num_workers "$WORKERS" \
  --data.pin_memory false \
  ${NUM_TRAIN:+--data.num_train "$NUM_TRAIN"}

# The run directory is timestamped by hepattn's CLI, so it cannot be predicted -- take the newest,
# then the checkpoint with the lowest val_loss. Checkpoint names carry the monitored value, which is
# the only place it is recorded when the logger is the Comet one.
RUN_DIR="$(ls -dt "$OUT"/*/ 2>/dev/null | head -1)"
[ -n "$RUN_DIR" ] || { echo "ABORT: no run directory under $OUT"; exit 1; }
CKPT="$(ls "$RUN_DIR"ckpts/*.ckpt 2>/dev/null | sed 's/.*val_loss=//' | sort -n | head -1)"
[ -n "$CKPT" ] || { echo "ABORT: no checkpoint under $RUN_DIR"; exit 1; }
CKPT="$(ls "$RUN_DIR"ckpts/*val_loss=$CKPT.ckpt | head -1)"
echo
echo "best checkpoint: $CKPT"

echo
echo "--- [2/2] dumping $EVENTS events from [$START, $((START+EVENTS))) ---"
run_in_container "$ENV_PYTHON" -m hepattn.experiments.colliderml.eval.dump "$CKPT" \
  --start-event "$START" --num-events "$EVENTS" --out "$OUT/store" \
  --chunk-size 25 --num-workers 8 \
  --store-mask-threshold 0.02 --max-hits-per-query 0

echo
echo "ARM $ARM DONE: $(date)"
ls -d "$OUT"/store/*/ 2>/dev/null || true
