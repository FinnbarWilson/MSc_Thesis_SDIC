#!/bin/bash -l
#SBATCH --job-name=calo_train
#SBATCH --partition=GPU
# Two cards, and this is not a throughput request: trainer.devices stays 1. One card on
# compute-gpu-0-1 kills jobs with uncorrectable ECC errors, its count still climbing, and the node
# has three, so any two include a healthy one. env.sh's select_gpu picks it out of the allocation.
# Re-check that before trusting it: if a second card has since gone bad, two no longer suffice.
#SBATCH --gres=gpu:a100:2
# Measured, not guessed. The CPU count is the biggest lever here, the job being input-bound.
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
# Size this from a measured rate and move it together with max_epochs; see below.
#SBATCH --time=24:00:00
# Absolute paths: SBATCH directives resolve against the submission directory, so a relative path
# aborts the job instantly.
# Relative to the directory sbatch was run from, so submit from the repository root.
#SBATCH --output=external/slurm_logs/calo_train_%j.out
#SBATCH --error=external/slurm_logs/calo_train_%j.err

# MaskFormer calo-clustering training on DIAS, from this repository's config.
#
#   mkdir -p ~/MSc_Thesis_SDIC/external/slurm_logs
#   sbatch src/maskformer/dias/train.sh              # pu0, the default
#   DATASET=pu200 sbatch src/maskformer/dias/train.sh
#   NUM_TRAIN=600 MAX_EPOCHS=1 sbatch .../train.sh   # probe the rate; re-run after any change
#
# The `-l` shebang gets a login shell, which sources ~/.bashrc, which exports COMET_API_KEY.
# Dropping it silently breaks Comet logging.
#
# Three things about this cluster, each of which has cost a run.
#
# 1. The job is INPUT-bound, not GPU-bound: at 32 CPUs / 24 workers the A100 still sits at 30%
#    utilisation. The unintuitive consequence is that raising the batch size does not raise
#    events/second, the dataloader being the ceiling either way, so it simply divides the
#    optimiser step count, and this model is step-starved. Raise --cpus-per-task and
#    data.num_workers first, and only consider a bigger batch once GPU utilisation is high.
#
# 2. `nvidia-smi` inside a job lists all three cards whatever was allocated, there being no cgroup
#    device isolation, so the visible list is not the allocation. env.sh's select_gpu reads
#    Slurm's own variables instead.
#
# 3. --mem does not bound resident memory here (no cgroups), but Slurm's VSizeFactor makes it a
#    hard `ulimit -v` of 1.1x, and expandable_segments reserves a large virtual range at start-up.
#    Under-requesting therefore kills CUDA context creation with "CUDA driver error: out of
#    memory" on an empty 80 GB card. data.py also caches decoded row groups per worker, so more
#    workers means more host RAM and not just more CPU.
#
# Walltime and max_epochs always move together. OneCycleLR is sized from total optimiser steps,
# so a run hitting the wall mid-schedule never reaches its decay phase and its final checkpoint is
# taken at a high learning rate. MAX_TIME stops Lightning cleanly an hour before the wall so the
# run ends with a written checkpoint rather than a SIGKILL; it should never trigger.
#
#   steps    = num_train x max_epochs / batch_size
#   walltime = num_train x max_epochs / (events per second)
#
# --time is set well above the estimate on purpose: a short probe fits inside the workers'
# row-group caches while the real run spans many more shards and does real disk reads.
# Undershooting the wall costs nothing; overshooting costs the decay phase.
#
# OVERRIDES (environment variables at submit time)
#
#   DATASET=pu200               which config to train                     (default pu0)
#   NUM_TRAIN=3000              --data.num_train
#   MAX_EPOCHS=8                --trainer.max_epochs   resize the wall to match
#   BATCH_SIZE=4                --data.batch_size      read note 1 above first
#   WORKERS=24                  --data.num_workers     raises host RSS, not just CPU use
#   MAX_TIME=00:23:00:00        --trainer.max_time     D:HH:MM:SS, Lightning's format
#   CKPT=<path>/last.ckpt       --ckpt_path            resume
#   OUT_DIR=<path>              --trainer.default_root_dir
#   DATA_DIR=<path>             where ttbar_<dataset>/ lives  (default ~/ColliderML_data)
#   SYNC=0                      skip re-copying this repo's files into the checkout
#   ALLOW_FAULTY_GPU=1          run even on a card reporting uncorrected ECC errors
set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
. "$REPO/src/maskformer/dias/env.sh"

echo "host      : $(hostname)"
echo "started   : $(date)"
echo "job       : ${SLURM_JOB_ID:-<interactive>}"
echo "dataset   : $DATASET"
echo "config    : $CONFIG"

preflight_paths || exit 1
sync_mirror     || exit 1
select_gpu      || exit 1

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "WARNING: COMET_API_KEY unset; Comet logging will fail. The -l shebang should have"
    echo "         sourced ~/.bashrc; check it exports the key."
fi

# pu200 only: refuse to start if the training window would run into the CLUE store windows at
# [7000,7050) and [7500,8000). Training into them makes the head-to-head a test on training data.
if [ "$DATASET" = "pu200" ]; then
    n="${NUM_TRAIN:-6000}"
    if [ "$n" -gt 6750 ]; then
        echo "ABORT: num_train=$n runs past the test window into the CLUE store windows."
        exit 2
    fi
fi

ARGS=(--config "$CONFIG" --data.pin_memory false --trainer.devices 1)

# The data directories are overridden here rather than read from the config. configs/pu0.yaml points at
# ce-ai-1's datastore, which does not exist on DIAS. Without this the job
# dies on the first read. Overriding rather than editing the config keeps one file valid on both
# machines, which is the same reason the ce-ai-1 overlay used to exist.
SHARDS="$DATA_DIR/ttbar_${DATASET}/"
ARGS+=(--data.train_dir "$SHARDS" --data.val_dir "$SHARDS" --data.test_dir "$SHARDS")
[ -n "${NUM_TRAIN:-}" ]  && ARGS+=(--data.num_train    "$NUM_TRAIN")
[ -n "${MAX_EPOCHS:-}" ] && ARGS+=(--trainer.max_epochs "$MAX_EPOCHS")
[ -n "${BATCH_SIZE:-}" ] && ARGS+=(--data.batch_size   "$BATCH_SIZE")
[ -n "${WORKERS:-}" ]    && ARGS+=(--data.num_workers  "$WORKERS")
# Moves with SBATCH --time above, always: one hour under the wall, so Lightning stops cleanly with a
# written checkpoint rather than taking a SIGKILL mid-schedule. It should never trigger.
ARGS+=(--trainer.max_time "${MAX_TIME:-00:23:00:00}")
[ -n "${CKPT:-}" ]       && ARGS+=(--ckpt_path         "$CKPT")
[ -n "${OUT_DIR:-}" ]    && ARGS+=(--trainer.default_root_dir "$OUT_DIR")

echo "args      : ${ARGS[*]}"
echo

# The container is the reason this script exists at all: DIAS is RHEL7 (glibc 2.17) and hepattn
# needs 2.28+. --nv passes the driver through; the binds are the dataset and this repository.
exec apptainer exec --nv \
    --bind "$DATA_DIR" \
    --bind "$REPO" \
    --bind "$HEPATTN" \
    "$SIF" \
    "$ENV_PYTHON" "$EXP_DIR/main.py" fit "${ARGS[@]}"
