#!/bin/bash -l
#SBATCH --job-name=calo_train
#SBATCH --partition=GPU
# TWO cards, and this is not a throughput request -- trainer.devices stays 1 and only one is used.
# compute-gpu-0-1 GPU 2 reported 844 uncorrected ECC errors (2026-08-05, up from 818 the week
# before) and kills jobs with "CUDA error: uncorrectable ECC error" minutes in; job 48163 died at
# 3:30. The node has three cards and only one was faulty, so any two are guaranteed to include a
# healthy one, and env.sh's select_gpu picks it out of the allocation. The job queues until two are
# free rather than gambling a day of walltime on which card it gets.
#
# RE-CHECK THAT BEFORE TRUSTING IT. The count was climbing. If a second card has since gone bad,
# two cards no longer guarantee a good one.
#SBATCH --gres=gpu:a100:2
# 32 CPUs and 256 GB, MEASURED rather than guessed -- see "RESOURCES" below. The CPU count is the
# single biggest lever on how long this takes, because the job is INPUT-bound on this cluster.
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
# The partition allows a maximum of two days. SIZE THIS FROM A MEASURED RATE, not from hope, and
# move it together with max_epochs -- see the walltime note below.
#SBATCH --time=36:00:00
# ABSOLUTE paths: SBATCH directives resolve against the submission directory, so a relative path
# aborts the job instantly unless that directory happens to contain the log dir.
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_train_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_train_%j.err

# MaskFormer calo-clustering training on DIAS, from THIS repository's config.
#
#   mkdir -p ~/MSc_Thesis_SDIC/external/slurm_logs
#   sbatch src/maskformer/dias/train.sh            # pu0, the default
#   DATASET=pu200 sbatch src/maskformer/dias/train.sh
#
# COMET_API_KEY does not need exporting first: the `-l` shebang gets a login shell, which sources
# ~/.bashrc, which exports it. Dropping the `-l` silently breaks Comet logging.
#
# ---------------------------------------------------------------------------------------------
# BEFORE THE FIRST REAL RUN, READ THIS. Three things about this cluster are different from ce-ai-1
# and each one has cost a run in the past.
#
# 1. THE JOB IS INPUT-BOUND HERE, NOT GPU-BOUND. Measured 2026-08-05 on the previous configuration,
#    two probes of the real thing at the real num_train:
#
#      allocation            rate        GPU util   peak GPU mem     host RSS   h/epoch
#      12 CPU / 12 workers   1.09 ev/s   16%        49.2 GB          131 GB     5.19
#      32 CPU / 24 workers   1.94 ev/s   30%        49.2 GB          255 GB     2.91
#
#    At 12 CPUs the A100 was idle 84% of the time waiting for the dataloader. The extra CPUs cost
#    nothing and took 9 hours off the run, and it was STILL input-bound at 30%.
#
#    THE CONSEQUENCE FOR BATCH SIZE, WHICH IS NOT INTUITIVE: on an input-bound job, raising the
#    batch does NOT raise events/second -- the dataloader is the ceiling either way -- so it simply
#    divides the optimiser step count by the batch size. This model is step-starved, so that is the
#    wrong direction. Raise --cpus-per-task and data.num_workers first, measure again, and only
#    consider a bigger batch once GPU utilisation is actually high.
#
# 2. THE CARDS. `nvidia-smi` inside a job on compute-gpu-0-1 lists all three cards whatever was
#    allocated -- there is no cgroup device isolation -- so the visible list is not the allocation.
#    env.sh's select_gpu reads Slurm's own variables instead. One card had climbing uncorrected ECC
#    errors; that is why two are requested.
#
#    CARD SIZE, CONFIRMED 2026-08-13 on the node: three x "NVIDIA A100 80GB PCIe, 81920 MiB".
#    The cluster docs saying 40 GB are out of date. Batch 4 peaks at ~34 GB, so it fits with room.
#
# 3. --mem IS NOT ENFORCED. This cluster runs TaskPlugin=task/affinity with no cgroups, so --mem is
#    used for SCHEDULING only -- job 48169 booked 128 GB, used 185 GB, and finished. Under-requesting
#    therefore "works" while over-booking the node for everyone else. The 256G above is what the job
#    measures at, not a guess with margin.
#
#    data.py caches 8 decoded row groups PER WORKER, ~1.4 GB decoded at pu0, so resident memory is
#    roughly 8 x workers x 1.4 GB -- the 255 GB measured at 24 workers. More workers means more RAM,
#    not just more CPU. Past 24, lower row_group_cache_size in the config first.
#
# ---------------------------------------------------------------------------------------------
# WALLTIME AND max_epochs MOVE TOGETHER, ALWAYS.
#
# OneCycleLR is sized from TOTAL optimiser steps, so a run that hits the wall mid-schedule never
# reaches its decay phase and its final checkpoint is taken at a high learning rate. MAX_TIME below
# stops Lightning cleanly an hour before the wall so the run ends with a written checkpoint rather
# than a SIGKILL. IT SHOULD NOT TRIGGER: if it does, the sizing was wrong.
#
#   steps    = num_train x max_epochs / batch_size
#   walltime = num_train x max_epochs / (events per second)
#
# configs/pu0.yaml is 20,000 events x 12 epochs at batch 4 = 60,000 steps. At the measured
# 1.94 ev/s that is 34.4 h, which is why --time is 36:00:00 and MAX_TIME below stops Lightning at
# 35 h. RE-MEASURE BEFORE COMMITTING: that rate came from a different configuration (1000 queries
# against this file's 400, and before the dataloader got faster), so it is likely optimistic in one
# direction and pessimistic in the other.
#
#   NUM_TRAIN=600 MAX_EPOCHS=1 sbatch src/maskformer/dias/train.sh   # ~300 steps, then read the rate
#
# ---------------------------------------------------------------------------------------------
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

REPO="${REPO:-/home/xucapfwi/MSc_Thesis_SDIC}"
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
    echo "WARNING: COMET_API_KEY unset -- Comet logging will fail. The -l shebang should have"
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

# THE DATA DIRECTORIES ARE OVERRIDDEN HERE, NOT READ FROM THE CONFIG. configs/pu0.yaml points at
# ce-ai-1's datastore (/mnt/ai-datastore/...), which does not exist on DIAS -- without this the job
# dies on the first read. Overriding rather than editing the config keeps one file valid on both
# machines, which is the same reason the ce-ai-1 overlay used to exist.
SHARDS="$DATA_DIR/ttbar_${DATASET}/"
ARGS+=(--data.train_dir "$SHARDS" --data.val_dir "$SHARDS" --data.test_dir "$SHARDS")
[ -n "${NUM_TRAIN:-}" ]  && ARGS+=(--data.num_train    "$NUM_TRAIN")
[ -n "${MAX_EPOCHS:-}" ] && ARGS+=(--trainer.max_epochs "$MAX_EPOCHS")
[ -n "${BATCH_SIZE:-}" ] && ARGS+=(--data.batch_size   "$BATCH_SIZE")
[ -n "${WORKERS:-}" ]    && ARGS+=(--data.num_workers  "$WORKERS")
ARGS+=(--trainer.max_time "${MAX_TIME:-00:35:00:00}")
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
