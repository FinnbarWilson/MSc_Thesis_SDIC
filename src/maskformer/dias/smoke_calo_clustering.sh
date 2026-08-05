#!/bin/bash -l
#SBATCH --job-name=calo_smoke
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_smoke_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_smoke_%j.err

# Overfit ten events, twenty epochs, ~20 minutes. A correctness smoke test, NOT a training run and
# NOT a throughput measurement -- benchmark_calo_clustering.sh is the one that sizes the walltime.
#
#   sbatch src/maskformer/dias/smoke_calo_clustering.sh
#
# WHAT PASSING LOOKS LIKE. The loss should fall steeply and keep falling: ten events against a
# 10.4M-parameter model is a memorisation test, so a high plateau means the objective is wired
# wrong rather than undertrained. Read it straight off the checkpoint names, which carry the
# monitored value -- the Comet logger writes no metrics.csv when it is offline:
#
#   ls external/logs_smoke/<run>/ckpts/
#
# THE VALIDATION SPLIT IS POINTED AT THE TRAINING EVENTS ON PURPOSE (val_start_event 0 below).
# calo_clustering.yaml puts validation at event 20000, and inheriting that here made the monitored
# value a loss on ten UNSEEN events -- which plateaus whatever happens, and so cannot show
# memorisation at all. Job 48246 ran that way and read 7.40 -> 6.84 -> 7.23 over twenty epochs,
# which looks like a failure and is really just a generalisation curve on ten events. Same events
# on both sides is what makes the criterion above measurable.
#
# Nothing here says anything about generalisation, and none of it is a result. This answers "does
# it run, and does the loss respond to the new objective at all".
#
# Overrides: NUM_EVENTS, MAX_EPOCHS, OVERLAYS, SYNC, ALLOW_FAULTY_GPU, OUT_DIR -- same meaning as
# in train_calo_clustering.sh, which carries the rationale for everything below.

set -euo pipefail

# shellcheck disable=SC1091
. /home/xucapfwi/MSc_Thesis_SDIC/src/maskformer/dias/env.sh

OUT_DIR="${OUT_DIR:-$REPO/external/logs_smoke}"
NUM_EVENTS="${NUM_EVENTS:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"

echo "Node:   $(hostname)"
echo "Config: $CONFIG"
echo "Smoke:  $NUM_EVENTS events x $MAX_EPOCHS epochs -> $OUT_DIR"

preflight_paths
mkdir -p "$OUT_DIR"
select_gpu
sync_mirror
build_config_args
load_comet_key

cd "$EXP_DIR"

# Comet offline: a smoke test is not a result and should not sit beside the real runs in the
# project the thesis figures are traced to. The logger is still exercised, which is part of what is
# being tested -- a run that only fails once the logger goes online has not been smoke-tested.
run_in_container "$ENV_PYTHON" main.py fit \
  "${CONFIG_ARGS[@]}" \
  --trainer.default_root_dir "$OUT_DIR" \
  --trainer.logger.init_args.online false \
  --data.num_train "$NUM_EVENTS" \
  --data.num_val "$NUM_EVENTS" \
  --data.val_start_event 0 \
  --data.num_workers 4 \
  --data.pin_memory false \
  --trainer.max_epochs "$MAX_EPOCHS" \
  --trainer.limit_val_batches 2 \
  --trainer.val_check_interval 1.0 \
  --trainer.log_every_n_steps 1

echo "Done. train/loss should be falling steeply and train/p1.0_calohit_eff climbing off the floor."
echo "Neither is a result. If both look sane, submit train_calo_clustering.sh."
