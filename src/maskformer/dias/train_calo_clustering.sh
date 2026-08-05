#!/bin/bash -l
#SBATCH --job-name=calo_train
#SBATCH --partition=GPU
# Two cards, and this is not a throughput request -- trainer.devices stays 1 and only one is used.
# compute-gpu-0-1 GPU 2 has 844 uncorrected ECC errors (2026-08-05, up from 818 the week before)
# and kills jobs with "CUDA error: uncorrectable ECC error" minutes in; job 48163 died at 3:30. The
# node has three cards and only one is faulty, so any two are guaranteed to include a healthy one,
# and env.sh's select_gpu picks it out of the allocation. The job queues until two are free rather
# than gambling a day of walltime on which card it gets.
#SBATCH --gres=gpu:a100:2
# 32 and 256G, not the 12 and 128G the mirrored script asks for. This run is INPUT-BOUND, and the
# CPU allocation is the single biggest lever on how long it takes -- measured, see "RESOURCES"
# below. The memory is not slack: 24 dataloader workers really do reach ~255 GB resident.
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
# 26 h against a measured 20.4 (7 epochs x 2.91 h). This is a CAP, not the expected duration --
# the job ends when the schedule does. Re-derive it with benchmark_calo_clustering.sh whenever the
# per-step cost or max_epochs changes; the two must move together.
#SBATCH --time=26:00:00
# Absolute paths: SBATCH directives resolve against the submission directory, so a relative path
# aborts the job instantly unless that directory happens to contain the log dir. These point inside
# this repository (external/ is gitignored), not into the hepattn checkout.
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_train_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_train_%j.err

# MaskFormer calo-clustering training on ColliderML pu0, on DIAS, from THIS repository's config.
#
#   mkdir -p /home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs
#   sbatch src/maskformer/dias/train_calo_clustering.sh
#
# COMET_API_KEY does not need exporting first: the `-l` shebang gets a login shell, which sources
# ~/.bashrc, which exports it. Dropping the `-l` would break Comet logging. See env.sh.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS EXISTS, GIVEN hepattn_colliderml/slurm/calo_clustering.sh ALREADY DOES SOMETHING SIMILAR
#
# That file is part of the verbatim mirror -- verify_sync.sh checks it byte-for-byte against the
# hepattn checkout, so it cannot be edited here without turning the mirror into a fork. It is also
# sized and pathed for the configuration as it stood when the reported pu0 checkpoint was trained,
# and the config has since moved. This script is mine, so it lives beside ce_ai_1/ for the reason
# ce_ai_1/env.sh gives: launchers that are not upstream's do not belong in a directory whose whole
# point is that `ls` answers "which files must stay identical to upstream".
#
# ---------------------------------------------------------------------------------------------
# WHAT IT TRAINS
#
# configs/calo_clustering.yaml as this repository now has it: the simplified-baseline arm of the
# ablation -- AdamW at max lr 1e-4, accumulate_grad_batches 1, gradient_clip_val 1.0, no incidence
# head, and a mask head with mask_dice = mask_bce = 1.0 in the loss and mask_dice alone in the cost.
# seed_everything: 42, so the arms of the ablation share an initialisation.
#
# THE CONFIG IS PASSED BY ABSOLUTE PATH INTO THIS REPOSITORY (env.sh sets CONFIG), never as a path
# resolved against the checkout, whose copy is still the superseded Lion configuration.
#
# ---------------------------------------------------------------------------------------------
# RESOURCES AND WALLTIME -- ALL MEASURED, 2026-08-05, benchmark_calo_clustering.sh on this config
#
# Two probes, twelve and ten minutes of the real configuration at the real num_train=20000:
#
#   allocation            rate        GPU util   peak GPU mem   host RSS   epoch    7 epochs
#   12 CPU / 12 workers   1.09 ev/s   16%        49.2 / 81.9 GB   131 GB   5.19 h    36.3 h   (48244)
#   32 CPU / 24 workers   1.94 ev/s   30%        49.2 / 81.9 GB   255 GB   2.91 h    20.4 h   (48245)
#
# THE RUN IS INPUT-BOUND, NOT GPU-BOUND, and that is the whole reason for the allocation above.
# At 12 CPUs the A100 is idle 84% of the time waiting for the dataloader; the extra CPUs cost
# nothing and take 9 hours off the run. It is STILL input-bound at 30% util, so this is not the
# ceiling -- it is where the ceiling stops being CPUs and starts being host RAM.
#
# WHY NOT MORE WORKERS. data.py caches 8 decoded row groups PER WORKER (_row_group_cache_size,
# hardcoded), one row group per shard, ~1.4 GB decoded at pu0 -- so resident memory is roughly
# 8 x workers x 1.4 GB, which is the 255 GB measured at 24. The node has 515 GB and is shared, so
# 24 workers is about half of it and roughly the limit worth taking. Going further means lowering
# that cache size in data.py first, not raising --cpus-per-task.
#
# NOTE ON --mem: this cluster runs TaskPlugin=task/affinity with no cgroups, so --mem is used for
# SCHEDULING but is not enforced -- job 48169 booked 128 GB and actually used 185 GB, and finished.
# That makes under-requesting silently "work" while over-booking the node for everyone else. The
# 256G above is what the job measures at, not a guess with margin on top.
#
# WALLTIME, AND WHAT THE EXTRA CPUs WERE SPENT ON. The 32-CPU allocation bought 1.78x the
# throughput; it was NOT taken as a shorter run. calo_clustering.yaml now sets max_epochs: 7, so
# the same ~20 h of walltime buys 140,000 optimiser steps instead of the 80,000 that 4 epochs at
# the old rate would have taken 20.8 h to reach. That is the trade the model wants:
# overlay_long_schedule.yaml sets out the evidence that it is step-starved rather than
# data-starved, and run 48169's validation loss was still falling at its final checkpoint.
#
#   7 epochs x 2.91 h = 20.4 h expected, 26 h requested, MAX_TIME stops it cleanly at 25 h.
#
# The 27% margin is for throughput drift, not for slack: a 20% slowdown still completes. For
# context, job 48169 trained the PREVIOUS configuration -- same events, 4 epochs, same card, but
# 12 CPUs -- in 19:37:05 against a 20:00:00 wall, with 23 minutes to spare. This configuration is
# slightly slower per event (AdamW carries two momentum buffers where Lion carried one, and
# accumulate_grad_batches 4 -> 1 pays the optimiser step four times as often), so at the old
# allocation even 4 epochs would have needed 20.8 h and been TRUNCATED by that wall. The partition
# allows 4-00:00:00; 20 h was always a choice rather than a limit.
#
# OneCycleLR is sized from TOTAL optimiser steps, so a run that hits the wall mid-schedule never
# reaches its decay phase and its final checkpoint is taken at a high learning rate -- the waste
# calo_clustering.yaml and ../ce_ai_1/PU200_STATUS.md §6 both describe. MAX_TIME below stops
# Lightning cleanly an hour before the wall so the run ends with a written checkpoint rather than a
# SIGKILL. IT SHOULD NOT TRIGGER: if it does, the sizing is wrong and the checkpoint it leaves is
# mid-schedule.
#
# ONE KNOB DELIBERATELY NOT TOUCHED: peak GPU memory is 49.2 GB of 81.9, so batch_size 2 would fit
# and would also raise GPU utilisation. It is left at 1 because the effective batch is coupled to
# lrs_config.max and to accumulate_grad_batches, and calo_clustering.yaml is explicit that those
# were set together. Changing it is a training-configuration decision, not a resource one.
#
# ---------------------------------------------------------------------------------------------
# OVERRIDES (environment variables at submit time)
#
#   NUM_TRAIN=3000              --data.num_train        (guarded: see the window check below)
#   MAX_EPOCHS=8                --trainer.max_epochs    re-run the benchmark and resize the wall
#   MAX_TIME=00:25:00:00        --trainer.max_time      D:HH:MM:SS, Lightning's format
#   WORKERS=24                  --data.num_workers      raising it raises RSS, not just CPU use
#   CKPT=<path>/last.ckpt       --ckpt_path             resume
#   OVERLAYS="a.yaml b.yaml"    extra --config files from this repo's configs/, in order
#   OUT_DIR=<path>              --trainer.default_root_dir  (default: external/logs here)
#   SYNC=0                      skip re-copying this repo's data.py/model.py into the checkout
#   ALLOW_FAULTY_GPU=1          run even on a card reporting uncorrected ECC errors
#
# DO NOT MOVE THIS TO LIGHTGPU. That partition is usually idle, and calo_dump_eventstore.sh
# defaults to it, which makes it a tempting way to skip the queue. It is compute-gpu-0-0, which
# /etc/slurm/gpu_variables.sh shows is carved into six MIG instances, and the cluster documentation
# (https://uclphysast.github.io/clusters/dias/) puts them at 20 GB each. Dumping a store is a
# forward pass and fits; training this configuration does not. env.sh warns if it is tried.
#
# The GPU partition's cards are 80 GB, measured: job 48244 reported "NVIDIA A100 80GB PCIe,
# 81920 MiB", MIG disabled. The same documentation page calls them 40 GB and is out of date.
# ---------------------------------------------------------------------------------------------

set -euo pipefail

# Hardcoded: Slurm copies this script into a spool directory, so ${BASH_SOURCE[0]} is "/" here.
# shellcheck disable=SC1091
. /home/xucapfwi/MSc_Thesis_SDIC/src/maskformer/dias/env.sh

OUT_DIR="${OUT_DIR:-$REPO/external/logs}"
MAX_TIME="${MAX_TIME:-00:25:00:00}"
# 24, NOT SLURM_CPUS_PER_TASK. Deriving workers from the CPU count is the obvious thing and it is
# wrong here: resident memory scales with workers (8 cached row groups each), not with cores, so
# tying them together makes --cpus-per-task silently a memory knob. 32 cores / 24 workers is the
# measured pairing; the spare cores serve the main process's collate and the Hungarian matcher.
WORKERS="${WORKERS:-24}"

echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "Config:    $CONFIG"
echo "Logs:      $OUT_DIR"

preflight_paths
mkdir -p "$OUT_DIR"
select_gpu

# --- event windows -------------------------------------------------------------------------------
# Refuse to start if the training window runs into the evaluation window. calo_clustering.yaml
# splits pu0 as train [0, 20000), val [20000, 20250), test [20250, 20750), and the event store the
# CLUE comparison is scored over is dumped from [20250, 20750). Training into it would make the
# head-to-head a test on training data. src/io/event_store.py asserts this, but hours later at
# scoring time; fail here instead, in seconds. Same guard as ce_ai_1/train_pu200.sh.
NUM_TRAIN_EFF="${NUM_TRAIN:-20000}"
if [ "$NUM_TRAIN_EFF" -gt 20000 ]; then
  echo "ABORT: num_train=$NUM_TRAIN_EFF runs past the validation window at 20000 and into the"
  echo "       evaluation window [20250, 20750) the event store is dumped from. Keep it at 20000,"
  echo "       or move val_start_event/test_start_event and the store window together."
  exit 1
fi
echo "Window check OK: train [0, $NUM_TRAIN_EFF) is disjoint from val/test and the store window."

sync_mirror
build_config_args
load_comet_key

EXTRA_ARGS=()
[ -n "${NUM_TRAIN:-}" ]  && EXTRA_ARGS+=(--data.num_train "$NUM_TRAIN")
[ -n "${MAX_EPOCHS:-}" ] && EXTRA_ARGS+=(--trainer.max_epochs "$MAX_EPOCHS")
[ -n "${CKPT:-}" ]       && EXTRA_ARGS+=(--ckpt_path "$CKPT")
echo "Extra:     ${EXTRA_ARGS[*]:-(none)}"

cd "$EXP_DIR"

run_in_container "$ENV_PYTHON" main.py fit \
  "${CONFIG_ARGS[@]}" \
  --trainer.default_root_dir "$OUT_DIR" \
  --trainer.max_time "$MAX_TIME" \
  --data.num_workers "$WORKERS" \
  --data.pin_memory false \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Finished:  $(date)"
echo
# WHAT TO CHECK, in this order:
#   1. Did it reach the end of the OneCycle schedule? If the last logged lr is not near
#      lrs_config.end (1e-6), the run was truncated and its final checkpoint is not the one to
#      quote. MAX_TIME triggering is the usual cause; re-run the benchmark and resize.
#   2. num_calohit_per_flow against num_calohit_per_part -- has it learned cluster SCALE? This
#      moves long before thresholded efficiency does, so it is the earliest honest signal.
#   3. p0.5_calohit_eff together with p0.5_calohit_pur, never either alone: a model that
#      over-assigns hits buys efficiency with purity and looks like progress when it is not.
#   4. flow_valid's eval_threshold is 0.2 in the config and STALE -- it was swept against the old
#      objective, which had the focal term and the incidence head. Re-run
#      scripts/sweep_pred_threshold.py on this checkpoint before quoting any efficiency.
echo "Logs and checkpoints: $OUT_DIR"
