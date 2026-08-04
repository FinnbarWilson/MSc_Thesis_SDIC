#!/bin/bash -l
#SBATCH --job-name=calo_train
#SBATCH --partition=GPU
# Two GPUs on purpose. compute-gpu-0-1 GPU 2 has 818 uncorrected ECC errors and kills jobs with
# "CUDA error: uncorrectable ECC error" within minutes (measured: job 48163 died at 3.5 min). It is
# usually the only free card, so a 1-GPU request repeatedly lands on it. The node has 3 GPUs and only
# one is faulty, so any two are guaranteed to include a healthy one; the block below selects it.
# The job queues until two are free rather than gambling an 18-hour run on which card it gets.
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=20:00:00
# Absolute paths: SBATCH directives are resolved against the directory you submit from, so a
# relative path fails (job aborts instantly) unless that dir happens to contain slurm_logs/.
#SBATCH --output=/home/xucapfwi/hepattn/src/hepattn/experiments/colliderml/slurm_logs/calo_train_%j.out
#SBATCH --error=/home/xucapfwi/hepattn/src/hepattn/experiments/colliderml/slurm_logs/calo_train_%j.err

# Proper MaskFormer calo-clustering training run on ColliderML pu0.
#
# Unlike slurm/light_calo_clustering.sh (a 10-event correctness smoke test), this trains on a real
# disjoint split defined in configs/calo_clustering.yaml: train = events [0, 3000),
# val = [3000, 3500), test = [3500, 4000). ~1e6 events are available if you want to scale further
# (bump num_train/num_val/num_test and the *_start_event offsets in the config, or use NUM_TRAIN=...).
#
# ENVIRONMENT: DIAS is RHEL7 (glibc 2.17) but the pixi env needs glibc 2.28+, so we run the
# (already built) env inside an Ubuntu 22.04 apptainer container. Build the image once on the
# login node:  apptainer build ~/ubuntu22.sif docker://ubuntu:22.04
#
# Submit from the repo root:
#     mkdir -p src/hepattn/experiments/colliderml/slurm_logs
#     sbatch src/hepattn/experiments/colliderml/slurm/calo_clustering.sh
#
# Optional overrides via environment variables when submitting, e.g.:
#     NUM_TRAIN=20000 MAX_EPOCHS=30 sbatch src/hepattn/experiments/colliderml/slurm/calo_clustering.sh
# Resume from a checkpoint:
#     CKPT=logs/<run>/ckpts/last.ckpt sbatch src/hepattn/experiments/colliderml/slurm/calo_clustering.sh

set -euo pipefail

# Raise the open-file-descriptor limit: many dataloader workers passing large tensors can exhaust
# it (data.py also sets the file_system sharing strategy, which is the primary guard against the
# "received 0 items of ancdata" crash). Best-effort; ignore if the shell caps it lower.
ulimit -n 65536 2>/dev/null || true

[ -f /etc/slurm/gpu_variables.sh ] && source /etc/slurm/gpu_variables.sh

SIF="${HEPATTN_SIF:-$HOME/ubuntu22.sif}"
ENV_PYTHON="$HOME/hepattn/.pixi/envs/default/bin/python"

# Hardcoded: Slurm copies this script into a spool dir before running it, so deriving the repo root
# from ${BASH_SOURCE[0]} resolves to "/" under sbatch. Everything else in the repo hardcodes the
# /home/xucapfwi paths too, so we do the same here.
REPO_ROOT="/home/xucapfwi/hepattn"
EXP_DIR="$REPO_ROOT/src/hepattn/experiments/colliderml"
cd "$EXP_DIR"

echo "Repo root: $REPO_ROOT"
echo "Node:      $(hostname)"
echo "SIF:       $SIF"
nvidia-smi --query-gpu=index,memory.used,ecc.errors.uncorrected.aggregate.total --format=csv || true

# Preflight: refuse to start on a GPU with uncorrected ECC errors. compute-gpu-0-1 GPU 2 has
# hundreds (764 and climbing) and kills jobs with "CUDA error: uncorrectable ECC error" as soon as
# training starts. GPUs 0/1 are usually busy so Slurm hands out the broken one by default. Failing
# here costs seconds instead of a queue slot and a day of walltime.
MY_GPUS="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
HEALTHY=""
for g in ${MY_GPUS//,/ }; do
  ecc=$(nvidia-smi -i "$g" --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null || echo 0)
  echo "Allocated GPU $g: uncorrected ECC errors = $ecc"
  case "$ecc" in
    ''|*[!0-9]*) HEALTHY="${HEALTHY:-$g}" ;;
    *) if [ "$ecc" -eq 0 ]; then HEALTHY="${HEALTHY:-$g}"; fi ;;
  esac
done
if [ -n "$MY_GPUS" ] && [ -z "$HEALTHY" ]; then
  echo "ABORT: every allocated GPU ($MY_GPUS) has uncorrected ECC errors. Report this node."
  exit 1
fi
if [ -n "$HEALTHY" ]; then
  echo "Using GPU $HEALTHY"
  export CUDA_VISIBLE_DEVICES="$HEALTHY"
fi

# Optional overrides (fall back to the values baked into the config).
CONFIG="${CONFIG:-configs/calo_clustering.yaml}"
EXTRA_ARGS=()
[ -n "${NUM_TRAIN:-}" ]  && EXTRA_ARGS+=(--data.num_train "$NUM_TRAIN")
[ -n "${MAX_EPOCHS:-}" ] && EXTRA_ARGS+=(--trainer.max_epochs "$MAX_EPOCHS")
[ -n "${CKPT:-}" ]       && EXTRA_ARGS+=(--ckpt_path "$CKPT")

# --- torch.compile ------------------------------------------------------------------------------
# The cost functions are torch.compile'd for a real throughput gain, but the bare ubuntu container
# has no C compiler for Triton's JIT. Two options:
#   (A) DEFAULT below: disable compile -> runs eager. Slower but bullet-proof for long unattended runs.
#   (B) Point Triton at the env's self-contained gcc so it can JIT and keep the speedup. To use it,
#       comment out the two --env ..._DISABLE=1 lines below and uncomment the CC block.
COMPILE_ENV=(--env TORCH_COMPILE_DISABLE=1 --env TORCHDYNAMO_DISABLE=1)
# COMPILE_ENV=(--env CC="$HOME/hepattn/.pixi/envs/default/bin/cc" \
#              --env CXX="$HOME/hepattn/.pixi/envs/default/bin/c++")
# ------------------------------------------------------------------------------------------------

echo "Config:     $CONFIG"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"

# Comet online logging: my own comet key.
if [ -z "${COMET_API_KEY:-}" ]; then
  echo "WARNING: COMET_API_KEY is not set. Comet logging will fail. Run 'export COMET_API_KEY=...' before sbatch."
fi

apptainer exec --nv --bind /home/xucapfwi/ColliderML_data \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  --env COMET_API_KEY="${COMET_API_KEY:-}" \
  "${COMPILE_ENV[@]}" "$SIF" \
  "$ENV_PYTHON" main.py fit \
    --config "$CONFIG" \
    --data.pin_memory false \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "Done. Inspect logs/<run>/metrics.csv and the checkpoints under logs/<run>/ckpts/."
