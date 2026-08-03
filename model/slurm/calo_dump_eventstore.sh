#!/bin/bash -l
#SBATCH --job-name=calo_dump
#SBATCH --partition=LIGHTGPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/xucapfwi/hepattn/src/hepattn/experiments/colliderml/slurm_logs/calo_dump_%j.out
#SBATCH --error=/home/xucapfwi/hepattn/src/hepattn/experiments/colliderml/slurm_logs/calo_dump_%j.err

# Dump cells, truth and MaskFormer predictions into a portable event store, for the CLUE
# comparison. One forward pass per event; nothing is trained and nothing is matched here.
#
#   CKPT=/path/to/clustering.ckpt START=20250 NUM=500 sbatch .../calo_dump_eventstore.sh
#
#   OUT=$HOME/eventstore     where the store directory is created
#   CHUNK=25                 events per .npz
#   STORE_MASK_THRESHOLD     mask probability floor for the stored sparse masks; this is the
#                            lowest working point any later scan can reach, so lowering it
#                            costs disk and raising it costs reach
#   MAX_HITS_PER_QUERY=0     cap stored hits per query (0 = uncapped)
#   INCIDENCE_TOP_K          (query, share) pairs kept per cell from the incidence head. 1 is
#                            enough for the exclusive metric, which only needs the argmax; >1
#                            is what the multi-owner capability study reads. Unset by default,
#                            so eval/format.py's measured INCIDENCE_TOP_K applies -- do NOT
#                            restate a number here, or this file silently overrides the one
#                            place the choice is justified (it did, at 4 against format's 16).
#   ALLOW_FAULTY_GPU=1       run even on a card with uncorrected ECC errors
#
# Defaults to LIGHTGPU (compute-gpu-0-0), which has six healthy A100s. The GPU partition maps
# to compute-gpu-0-1, whose GPU 2 has ~818 uncorrected ECC errors; the check below still runs
# so an explicit --partition=GPU cannot silently land on it.

set -euo pipefail

ulimit -n 65536 2>/dev/null || true
[ -f /etc/slurm/gpu_variables.sh ] && source /etc/slurm/gpu_variables.sh

SIF="${HEPATTN_SIF:-$HOME/ubuntu22.sif}"
ENV_PYTHON="$HOME/hepattn/.pixi/envs/default/bin/python"
EXP_DIR="/home/xucapfwi/hepattn/src/hepattn/experiments/colliderml"
cd "$EXP_DIR"

: "${CKPT:?Set CKPT=/path/to/clustering/checkpoint.ckpt}"
START="${START:-20250}"
NUM="${NUM:-500}"
OUT="${OUT:-$HOME/eventstore}"
CHUNK="${CHUNK:-25}"
STORE_MASK_THRESHOLD="${STORE_MASK_THRESHOLD:-0.02}"
MAX_HITS_PER_QUERY="${MAX_HITS_PER_QUERY:-0}"

echo "Node: $(hostname)"
echo "Dumping events [$START, $((START + NUM))) from $CKPT to $OUT"

# Enumerate the devices VISIBLE to this job, not the ids slurm reports. SLURM_JOB_GPUS gives
# node-global indices (e.g. "12"), but the job's cgroup exposes only its own allocation,
# renumbered from 0 -- so `nvidia-smi -i 12` answers "No devices were found" and a naive
# check mistakes that for a fault and aborts.
HEALTHY=""; FIRST=""
while IFS=, read -r idx ecc; do
  idx="${idx// /}"; ecc="${ecc// /}"
  [ -z "$idx" ] && continue
  echo "Visible GPU $idx: uncorrected ECC errors = $ecc"
  FIRST="${FIRST:-$idx}"
  case "$ecc" in
    ''|*[!0-9]*) ;;
    *) if [ "$ecc" -eq 0 ]; then HEALTHY="${HEALTHY:-$idx}"; fi ;;
  esac
done < <(nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null)

if [ -z "$FIRST" ]; then
  echo "ABORT: no GPU visible to this job."
  exit 1
fi
if [ -z "$HEALTHY" ]; then
  if [ "${ALLOW_FAULTY_GPU:-0}" != "1" ]; then
    echo "ABORT: every visible GPU reports uncorrected ECC errors. Set ALLOW_FAULTY_GPU=1 to try anyway."
    exit 1
  fi
  echo "WARNING: proceeding on faulty GPU $FIRST because ALLOW_FAULTY_GPU=1."
  HEALTHY="$FIRST"
fi
export CUDA_VISIBLE_DEVICES="$HEALTHY"

df -h "$OUT" 2>/dev/null || df -h "$HOME"

apptainer exec --nv --bind /home/xucapfwi/ColliderML_data \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env TORCH_COMPILE_DISABLE=1 --env TORCHDYNAMO_DISABLE=1 "$SIF" \
  "$ENV_PYTHON" -m hepattn.experiments.colliderml.eval.dump "$CKPT" \
    --start-event "$START" --num-events "$NUM" --out "$OUT" \
    --chunk-size "$CHUNK" --num-workers "${SLURM_CPUS_PER_TASK:-10}" \
    --store-mask-threshold "$STORE_MASK_THRESHOLD" \
    --max-hits-per-query "$MAX_HITS_PER_QUERY" \
    ${INCIDENCE_TOP_K:+--incidence-top-k "$INCIDENCE_TOP_K"}
