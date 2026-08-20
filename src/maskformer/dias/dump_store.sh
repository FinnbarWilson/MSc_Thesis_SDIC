#!/bin/bash -l
#SBATCH --job-name=calo_dump
#SBATCH --partition=GPU
# Two cards for the same reason train.sh asks for two: one card on this node reports climbing
# uncorrected ECC errors. Only one is used; env.sh's select_gpu picks the healthy one.
#SBATCH --gres=gpu:a100:2
# Inference, not training: fewer workers than train.sh and no optimiser state.
#
# --mem looks absurd for inference and is not negotiable: Slurm's VSizeFactor here turns it into
# a hard virtual cap of 1.1x, and expandable_segments reserves a large virtual range. An
# under-request surfaces as "CUDA driver error: out of memory" on an idle 80 GB card.
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00
# Relative to the directory sbatch was run from, so submit from the repository root.
#SBATCH --output=external/slurm_logs/calo_dump_%j.out
#SBATCH --error=external/slurm_logs/calo_dump_%j.err

# Dump an event store on DIAS. The counterpart to ce_ai_1/dump_store.sh, and the same step in the
# argument: CLUE does not read ColliderML, it reads the store written here from the MODEL's own
# dataloader. Every cut the training config applied, including the shower-level truth definition
# `particle_collapse_shower_secondaries`, therefore applies to both methods by construction
# rather than by two configs agreeing.
#
# What differs from ce-ai-1, and it is the same three things train.sh differs by:
#   RHEL7 glibc 2.17          -> everything runs inside ~/ubuntu22.sif
#   no scheduler              -> sbatch, and a card that has to be chosen around an ECC fault
#   plain venv                -> the pixi env inside the container
#
# Dump both stores after training, from the windows the training config leaves free:
#
#   CKPT=<ckpt> sbatch src/maskformer/dias/dump_store.sh pu0 tune   # [20000, 20050),  50 events
#   CKPT=<ckpt> sbatch src/maskformer/dias/dump_store.sh pu0 eval   # [20250, 20750), 500 events
#
# The tune window is the training run's own validation split: unseen by the optimiser, but seen by
# model selection. Same status as CLUE's tuning window: both methods pick
# their working point on the same events and neither picks it on the reported ones.
#
# The store is written under STORE_ROOT, which defaults to external/eventstores, the directory
# config/experiment.yaml already reads. Its name is what matters: eval/dump.py builds it as
#   <dataset_prefix>_<start>_<start+num>_v<FORMAT_VERSION>
# so it encodes the window and format version, and event_store.py checks both against the config.
#
#   export CALO_STORE_ROOT=<dir>      # only if you put the stores outside external/
#
# OVERRIDES
#   CKPT=<path>       the checkpoint to dump from        (required)
#   STORE_ROOT=<dir>  where the store directory is made  (default external/eventstores)
#   CHUNK=<n>         events held in memory per .npz     (default 25 at pu0, 10 at pu200)
#   WORKERS=<n>       dataloader workers for the dump    (default 8)
set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
. "$REPO/src/maskformer/dias/env.sh"

: "${CKPT:?Set CKPT=/path/to/checkpoint.ckpt}"
DUMP_DATASET="${1:-pu0}"
WHICH="${2:-eval}"

# The windows are the ce-ai-1 script's, unchanged: they are a property of the experiment, not of
# the machine, and the two must not drift apart.
case "$DUMP_DATASET:$WHICH" in
    pu0:tune)    START=20000; NUM=50  ;;
    pu0:eval)    START=20250; NUM=500 ;;
    pu200:tune)  START=7000;  NUM=50  ;;
    pu200:eval)  START=7500;  NUM=500 ;;
    *) echo "usage: $0 [pu0|pu200] [tune|eval]   (got '${DUMP_DATASET:-}' '${WHICH:-}')" >&2; exit 2 ;;
esac

STORE_ROOT="${STORE_ROOT:-${CALO_STORE_ROOT:-$EXTERNAL/eventstores}}"

# CHUNK is the memory knob: the dump holds a whole chunk in memory while writing it. A pu200 event
# carries ~117k cells against pu0's ~22k, so it needs a smaller chunk for the same footprint.
if [ "$DUMP_DATASET" = "pu200" ]; then CHUNK="${CHUNK:-10}"; else CHUNK="${CHUNK:-25}"; fi

# INCIDENCE_TOP_K is deliberately not set. eval/format.py carries the measured value (16) and is
# the one place that choice is justified; setting it here silently overrides it, which has happened
# once before (4 against format's 16).

# The run config, and the fallback that makes a FETCHED checkpoint work.
#
# eval/dump.py defaults --config to config.yaml two levels above the checkpoint, which Lightning
# writes during training. scripts.fetch_checkpoints downloads only the .ckpt, so that file does
# not exist for a fetched one. Fall back to this repository's config for the condition, and pass
# the shard directory explicitly, because the repository config's paths are relative to the
# repository root while this script may run from elsewhere.
RUN_CONFIG="${RUN_CONFIG:-}"
if [ -z "$RUN_CONFIG" ] && [ ! -f "$(dirname "$(dirname "$CKPT")")/config.yaml" ]; then
    RUN_CONFIG="$MIRROR/configs/${DUMP_DATASET}.yaml"
    echo "note       : no run config beside the checkpoint; using $RUN_CONFIG"
fi
CONFIG_ARGS=()
if [ -n "$RUN_CONFIG" ]; then
    [ -f "$RUN_CONFIG" ] || { echo "ABORT: no config at $RUN_CONFIG"; exit 1; }
    CONFIG_ARGS+=(--config "$RUN_CONFIG" --data-dir "$DATA_DIR/ttbar_${DUMP_DATASET}/")
fi

echo "host       : $(hostname)"
echo "started    : $(date)"
echo "job        : ${SLURM_JOB_ID:-<interactive>}"
echo "checkpoint : $CKPT"
echo "dataset    : $DUMP_DATASET"
echo "window     : [$START, $((START + NUM)))  ($WHICH store)"
echo "out        : $STORE_ROOT"
echo "chunk      : $CHUNK events per .npz"

[ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT"; exit 1; }
for p in "$SIF" "$ENV_PYTHON" "$EXP_DIR"; do
    [ -e "$p" ] || { echo "ABORT: missing $p"; exit 1; }
done
mkdir -p "$STORE_ROOT" || exit 1
df -h "$STORE_ROOT" | tail -1

sync_mirror || exit 1
select_gpu  || exit 1

# The dump reads the raw shards through the model's own dataloader, so the dataset is bound into
# the container exactly as training does.
echo
exec apptainer exec --nv \
    --bind "$DATA_DIR" \
    --bind "$REPO" \
    --bind "$HEPATTN" \
    --bind "$STORE_ROOT" \
    "$SIF" \
    "$ENV_PYTHON" -m hepattn.experiments.colliderml.eval.dump "$CKPT" \
        --start-event "$START" --num-events "$NUM" --out "$STORE_ROOT" \
        --chunk-size "$CHUNK" --num-workers "${WORKERS:-8}" \
        ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"}
