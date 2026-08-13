# Shared environment for the DIAS job scripts. Sourced by them, not run directly.
#
# The counterpart to ce_ai_1/env.sh. That file exists because ce-ai-1 needs no container and has no
# scheduler; this one exists because DIAS has both, plus a faulty GPU that has to be routed around.
#
# Sourced by ABSOLUTE PATH, not derived from ${BASH_SOURCE[0]}: Slurm copies a submitted script into
# a spool directory before running it, so BASH_SOURCE resolves to "/" under sbatch.

REPO="${REPO:-/home/xucapfwi/MSc_Thesis_SDIC}"
MIRROR="$REPO/src/maskformer/hepattn_colliderml"
HEPATTN="${HEPATTN:-/home/xucapfwi/hepattn}"
EXP_DIR="$HEPATTN/src/hepattn/experiments/colliderml"
DATA_DIR="${DATA_DIR:-/home/xucapfwi/ColliderML_data}"

# DIAS is RHEL7 (glibc 2.17) and hepattn needs glibc 2.28+, so everything runs inside an Ubuntu
# 22.04 container. This is the one hard difference from ce-ai-1, which runs a plain venv.
#   apptainer build ~/ubuntu22.sif docker://ubuntu:22.04
SIF="${HEPATTN_SIF:-$HOME/ubuntu22.sif}"
ENV_PYTHON="${ENV_PYTHON:-$HEPATTN/.pixi/envs/default/bin/python}"

# THE config, by absolute path into this repository. main.py runs from the hepattn CHECKOUT, whose
# configs/ is whatever was last copied there -- a relative --config can silently train a stale copy.
DATASET="${DATASET:-pu0}"
CONFIG="${CONFIG:-$MIRROR/configs/${DATASET}.yaml}"

# Many dataloader workers passing large tensors can exhaust the fd limit. data.py also sets the
# file_system sharing strategy, which is the primary guard against "received 0 items of ancdata".
ulimit -n 65536 2>/dev/null || true

# The cluster's own hook. On compute-gpu-0-0 (LIGHTGPU) it rewrites CUDA_VISIBLE_DEVICES from a
# Slurm index to a MIG UUID; on compute-gpu-0-1 (GPU) it does nothing. Source it BEFORE reading
# CUDA_VISIBLE_DEVICES, or the MIG case reads a number that means nothing.
[ -f /etc/slurm/gpu_variables.sh ] && source /etc/slurm/gpu_variables.sh

# Re-sync this repository's copies into the checkout, as ce_ai_1/env.sh does and for the same
# reason: main.py imports from the checkout, so editing a config here and submitting without
# copying it across trains the stale copy and looks like the edit did nothing.
sync_mirror() {
    [ "${SYNC:-1}" = "0" ] && { echo "SYNC=0: leaving the checkout as it is"; return 0; }
    [ -d "$EXP_DIR" ] || { echo "ABORT: no experiment dir at $EXP_DIR"; return 1; }
    mkdir -p "$EXP_DIR"/{configs,eval}
    cp "$MIRROR"/*.py          "$EXP_DIR/"         || return 1
    cp "$MIRROR"/configs/*.yaml "$EXP_DIR/configs/" || return 1
    cp "$MIRROR"/eval/*.py      "$EXP_DIR/eval/"    || return 1
    echo "synced $MIRROR -> $EXP_DIR"
}

preflight_paths() {
    for p in "$SIF" "$ENV_PYTHON" "$CONFIG" "$MIRROR/main.py" "$EXP_DIR"; do
        [ -e "$p" ] || { echo "ABORT: missing $p"; return 1; }
    done
    [ -d "$DATA_DIR/ttbar_${DATASET}" ] || { echo "ABORT: no dataset at $DATA_DIR/ttbar_${DATASET}"; return 1; }
}

# Pick a card that is (a) ours and (b) not the one that kills jobs. Exports CUDA_VISIBLE_DEVICES.
#
# THIS NODE HAS NO CGROUP DEVICE ISOLATION. `nvidia-smi` inside a job on compute-gpu-0-1 lists all
# three cards whatever was allocated -- verified in job 48244, which asked for two and saw indices
# 0, 1 and 2. So the visible device list is NOT the allocation, and choosing from it would happily
# select a card belonging to somebody else's job. The allocation only ever comes from Slurm's own
# variables, and a card outside it is never a candidate.
#
# The faulty card is compute-gpu-0-1 GPU 2: 844 uncorrected ECC errors as of 2026-08-05 and
# climbing (818 a week earlier), which kills jobs with "CUDA error: uncorrectable ECC error"
# minutes in -- job 48163 died at 3:30. Requesting TWO cards is what guarantees the allocation
# contains a healthy one: the node has three, only one is faulty, so any two must include a good
# one. Only one is ever used; trainer.devices stays 1.
#
# RE-CHECK THE ERROR COUNT before trusting this comment -- it was climbing. If a second card has
# gone bad, two is no longer enough.
select_gpu() {
    local alloc ecc healthy="" first=""
    alloc="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}}"

    # MIG (LIGHTGPU): the allocation is a UUID and ECC is not queryable per instance by index.
    case "$alloc" in
        *MIG*|*GPU-*) export CUDA_VISIBLE_DEVICES="$alloc"; echo "MIG instance: $alloc"; return 0 ;;
    esac
    [ -n "$alloc" ] || { echo "ABORT: no GPU allocation visible (CUDA_VISIBLE_DEVICES / SLURM_JOB_GPUS unset)"; return 1; }

    for idx in ${alloc//,/ }; do
        [ -n "$first" ] || first="$idx"
        ecc=$(nvidia-smi -i "$idx" --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null)
        case "$ecc" in
            ''|*[!0-9]*) echo "  GPU $idx: ECC count unreadable ('$ecc')" ;;
            0)           echo "  GPU $idx: 0 uncorrected ECC errors -- healthy"; [ -n "$healthy" ] || healthy="$idx" ;;
            *)           echo "  GPU $idx: $ecc uncorrected ECC errors -- AVOIDING" ;;
        esac
    done

    if [ -z "$healthy" ]; then
        [ "${ALLOW_FAULTY_GPU:-0}" = "1" ] || {
            echo "ABORT: no healthy card in the allocation. Set ALLOW_FAULTY_GPU=1 to run anyway."; return 1; }
        healthy="$first"
        echo "ALLOW_FAULTY_GPU=1: using GPU $healthy despite its ECC count"
    fi
    export CUDA_VISIBLE_DEVICES="$healthy"
    echo "using GPU $healthy"
    nvidia-smi -i "$healthy" --query-gpu=name,memory.total --format=csv,noheader
}

# torch 2.9 renamed this; set both. Event sizes vary a lot (pu0 cells/event: ~22k mean, 62k max) so
# the allocator sees a different shape every step and fragments. expandable_segments lets it grow
# segments rather than hoarding fixed-size ones, which is what stops an OOM hours in with plenty of
# free memory in the wrong-sized blocks.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
