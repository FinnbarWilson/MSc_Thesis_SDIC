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

# A C COMPILER, WITHOUT WHICH THE RUN DIES AT THE FIRST TRITON KERNEL. docker://ubuntu:22.04 is the
# bare base image and ships no cc/gcc/g++ at all. Triton does not merely want a compiler for
# torch.compile -- it shells out to one during DRIVER INITIALISATION, to build its CUDA utils
# module, so ANY triton kernel launch needs it. Job 48398 died this way ~4 minutes in with
# "RuntimeError: Failed to find C compiler. Please specify via CC environment variable".
#
# The compiler already exists: the pixi env carries a full conda-forge GCC toolchain. It is simply
# not on PATH, because the job invokes $ENV_PYTHON by absolute path and never activates the env.
# Pointing CC/CXX at it is enough -- verified in-container on the node: a trivial C binary builds
# and triton driver init reaches CudaDriver. Apptainer forwards the host environment by default,
# so exporting here (outside the container) is what lands inside it.
CONDA_BIN="$HEPATTN/.pixi/envs/default/bin"
export CC="${CC:-$CONDA_BIN/gcc}"
export CXX="${CXX:-$CONDA_BIN/g++}"

# THE config, by absolute path into this repository. main.py runs from the hepattn CHECKOUT, whose
# configs/ is whatever was last copied there -- a relative --config can silently train a stale copy.
DATASET="${DATASET:-pu0}"
CONFIG="${CONFIG:-$MIRROR/configs/${DATASET}.yaml}"

# Many dataloader workers passing large tensors can exhaust the fd limit. data.py also sets the
# file_system sharing strategy, which is the primary guard against "received 0 items of ancdata".
ulimit -n 65536 2>/dev/null || true

# PIN EVERY NATIVE THREAD POOL TO ONE THREAD. THIS IS THE DIFFERENCE BETWEEN THIS CLUSTER AND
# ce-ai-1, and without it the job does not train at all.
#
# ce-ai-1 is a whole box: 104 cores, nothing else on them, no affinity mask. DIAS is Slurm with
# TaskPlugin=task/affinity, so the job is confined to a 32-CPU mask out of the node's 128 -- but the
# libraries inside each dataloader worker size their pools from the MACHINE, via sysconf/cpu_count,
# and never see the mask. pyarrow's parquet decode pool is the main offender; OpenMP/MKL/OpenBLAS
# do the same.
#
# MEASURED on job 48405, 24 configured workers: 48 worker processes (train and val dataloaders both
# live) holding 960 threads on 32 permitted cores -- 30x oversubscription. The result after 35
# minutes was 31 cores pegged, 0 MB read from disk in 30 s, 0% GPU utilisation and NOT ONE optimiser
# step. All of the CPU went into contention rather than decode, which is also why the earlier
# "input-bound, 30% GPU" readings should be re-measured rather than trusted -- some of that was
# probably this, not the dataset.
#
# torch already calls set_num_threads(1) inside its workers; these cover everything it does not own.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
# OpenMP's default is to SPIN while waiting for work, which is what turns oversubscription into
# pegged cores rather than idle ones. Nothing here benefits from the low-latency wake-up.
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"

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
# The faulty card is compute-gpu-0-1 GPU 2, which kills jobs with "CUDA error: uncorrectable ECC
# error" minutes in -- job 48163 died at 3:30. Requesting TWO cards is what guarantees the
# allocation contains a healthy one: the node has three, only one is faulty, so any two must
# include a good one. Only one is ever used; trainer.devices stays 1.
#
# ERROR COUNT, re-checked 2026-08-13 and still climbing: GPU 2 is at 1413 aggregate, up from 844
# on 2026-08-05 and 818 the week before. GPUs 0 and 1 remain at 0/0, so two cards is still enough.
# RE-CHECK IT AGAIN before trusting this: if a second card goes bad, two no longer guarantees one.
select_gpu() {
    local alloc ecc vol agg healthy="" first=""
    alloc="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}}"

    # MIG (LIGHTGPU): the allocation is a UUID and ECC is not queryable per instance by index.
    case "$alloc" in
        *MIG*|*GPU-*) export CUDA_VISIBLE_DEVICES="$alloc"; echo "MIG instance: $alloc"; return 0 ;;
    esac
    [ -n "$alloc" ] || { echo "ABORT: no GPU allocation visible (CUDA_VISIBLE_DEVICES / SLURM_JOB_GPUS unset)"; return 1; }

    for idx in ${alloc//,/ }; do
        [ -n "$first" ] || first="$idx"
        # BOTH counters, not just volatile. The volatile count resets on a driver reload or GPU
        # reset; the aggregate one lives in the inforom and survives. On 2026-08-13 the faulty card
        # read volatile=0 against aggregate=1413, so the volatile-only check this used to do called
        # it HEALTHY and would have routed the job straight onto it -- defeating the whole point of
        # asking for two cards. A card is a candidate only if both counters are zero.
        ecc=$(nvidia-smi -i "$idx" \
              --query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total \
              --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
        vol="${ecc%%,*}"; agg="${ecc##*,}"
        case "$vol$agg" in
            ''|*[!0-9]*) echo "  GPU $idx: ECC count unreadable ('$ecc')" ;;
            *[!0]*)      echo "  GPU $idx: $vol volatile / $agg aggregate uncorrected ECC errors -- AVOIDING" ;;
            *)           echo "  GPU $idx: 0 uncorrected ECC errors -- healthy"; [ -n "$healthy" ] || healthy="$idx" ;;
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
#
# THIS COUPLES THE RUN TO --mem, WHICH train.sh's "--mem IS NOT ENFORCED" NOTE DOES NOT COVER.
# Slurm here applies VSizeFactor, setting `ulimit -v` to 1.1 x --mem: measured 2026-08-13, --mem=32G
# gave a 35.2 GB cap and --mem=256G gave 281.6 GB. expandable_segments reserves a large VIRTUAL
# address range up front, so under a small cap CUDA context creation itself fails -- and it fails as
# "RuntimeError: CUDA driver error: out of memory" on a card with 78.8 GB free and no processes on
# it, which reads like a GPU problem and is not one. Probed both ways at both sizes: 32G fails with
# expandable_segments and succeeds without it; 256G succeeds either way.
#
# So --mem is unenforced for RESIDENT memory but is a hard cap on VIRTUAL memory. Lowering it to
# "just what the job needs" will break CUDA start-up long before it ever limits RSS.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
