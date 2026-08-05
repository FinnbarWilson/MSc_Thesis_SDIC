# Shared environment for the DIAS job scripts. Sourced by the scripts beside it, not run directly.
#
# The counterpart to ce_ai_1/env.sh. That file exists because ce-ai-1 needed different paths and no
# container; this one exists because three job scripts were each carrying their own copy of the GPU
# preflight, and one wrong copy of a safety check is worse than none -- it looks like the check is
# being made.
#
# Sourced by absolute path, not derived from ${BASH_SOURCE[0]}: Slurm copies a submitted script
# into a spool directory before running it, so BASH_SOURCE resolves to "/" under sbatch.

REPO="/home/xucapfwi/MSc_Thesis_SDIC"
MIRROR="$REPO/src/maskformer/hepattn_colliderml"
HEPATTN="/home/xucapfwi/hepattn"
EXP_DIR="$HEPATTN/src/hepattn/experiments/colliderml"
DATA_DIR="/home/xucapfwi/ColliderML_data"

SIF="${HEPATTN_SIF:-$HOME/ubuntu22.sif}"
ENV_PYTHON="$HEPATTN/.pixi/envs/default/bin/python"

# THE config, by absolute path into this repository. main.py runs from the hepattn checkout, whose
# own copy is the superseded Lion configuration -- a relative --config would silently train that.
CONFIG="${CONFIG:-$MIRROR/configs/calo_clustering.yaml}"

# Many dataloader workers passing large tensors can exhaust the fd limit. data.py also sets the
# file_system sharing strategy, which is the primary guard against "received 0 items of ancdata".
ulimit -n 65536 2>/dev/null || true

# The cluster's own hook. On compute-gpu-0-0 (LIGHTGPU) it rewrites CUDA_VISIBLE_DEVICES from a
# Slurm index to a MIG UUID; on compute-gpu-0-1 (GPU) it does nothing. Source it before reading
# CUDA_VISIBLE_DEVICES, or the MIG case reads a number that means nothing.
[ -f /etc/slurm/gpu_variables.sh ] && source /etc/slurm/gpu_variables.sh

preflight_paths() {
    for p in "$SIF" "$ENV_PYTHON" "$CONFIG" "$MIRROR/main.py" "$EXP_DIR"; do
        [ -e "$p" ] || { echo "ABORT: missing $p"; return 1; }
    done
    [ -d "$DATA_DIR/ttbar_pu0" ] || { echo "ABORT: no dataset at $DATA_DIR/ttbar_pu0"; return 1; }
}

# Pick a card that is (a) ours and (b) not the one that kills jobs. Exports CUDA_VISIBLE_DEVICES.
#
# THIS NODE HAS NO CGROUP DEVICE ISOLATION. `nvidia-smi` inside a job on compute-gpu-0-1 lists all
# three cards whatever was allocated -- verified in job 48244, which asked for two and saw indices
# 0, 1 and 2. So the visible device list is NOT the allocation, and choosing from it would happily
# select a card belonging to somebody else's job. The allocation only ever comes from Slurm's own
# variables, and a card outside it is never a candidate. (calo_dump_eventstore.sh chooses from the
# visible list on the assumption that a cgroup renumbers the allocation from 0. That is the right
# algorithm on a cluster that isolates devices; it is not what this one does.)
#
# The faulty card is compute-gpu-0-1 GPU 2: 844 uncorrected ECC errors as of 2026-08-05 and
# climbing (818 a week earlier), which kills jobs with "CUDA error: uncorrectable ECC error"
# minutes in -- job 48163 died at 3:30. Requesting two cards is what guarantees the allocation
# contains a healthy one: the node has three, only one is faulty, so any two must include a good
# one. Only one is ever used (trainer.devices is 1).
select_gpu() {
    local alloc ecc healthy="" first=""

    alloc="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}}"

    # MIG (LIGHTGPU): the allocation is a UUID, ECC is not queryable per instance by index, and
    # nothing here should be training on a 20 GB slice anyway.
    case "$alloc" in
        *MIG-*)
            echo "GPU: MIG instance $alloc (LIGHTGPU)."
            echo "WARNING: LIGHTGPU is the GPU node divided into 6 x 20 GB MIG instances. Dumping"
            echo "         a store fits; training this configuration does not. Expect a CUDA OOM."
            export CUDA_VISIBLE_DEVICES="$alloc"
            return 0
            ;;
    esac

    if [ -z "$alloc" ]; then
        echo "WARNING: no Slurm GPU allocation in the environment; falling back to every visible"
        echo "         card. Fine interactively, wrong under sbatch -- check the submission."
        alloc=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd,)
    fi

    for g in ${alloc//,/ }; do
        ecc=$(nvidia-smi -i "$g" --query-gpu=ecc.errors.uncorrected.aggregate.total \
              --format=csv,noheader 2>/dev/null | tr -d ' ')
        echo "Allocated GPU $g: uncorrected ECC errors = ${ecc:-unreadable}"
        first="${first:-$g}"
        case "$ecc" in
            ''|*[!0-9]*) ;;
            *) [ "$ecc" -eq 0 ] && healthy="${healthy:-$g}" ;;
        esac
    done

    if [ -z "$first" ]; then
        echo "ABORT: no GPU in this job's allocation."
        return 1
    fi
    if [ -z "$healthy" ]; then
        if [ "${ALLOW_FAULTY_GPU:-0}" != "1" ]; then
            echo "ABORT: every allocated GPU reports uncorrected ECC errors, or none could be read."
            echo "       Do not start a long run on hardware that kills jobs mid-flight. Resubmit,"
            echo "       or set ALLOW_FAULTY_GPU=1 to override."
            return 1
        fi
        echo "WARNING: proceeding on GPU $first because ALLOW_FAULTY_GPU=1."
        healthy="$first"
    fi

    export CUDA_VISIBLE_DEVICES="$healthy"
    echo "Using GPU $healthy"
    nvidia-smi -i "$healthy" --query-gpu=index,name,memory.total --format=csv,noheader || true
}

# Copy this repository's modules into the checkout before running.
#
# main.py imports hepattn.experiments.colliderml.{data,model}, which resolve to the installed
# hepattn checkout, NOT to this repository -- so passing the config by absolute path fixes what is
# configured but not what is executed. This repository is the source of record
# (src/maskformer/README.md), so it wins, automatically, every time.
#
# Plain cp, not the `cp -u` in ce_ai_1/env.sh: -u compares mtimes, and a freshly checked-out file
# can be OLDER than the stale copy it has to replace, in which case the sync silently does nothing.
# The cost of an unconditional copy is a redundant write; the cost of the other is a run that
# reports the right config and executes the wrong code.
#
# This is the only thing in these scripts that writes outside this repository.
sync_mirror() {
    if [ "${SYNC:-1}" != "1" ]; then
        echo "SYNC=0: running the checkout's own data.py/model.py (config still comes from here)."
        return 0
    fi
    mkdir -p "$EXP_DIR"/{configs,eval,scripts}
    cp "$MIRROR"/*.py           "$EXP_DIR/"
    cp "$MIRROR"/configs/*.yaml "$EXP_DIR/configs/"
    cp "$MIRROR"/eval/*.py      "$EXP_DIR/eval/"
    cp "$MIRROR"/scripts/*.py   "$EXP_DIR/scripts/" 2>/dev/null || true
    echo "Synced $MIRROR -> $EXP_DIR"
}

# calo_clustering.yaml logs to Comet online, so an unset key fails the logger rather than quietly
# falling back to offline.
#
# ON DIAS THE KEY ARRIVES BY ITSELF, and the `-l` in each script's shebang is what makes that true:
# a login shell sources ~/.bash_profile, which sources ~/.bashrc, which exports COMET_API_KEY.
# There is no need to export it before sbatch, and dropping the `-l` would break Comet logging in a
# way that presents as a Comet problem. (~/.bashrc is mode 0644 on a shared cluster, so any account
# on it can read the key -- ce_ai_1/README.md's "rotate it if it has been shared anywhere" applies.)
#
# The 0600 key file is ce-ai-1's convention, supported here only so both machines can read the same
# file if it is ever copied over.
load_comet_key() {
    COMET_ENV_FILE="${COMET_ENV_FILE:-$HOME/.config/colliderml/comet.env}"
    if [ -z "${COMET_API_KEY:-}" ] && [ -f "$COMET_ENV_FILE" ]; then
        # shellcheck disable=SC1090
        . "$COMET_ENV_FILE"
    fi
    if [ -z "${COMET_API_KEY:-}" ]; then
        echo "WARNING: COMET_API_KEY is unset. Comet logging will fail. It is normally exported by"
        echo "         ~/.bashrc -- check that this script's shebang is still '#!/bin/bash -l'."
    fi
}

# Extra --config files, in order, from this repository's configs/. Later files win on conflicts.
#   OVERLAYS="overlay_metric_aligned.yaml overlay_long_schedule.yaml"
build_config_args() {
    CONFIG_ARGS=(--config "$CONFIG")
    for overlay in ${OVERLAYS:-}; do
        local o="$MIRROR/configs/$overlay"
        [ -f "$o" ] || { echo "ABORT: no such overlay: $o"; return 1; }
        CONFIG_ARGS+=(--config "$o")
        echo "Overlay: $o"
    done
}

# ENVIRONMENT: DIAS is RHEL7 (glibc 2.17) and the pixi env needs glibc 2.28+, so everything runs
# inside an Ubuntu 22.04 container. Build the image once on the login node:
#     apptainer build ~/ubuntu22.sif docker://ubuntu:22.04
#
# TORCH_COMPILE_DISABLE: the cost functions are torch.compile'd, but the bare container has no C
# compiler for Triton's JIT, so compile is disabled and they run eager. Slower, and bullet-proof
# for a long unattended run. To keep the speedup instead, drop the two DISABLE lines and add
#     --env CC="$HEPATTN/.pixi/envs/default/bin/cc" --env CXX="$HEPATTN/.pixi/envs/default/bin/c++"
run_in_container() {
    apptainer exec --nv --bind "$DATA_DIR" --bind "$REPO" \
        --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        --env COMET_API_KEY="${COMET_API_KEY:-}" \
        --env TORCH_COMPILE_DISABLE=1 --env TORCHDYNAMO_DISABLE=1 \
        "$SIF" "$@"
}
