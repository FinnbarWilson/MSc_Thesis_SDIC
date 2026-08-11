# Shared environment for the ce-ai-1 runs. Sourced by the scripts beside it, not run directly.
#
# WHY THESE SCRIPTS ARE NOT IN hepattn_colliderml/slurm/
#
# Everything under hepattn_colliderml/ is a verbatim mirror of the hepattn checkout and is checked
# by verify_sync.sh. These are machine-specific launchers for ce-ai-1 and are mine, not upstream's,
# so putting them in the mirror would make verify_sync.sh fail against any clean checkout. They
# address the checkout the same way the slurm scripts do.
#
# WHAT CHANGED MOVING OFF DIAS
#
#   DIAS (old)                                  ce-ai-1 (here)
#   Slurm: sbatch, walltime caps, queueing      run it directly; nohup for long runs
#   RHEL7 glibc 2.17 -> apptainer container     Ubuntu 24.04, glibc 2.39: no container
#   pixi env inside the container               plain venv, system python 3.12
#   3 GPUs, one with 818 uncorrected ECC        one A100 80GB, healthy: no ECC preflight
#   --mem=128G                                  1.5 TB RAM: the host-memory ceiling is gone
#   $HOME had room                              / has ~100 GB free: everything is on the datastore

# One place defines where things live, and it is not this file.
# shellcheck disable=SC1091
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../setup" && pwd)/paths.sh"

EXP_DIR="$HEPATTN/src/hepattn/experiments/colliderml"
PYTHON="$VENV_TRAIN/bin/python"

# Re-sync the mirrored subtree into the checkout on every run.
#
# This is not tidiness, it is a correctness trap that has already bitten once. main.py runs from
# the CHECKOUT, which holds copies made at install time -- so editing
# src/maskformer/hepattn_colliderml/configs/overlay_pu200_barrel.yaml in the repository and launching a
# run silently uses the stale copy. The symptom is a config change that appears to do nothing
# (observed: max_epochs edited to 9, --print_config still reporting 3). The repository is the
# source of record, so it wins here, every time, automatically.
if [ -d "$EXP_DIR" ]; then
    _mirror="$REPO_ROOT/src/maskformer/hepattn_colliderml"
    cp -u "$_mirror"/*.py                "$EXP_DIR/"          2>/dev/null || true
    cp -u "$_mirror"/configs/*.yaml       "$EXP_DIR/configs/"  2>/dev/null || true
    cp -u "$_mirror"/eval/*.py            "$EXP_DIR/eval/"     2>/dev/null || true
    cp -u "$_mirror"/scripts/*.py         "$EXP_DIR/scripts/"  2>/dev/null || true
    unset _mirror
fi

# Comet key. A 0600 file in the user's own config directory: outside the git worktree so it cannot
# be committed, and not on the shared datastore, which holds the dataset and nothing else.
# calo_clustering.yaml already reads the key from the environment rather than inline; this just
# supplies it. Rotate it at comet.com if it has been shared anywhere.
if [ -f "$COMET_ENV_FILE" ]; then
    # shellcheck disable=SC1091
    . "$COMET_ENV_FILE"
else
    echo "WARNING: $COMET_ENV_FILE not found; Comet logging will fail." >&2
fi

# Many dataloader workers passing large tensors can exhaust the fd limit. data.py also sets the
# file_system sharing strategy, which is the primary guard against "received 0 items of ancdata".
ulimit -n 65536 2>/dev/null || true

# pu200 runs the card hot: the benchmark peaked at 68,456 MiB of 81,037, i.e. 84% with ~12 GiB
# spare. Event sizes vary a lot (measured 117k hits/event mean against a 151k max, +29%), and the
# sequence length changes every step, so the allocator sees a different shape each time and
# fragments -- the failure mode is an OOM on a busy event hours in, with plenty of free memory that
# is in the wrong-sized blocks. expandable_segments lets it grow segments instead of hoarding
# fixed-size ones. If a long run still OOMs, drop num_queries to 400 (measured max targets: 368)
# before touching anything else.
# torch 2.9 renamed this: it warns "PYTORCH_CUDA_ALLOC_CONF is deprecated, use PYTORCH_ALLOC_CONF
# instead". Set the new name, and keep the old one for anything older.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The cost functions are torch.compile'd. The venv has a real gcc (unlike the bare container on
# DIAS), so Triton can JIT and we keep the speedup. Set TORCH_COMPILE_DISABLE=1 to fall back to
# eager if a long run ever dies inside dynamo.
: "${TORCH_COMPILE_DISABLE:=}"
export TORCH_COMPILE_DISABLE

# Sanity: the GPU here is a single MIG 7g.80gb instance, which is the whole card. Nothing to select
# between, so no CUDA_VISIBLE_DEVICES juggling and no ECC preflight -- both existed on DIAS only
# because one of its three cards was faulty.

sample_gpu_peak() {
    # Peak GPU memory of our own process, sampled from nvidia-smi. Under MIG the per-GPU
    # memory.used query does not report instance usage, so query the compute apps instead.
    local out=$1 pid=$2 peak=0 cur
    while kill -0 "$pid" 2>/dev/null; do
        cur=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
              | sort -rn | head -1)
        [ -n "$cur" ] && [ "$cur" -gt "$peak" ] 2>/dev/null && peak=$cur
        sleep 2
    done
    echo "$peak" > "$out"
}
