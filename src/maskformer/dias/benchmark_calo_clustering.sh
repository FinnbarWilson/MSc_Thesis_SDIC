#!/bin/bash -l
#SBATCH --job-name=calo_bench
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:2
# Matched to train_calo_clustering.sh on purpose: a benchmark under a different allocation measures
# a different job. Override both together when probing an alternative, which is how the comparison
# in that file's header was produced:
#   MINUTES=10 WORKERS=12 sbatch --cpus-per-task=12 --mem=128G .../benchmark_calo_clustering.sh
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:35:00
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_bench_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_bench_%j.err

# Measure this configuration's real throughput on DIAS, then print the walltime a full run needs
# and whether the resources being asked for are the right ones.
#
#   sbatch src/maskformer/dias/benchmark_calo_clustering.sh
#   MINUTES=20 WORKERS=16 sbatch src/maskformer/dias/benchmark_calo_clustering.sh
#
# RUN THIS BEFORE train_calo_clustering.sh, and re-run it whenever the config's per-step cost
# changes. OneCycleLR is sized from TOTAL optimiser steps, so a run that overruns its walltime
# never reaches its decay phase and its final checkpoint is taken at a high learning rate -- the
# waste ../ce_ai_1/PU200_STATUS.md §6 describes. Sizing the wall from a guess is how it happens.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS RUNS AT THE FULL num_train RATHER THAN ON A SMALL SUBSET
#
# ce_ai_1/benchmark_pu200.sh benchmarks 200 events and then de-rates the result by 25% for parquet
# cache misses: data.py caches 8 decoded row groups per worker, there is one row group per shard,
# and a 200-event benchmark touches 2 shards -- a hit rate the real run never sees. That correction
# was measured at -25% at pu0 and turned out to be -41% at pu200, i.e. the correction itself needed
# correcting.
#
# It is not needed here. `shuffle=True` on the training dataloader means a run at the real
# num_train=20000 draws from all 200 shards from the first batch, so capping by TIME rather than by
# event count gives the real cache-miss rate directly, with no extrapolation. What it cannot see is
# slow drift over hours, so the rate over the last third of the window is reported separately.
#
# ---------------------------------------------------------------------------------------------
# WHAT IT REPORTS, AND WHAT EACH ANSWER MEANS
#
#   rate            events/s from Lightning's own progress bar, overall and over the last third.
#                   The overall figure is dragged down by startup -- prefer the last third.
#   peak GPU memory of 81920 MiB on this partition's cards. Headroom is the batch_size question,
#                   which calo_clustering.yaml couples to the learning rate -- see the note in
#                   train_calo_clustering.sh before acting on it.
#   mean GPU util   THE ANSWER TO "ARE WE ASKING FOR THE RIGHT THING". High (>85%) means GPU-bound
#                   and more CPUs buy nothing. Low means the card is waiting on the dataloader, and
#                   --cpus-per-task with data.num_workers to match is what shortens the run.
#   host RSS        printed by `sacct -j <id> --format=MaxRSS` after the job, not here. It scales
#                   with WORKERS (8 cached row groups each), so it is the ceiling on raising them.
# ---------------------------------------------------------------------------------------------

set -euo pipefail

# Hardcoded: Slurm copies this script into a spool directory, so ${BASH_SOURCE[0]} is "/" here.
# shellcheck disable=SC1091
. /home/xucapfwi/MSc_Thesis_SDIC/src/maskformer/dias/env.sh

OUT_DIR="${OUT_DIR:-$REPO/external/logs_bench}"
MINUTES="${MINUTES:-12}"                 # of TRAINING, after startup
NUM_TRAIN="${NUM_TRAIN:-20000}"          # the real value: this is what sets the shard working set
MAX_EPOCHS="${MAX_EPOCHS:-4}"            # only used for the "full run" arithmetic at the end
WORKERS="${WORKERS:-24}"                 # matched to train_calo_clustering.sh
MAX_TIME=$(printf "00:00:%02d:00" "$MINUTES")

echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Probe:   ${MINUTES} min of training at num_train=$NUM_TRAIN, $WORKERS workers, validation off"
echo "Alloc:   ${SLURM_CPUS_PER_TASK:-?} CPUs, ${SLURM_MEM_PER_NODE:-?} MB requested"
echo "Config:  $CONFIG"

preflight_paths
mkdir -p "$OUT_DIR"
select_gpu
sync_mirror
build_config_args

cd "$EXP_DIR"
LOGFILE="$OUT_DIR/bench_${SLURM_JOB_ID:-manual}.log"
SAMPLES="$OUT_DIR/bench_${SLURM_JOB_ID:-manual}.gpu.csv"

# Sample the card we were given. This partition is not exclusive, so another job on the same card
# would inflate memory.used; utilization.gpu is device-wide by nature. Both are read as trends, not
# as precise per-process numbers.
( while true; do
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits >> "$SAMPLES" 2>/dev/null || true
    sleep 5
  done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

# Comet offline: a benchmark is not a result and does not belong in the project the thesis figures
# are traced to. Validation off so the number measured is the training rate alone; the arithmetic
# at the bottom adds validation back explicitly.
set +e
run_in_container "$ENV_PYTHON" main.py fit \
  "${CONFIG_ARGS[@]}" \
  --trainer.default_root_dir "$OUT_DIR" \
  --trainer.max_time "$MAX_TIME" \
  --trainer.max_epochs "$MAX_EPOCHS" \
  --trainer.limit_val_batches 0 \
  --trainer.logger.init_args.online false \
  --data.num_train "$NUM_TRAIN" \
  --data.num_workers "$WORKERS" \
  --data.pin_memory false 2>&1 | tee "$LOGFILE"
RC=${PIPESTATUS[0]}
set -e
kill "$SAMPLER" 2>/dev/null || true

echo
echo "=============================== result ==============================="
if [ "$RC" -ne 0 ]; then
  echo "RUN FAILED (exit $RC). The log is $LOGFILE"
  echo "A CUDA OOM here means the configuration does not fit at all, and the full run would have"
  echo "died the same way an hour in -- which is what this probe is for."
  exit "$RC"
fi

# Parsed inside the container: the pixi interpreter is built against glibc 2.28+ and cannot run on
# RHEL7 directly, and the host's python3 is whatever conda put on PATH, which is not something a
# batch job should depend on.
apptainer exec --bind "$REPO" "$SIF" \
  "$ENV_PYTHON" - "$LOGFILE" "$SAMPLES" "$NUM_TRAIN" "$MAX_EPOCHS" "$WORKERS" <<'PY'
import re, sys, statistics

log, samples, num_train, max_epochs, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

# Lightning's progress bar is the authority on the rate, not the wall clock: the wall includes
# imports, the dataset scan and cuda init, and ce_ai_1/benchmark_pu200.sh records that subtracting
# a guessed startup overstated the rate by 5x. Bar updates are \r-separated, so split on those and
# take (elapsed, step) pairs -- the bar's clock starts at the training loop.
text = open(log, errors="replace").read().replace("\r", "\n")
pts = []
for m in re.finditer(r"(\d+)/(\d+)\s+\[(\d+):(\d\d)(?::(\d\d))?<", text):
    a, b, c = m.group(3), m.group(4), m.group(5)
    secs = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    pts.append((secs, int(m.group(1))))
pts = sorted(set(pts))

if len(pts) < 3:
    print("Could not read the progress bar; size the walltime by hand from", log)
    raise SystemExit(0)

(t0, s0), (t1, s1) = pts[0], pts[-1]
overall = (s1 - s0) / (t1 - t0) if t1 > t0 else 0.0

# The last third, to expose drift. PU200_STATUS.md §6 saw the rate DECLINE through a run as the
# working set outgrew data.py's row-group cache; if that is happening here, the overall figure is
# optimistic and this one is what to size from.
cut = t0 + 2 * (t1 - t0) / 3
tail = [p for p in pts if p[0] >= cut]
recent = (tail[-1][1] - tail[0][1]) / (tail[-1][0] - tail[0][0]) if len(tail) > 1 and tail[-1][0] > tail[0][0] else overall

print(f"steps measured : {s1 - s0} over {t1 - t0}s of training loop ({s1} reached of {num_train})")
print(f"rate           : {overall:.2f} events/s overall, {recent:.2f} events/s over the last third")
if overall > 0 and recent < 0.9 * overall:
    print("                 DECLINING -- the working set is outgrowing data.py's row-group cache.")
    print("                 Size from the last-third figure, and expect it to fall further.")

utils, mems = [], []
try:
    for line in open(samples):
        u, m = line.split(",")
        utils.append(float(u)); mems.append(float(m))
except Exception:
    pass
if utils:
    warm = utils[len(utils) // 4:]  # drop startup, when the card is idle
    mean_u = statistics.mean(warm)
    print(f"peak GPU memory: {max(mems):.0f} MiB of 81920 MiB ({max(mems)/81920*100:.0f}%)")
    print(f"mean GPU util  : {mean_u:.0f}% over the training loop, {workers} dataloader workers")
    if mean_u < 70:
        print("                 INPUT-BOUND. The card is idle waiting for data, so the lever that")
        print("                 shortens this run is --cpus-per-task and data.num_workers to match.")
        print("                 Raise both and re-run -- but check MaxRSS after: resident memory")
        print("                 scales with workers (8 cached row groups each), and on this node")
        print("                 that ceiling arrives before the CPU one does.")
    elif mean_u > 85:
        print("                 GPU-BOUND. More CPUs would buy nothing; the walltime is the run.")

rate = recent if recent > 0 else overall
val_s = 1000 / (rate * 3)         # val_check_interval 0.25 x num_val 250, no backward pass
epoch_s = num_train / rate + val_s
total_h = max_epochs * epoch_s / 3600
print()
print(f"epoch          : {epoch_s/3600:.2f} h at num_train={num_train} (incl. ~{val_s/60:.0f} min validation)")
print(f"full run       : {total_h:.1f} h at max_epochs={max_epochs}")
print(f"--> request    : #SBATCH --time={int(total_h*1.35)+1:02d}:00:00   (35% margin)")
print(f"    and        : MAX_TIME=\"00:{int(total_h*1.35):02d}:00:00\" as the runaway guard")
print()
print("Then check host memory:  sacct -j $SLURM_JOB_ID --format=JobID,ReqMem,MaxRSS")
print("--mem is NOT enforced on this cluster (task/affinity, no cgroups), so a job that exceeds it")
print("runs anyway and quietly over-books the node for everyone else. Ask for what it measures at.")
PY
echo "Finished: $(date)"
