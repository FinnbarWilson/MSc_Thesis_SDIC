#!/usr/bin/env bash
# Measure pu200 throughput and peak GPU memory, then print the max_epochs that fits a target run.
#
#   ./benchmark_pu200.sh            # 200 events, ~15 min
#   EVENTS=400 TARGET_HOURS=22 ./benchmark_pu200.sh
#
# RUN THIS BEFORE THE REAL TRAINING RUN. It answers the two questions the pu200 overlay cannot
# answer from pu0 measurements:
#
#   1. Does it fit? The overlay's cuts put the mask-logit footprint at 2.66x pu0, and pu0 OOMed at
#      4x -- so it should, but "should" is not "does", and finding out 30 seconds in beats finding
#      out at hour six. Peak GPU memory is reported below.
#   2. How fast is it? overlay_pu200.yaml sizes max_epochs from an ESTIMATE of 0.25 events/s,
#      extrapolated from pu0's measured 1.13. OneCycleLR is sized from total steps, so a wrong
#      estimate does not just mean a run of the wrong length -- it means a run whose final
#      checkpoint sits at a high learning rate, which is how the hit-filter run was wasted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

EVENTS="${EVENTS:-200}"
TARGET_HOURS="${TARGET_HOURS:-22}"
NUM_TRAIN="${NUM_TRAIN:-6000}"   # must match overlay_pu200.yaml for the epoch arithmetic below

cd "$EXP_DIR"
PEAK_FILE=$(mktemp)
LOGFILE=$(mktemp)

echo "=== pu200 benchmark: $EVENTS events, validation disabled ==="
START=$(date +%s)

"$PYTHON" main.py fit \
    --config configs/calo_clustering.yaml \
    --config "configs/${OVERLAY:-overlay_pu200_barrel.yaml}" \
    --data.num_train "$EVENTS" \
    --trainer.max_epochs 1 \
    --trainer.limit_val_batches 0 \
    --trainer.max_time null \
    --trainer.logger.init_args.online false \
    --data.pin_memory false 2>&1 | tee "$LOGFILE" &
RUN_PID=$!
sample_gpu_peak "$PEAK_FILE" "$RUN_PID" &
SAMPLER=$!

set +e
wait "$RUN_PID"; RC=$?
set -e
wait "$SAMPLER" 2>/dev/null || true
END=$(date +%s)

PEAK=$(cat "$PEAK_FILE" 2>/dev/null || echo 0); rm -f "$PEAK_FILE"
ELAPSED=$((END - START))

echo
echo "=================== result ==================="
if [ "$RC" -ne 0 ]; then
    echo "RUN FAILED (exit $RC)."
    echo "If this was a CUDA OOM, apply the ladder in the header of configs/overlay_pu200.yaml:"
    echo "  1. calohit_min_energy 1e-3 -> 2e-3   (117k -> 43k hits/event)"
    echo "  2. num_queries + event_max_num_particles 500 -> 400"
    exit "$RC"
fi

echo "wall time      : ${ELAPSED}s for $EVENTS events (includes ~2-4 min of startup)"
echo "peak GPU memory: ${PEAK} MiB of 81037 MiB"

# Lightning's own progress bar is the authority on the training rate, not the wall clock. An
# earlier version of this script subtracted a flat 180 s "startup" from the wall time, which on a
# 223 s benchmark left 43 s and overstated the rate by ~5x (4.65 events/s against a real 0.98).
# Take the rate Lightning measured over the training loop instead; \r-separated bar updates mean
# the last one has to be pulled out of the carriage returns.
RATE=$(tr '\r' '\n' < "$LOGFILE" | grep -oE "[0-9]+\.[0-9]+it/s" | tail -1 | sed 's/it\/s//')

python3 - "$EVENTS" "$ELAPSED" "$TARGET_HOURS" "$NUM_TRAIN" "${RATE:-0}" <<'PY'
import sys
events, elapsed, target_h, num_train = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rate = float(sys.argv[5])
if rate <= 0:
    print("\nCould not read a rate from Lightning's progress bar; size max_epochs by hand.")
    raise SystemExit(0)
print(f"\nmeasured rate  : {rate:.2f} events/s from Lightning  (pu0 measured 1.13 at 22k hits/event)")

# THE CACHE CORRECTION, and why this benchmark cannot be trusted at face value.
# data.py caches 8 decoded row groups per worker, and there is one row group per shard. This
# benchmark touches EVENTS/100 shards, so with the default 200 events the whole working set is 2
# shards and stays cached -- a hit rate the real run never sees. calo_clustering.yaml measured
# exactly this at pu0: 1.89 events/s at num_train=3000 fell to 1.42 at num_train=20000 (-25%)
# purely from cache misses, and sizing from the optimistic figure would have turned a "39 h" run
# into 53 h and had it killed mid-decay.
bench_shards, real_shards = max(events // 100, 1), max(num_train // 100, 1)
if real_shards > bench_shards:
    rate *= 0.75
    print(f"                 benchmark saw {bench_shards} shards, the real run sees {real_shards};")
    print(f"                 de-rated 25% for parquet cache misses -> {rate:.2f} events/s")

# Validation: val_check_interval 0.25 x num_val 250 = 1000 val events per epoch, no backward pass,
# so roughly 3x training speed.
val_s = 1000 / (rate * 3)
epoch_s = num_train / rate + val_s
print(f"epoch estimate : {epoch_s/3600:.2f} h at num_train={num_train} (incl. ~{val_s/60:.0f} min validation)")
fit = max(int(target_h * 3600 // epoch_s), 1)
print(f"\n--> for a ~{target_h:.0f} h run set  trainer.max_epochs: {fit}   ({fit*epoch_s/3600:.1f} h)")
print("    Set it in configs/overlay_pu200.yaml. Do NOT let a longer schedule be truncated:")
print("    OneCycleLR is sized from total steps and a truncated run ends at a high LR.")
PY
