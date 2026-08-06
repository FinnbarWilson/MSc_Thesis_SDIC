#!/usr/bin/env bash
# Mask-objective sweep at pu200 barrel. Three ~2 h arms, run SEQUENTIALLY on the one A100.
#
#   nohup ./sweep_mask_loss.sh > ../../../external/sweep_mask_loss.log 2>&1 &
#
# WHY THIS EXISTS. Two mask objectives have been trained to convergence at pu200 and both collapsed
# to predicting the target prior everywhere. The rationale, the measurements behind it and the
# success criterion are in ../hepattn_colliderml/configs/overlay_sweep_short.yaml; the arms are in
# the sweep_*.yaml beside it. This script only sequences them and then measures the result the same
# way for each, so the comparison is not done by eye off three Comet tabs.
#
# WHAT IT MEASURES. Not val/loss -- that is not comparable across arms with different loss weights.
# It runs eval/diagnose_mask.py on each arm's final checkpoint and reports max mask probability and
# the count of cells above 0.5. An arm with any cells above 0.5 has escaped the collapse.
#
# HOST MEMORY, THE ONE THING TO WATCH. data.py hardcodes `_row_group_cache_size = 8` per worker and
# a decoded pu200 shard is 4.0 GB, so RSS scales as workers x 8 x 4 GB. The 8-worker run measured
# 229 GB; overlay_sweep_short.yaml uses 16, so expect ~450 GB peak on a 1.5 TB box that other people
# share. If that is too much, the fix is NOT fewer workers -- it is lowering that cache size in
# data.py so worker count and RSS stop being locked together. Left alone here because changing it
# mid-sweep would make the arms incomparable.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

ARMS=(sweep_a_dice_dominant sweep_b_dice_only sweep_c_focal_control)
LOGROOT="$EXP_DIR/logs"
RESULTS="$REPO_ROOT/external/sweep_mask_loss_results.txt"
: > "$RESULTS"

cd "$EXP_DIR"

for arm in "${ARMS[@]}"; do
    echo ""
    echo "================================================================"
    echo "ARM $arm   started $(date)"
    echo "================================================================"

    before=$(ls -1d "$LOGROOT"/ColliderML_Calo_Clustering_* 2>/dev/null | wc -l)

    "$PYTHON" main.py fit \
        --config configs/calo_clustering.yaml \
        --config configs/overlay_pu200_barrel.yaml \
        --config configs/overlay_sweep_short.yaml \
        --config "configs/${arm}.yaml" \
        --data.pin_memory false
    status=$?

    # Newest run directory, which is this arm's -- the CLI timestamps each one.
    rundir=$(ls -1dt "$LOGROOT"/ColliderML_Calo_Clustering_* 2>/dev/null | head -1)
    after=$(ls -1d "$LOGROOT"/ColliderML_Calo_Clustering_* 2>/dev/null | wc -l)

    if [ "$status" -ne 0 ] || [ "$after" -le "$before" ]; then
        echo "ARM $arm FAILED (exit $status) -- see above" | tee -a "$RESULTS"
        continue
    fi

    echo "--- $arm: $rundir" | tee -a "$RESULTS"
    "$PYTHON" "$REPO_ROOT/src/maskformer/hepattn_colliderml/eval/diagnose_mask.py" \
        "$rundir" --events 2 2>&1 | grep -E "^  |^---" | tee -a "$RESULTS"
done

echo ""
echo "================================================================"
echo "SWEEP COMPLETE $(date).  Summary:"
cat "$RESULTS"
