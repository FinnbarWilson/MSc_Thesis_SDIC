#!/usr/bin/env bash
# The mask-head variant comparison: control plus the three variants, SEQUENTIALLY on the one A100.
#
#   nohup ./run_variants.sh > ../../../external/run_variants.log 2>&1 &
#
# Instructions and evidence: ./VARIANTS_HANDOVER.md. Schedule and sizing:
# ../hepattn_colliderml/configs/overlay_variants_short.yaml. Read both before changing anything.
#
# WHY FOUR RUNS AND NOT THREE. The handover (section 5) requires a control on the same dataset. The
# completed 2-epoch barrel run cannot serve as one -- it is 12,000 steps against these 4,000 -- and
# neither can its epoch-0 checkpoint, because OneCycleLR is sized from TOTAL steps, so at step 6,000
# of a 12,000-step schedule it sat mid-schedule at a high learning rate while a 4,000-step run has
# finished decaying. Same step count, different optimiser state. The control is re-run here so all
# four share a schedule and differ only in the variant switch.
#
# WHY SEQUENTIAL. One A100, and the barrel config peaks around 60 GB of 81 with another user
# typically holding a few GB. Two concurrent runs OOM.
#
# ORDER. Control first, so a broken baseline is caught before three variants are spent against it.
# v3 last: it is the only arm that alters the forward pass, and the only one whose per-step cost is
# unknown (it rebuilds an 8-NN graph over ~54k cells every forward).
#
# THE OBJECTIVE THE VARIANTS SIT ON. All four use pu200's dice 20 + focal 1, NOT the dice 1 + bce 1
# the overlay_v*.yaml files originally carried. Those were written on DIAS where that is the pu0
# objective; on pu200 barrel it is measured to collapse. See the note in each overlay.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

ARMS=(
    "control:"
    "v1_coverage:overlay_v1_coverage.yaml"
    "v2_recall:overlay_v2_recall.yaml"
    "v3_propagation:overlay_v3_propagation.yaml"
)

LOGROOT="$EXP_DIR/logs"
RESULTS="$REPO_ROOT/external/variants_results.txt"
: > "$RESULTS"
cd "$EXP_DIR"

for entry in "${ARMS[@]}"; do
    name=${entry%%:*}
    overlay=${entry#*:}

    echo ""
    echo "================================================================"
    echo "ARM $name   started $(date)"
    echo "================================================================"

    # overlay_variants_short.yaml LAST so its schedule wins over anything an arm sets.
    stack="overlay_pu200_barrel.yaml"
    [ -n "$overlay" ] && stack="$stack $overlay"
    stack="$stack overlay_variants_short.yaml"

    before=$(ls -1d "$LOGROOT"/ColliderML_Calo_Clustering_* 2>/dev/null | wc -l)
    OVERLAYS="$stack" "$HERE/train_pu200.sh"
    status=$?
    after=$(ls -1d "$LOGROOT"/ColliderML_Calo_Clustering_* 2>/dev/null | wc -l)
    rundir=$(ls -1dt "$LOGROOT"/ColliderML_Calo_Clustering_* 2>/dev/null | head -1)

    if [ "$status" -ne 0 ] || [ "$after" -le "$before" ]; then
        echo "ARM $name FAILED (exit $status)" | tee -a "$RESULTS"
        continue
    fi

    echo "--- $name  $rundir" | tee -a "$RESULTS"
    # Collapse check first: if the mask head died, every downstream number is 0 by construction and
    # the arm says nothing about its own hypothesis.
    "$PYTHON" "$REPO_ROOT/src/maskformer/hepattn_colliderml/eval/diagnose_mask.py" \
        "$rundir" --events 2 2>&1 | grep -E "^  (mask prob|cells|queries|flow_valid|VERDICT)" | tee -a "$RESULTS"
done

echo ""
echo "================================================================"
echo "ALL ARMS DONE $(date)"
cat "$RESULTS"
echo ""
echo "NEXT: efficiency/purity and the cluster-size slope, which is what section 5 says to judge on."
echo "  dias/compare_probes.py  (slope column)  on dumped stores"
