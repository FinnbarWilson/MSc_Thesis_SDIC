#!/usr/bin/env bash
# Dump pu200 event stores. Replaces slurm/calo_dump_eventstore.sh on ce-ai-1.
#
# This is the step that makes the comparison controlled: CLUE does not read ColliderML, it reads
# the store written here from the MODEL's own dataloader. Whatever cuts overlay_pu200_barrel.yaml applies
# (calohit_min_energy 1e-3, particle_min_pt 2.0) therefore apply to both methods by construction.
#
# Dump BOTH stores after training, from the windows overlay_pu200_barrel.yaml leaves free:
#
#   CKPT=<your pu200 ckpt> ./dump_store_pu200.sh tune    # events [7000, 7050),  50 events
#   CKPT=<your pu200 ckpt> ./dump_store_pu200.sh eval    # events [7500, 8000), 500 events
#
# Then set dataset.pu200.store / .tune_store / .windows in config/experiment.yaml to match, and
# check with `python -m scripts.show_config`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

: "${CKPT:?Set CKPT=/path/to/your/pu200/checkpoint.ckpt}"
WHICH="${1:-eval}"

case "$WHICH" in
    tune) START=7000; NUM=50  ;;
    eval) START=7500; NUM=500 ;;
    *) echo "usage: $0 [tune|eval]   (got '$WHICH')" >&2; exit 2 ;;
esac

# OUT must differ from the pu0 stores. The store name encodes the EVENT WINDOW, not the pileup
# condition, so a pu0 and a pu200 dump of the same range would collide silently -- hence pu200 in
# the directory name rather than trusting the window to disambiguate.
OUT="${OUT:-/mnt/ai-datastore/finnbar/eventstore_pu200}"

# 10, down from the default 25. CHUNK is the memory knob: the dump holds a whole chunk in memory
# while writing it, and a pu200 event carries ~117k cells against ~22k at pu0, so a pu0-sized chunk
# is ~5x the resident footprint it was sized for. Lower this further if the dump is killed.
CHUNK="${CHUNK:-10}"

# INCIDENCE_TOP_K is deliberately NOT set. eval/format.py carries the measured value (16) and is
# the one place that choice is justified; setting it here silently overrides it, which has happened
# once before (4 against format's 16). Leave it alone unless the multi-owner study says the
# truncation is binding.

echo "checkpoint : $CKPT"
echo "window     : [$START, $((START + NUM)))  ($WHICH store)"
echo "out        : $OUT"
echo "chunk      : $CHUNK events per .npz"
mkdir -p "$OUT"
df -h "$OUT" | tail -1

cd "$EXP_DIR"
exec "$PYTHON" -m hepattn.experiments.colliderml.eval.dump "$CKPT" \
    --start-event "$START" --num-events "$NUM" --out "$OUT" \
    --chunk-size "$CHUNK" --num-workers 8
