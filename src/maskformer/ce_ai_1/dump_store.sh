#!/usr/bin/env bash
# Dump an event store on ce-ai-1.
#
# This is the step that makes the comparison controlled: CLUE does not read ColliderML, it reads
# the store written here from the MODEL's own dataloader, so every cut the training config applied
# reaches both methods by construction rather than by two configs agreeing.
#
# Dump both stores after training, from the windows the training config leaves free:
#
#   CKPT=<ckpt> ./dump_store.sh pu0   tune    # events [20000, 20050),  50 events
#   CKPT=<ckpt> ./dump_store.sh pu0   eval    # events [20250, 20750), 500 events
#   CKPT=<ckpt> ./dump_store.sh pu200 tune    # events [7000, 7050),    50 events
#   CKPT=<ckpt> ./dump_store.sh pu200 eval    # events [7500, 8000),   500 events
#
# The tune window is the training run's own validation split: unseen by the optimiser, seen by
# model selection, which is the same status CLUE's tuning window has. Neither method picks its
# working point on the reported events.
#
# Then set dataset.<pu>.store / .tune_store in config/experiment.yaml to match, and check with
# `python -m scripts.show_config`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

: "${CKPT:?Set CKPT=/path/to/your/checkpoint.ckpt}"
DATASET="${1:-}"
WHICH="${2:-eval}"

case "$DATASET:$WHICH" in
    pu0:tune)    START=20000; NUM=50  ;;
    pu0:eval)    START=20250; NUM=500 ;;
    pu200:tune)  START=7000;  NUM=50  ;;
    pu200:eval)  START=7500;  NUM=500 ;;
    *) echo "usage: $0 [pu0|pu200] [tune|eval]   (got '${DATASET:-}' '${WHICH:-}')" >&2; exit 2 ;;
esac

# The store name encodes the EVENT WINDOW, not the pileup condition, so a pu0 and a pu200 dump of
# the same range would collide silently, hence the condition in the directory name.
OUT="${OUT:-$EXTERNAL/eventstores}"

# CHUNK is the memory knob: the dump holds a whole chunk in memory while writing it. A pu200 event
# carries ~117k cells against pu0's ~22k, so it needs a smaller chunk for the same footprint.
# Lower it further if the dump is killed.
if [ "$DATASET" = "pu200" ]; then CHUNK="${CHUNK:-10}"; else CHUNK="${CHUNK:-25}"; fi

# INCIDENCE_TOP_K is deliberately not set. eval/format.py carries the measured value (16) and is
# the one place that choice is justified; setting it here silently overrides it, which has happened
# once before (4 against format's 16).

echo "checkpoint : $CKPT"
echo "dataset    : $DATASET"
echo "window     : [$START, $((START + NUM)))  ($WHICH store)"
echo "out        : $OUT"
echo "chunk      : $CHUNK events per .npz"
mkdir -p "$OUT"
df -h "$OUT" | tail -1

cd "$EXP_DIR"
exec "$PYTHON" -m hepattn.experiments.colliderml.eval.dump "$CKPT" \
    --start-event "$START" --num-events "$NUM" --out "$OUT" \
    --chunk-size "$CHUNK" --num-workers 8
