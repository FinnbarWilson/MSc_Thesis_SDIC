#!/usr/bin/env bash
# Dump pu0 event stores on ce-ai-1. The pu0 counterpart of dump_store_pu0's sibling
# dump_store_pu200.sh, and a separate file for the same reason train_pu0.sh is: the windows and
# the chunk size differ, and a shared script would need an "if pu0" on most of its lines.
#
# This is the step that makes the comparison controlled: CLUE does not read ColliderML, it reads
# the store written here from the MODEL's own dataloader. Every cut the training config applied --
# including particle_collapse_shower_secondaries, the shower-level truth definition -- therefore
# applies to both methods by construction rather than by two configs agreeing.
#
# Dump BOTH stores, from the windows calo_clustering.yaml leaves free
# (train [0, 20000), val [20000, 20250), test [20250, 20750)):
#
#   CKPT=<your pu0 ckpt> ./dump_store_pu0.sh tune    # events [20000, 20050),  50 events
#   CKPT=<your pu0 ckpt> ./dump_store_pu0.sh eval    # events [20250, 20750), 500 events
#
# The tune window is the training run's own validation split: unseen by the optimiser, but seen by
# model selection. Same status as CLUE's tuning window, which is the point -- both methods pick
# their working point on the same events and neither picks it on the reported ones.
#
# Then set dataset.pu0.store / .tune_store in config/experiment.yaml to match, and check with
# `python -m scripts.show_config`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/env.sh"

: "${CKPT:?Set CKPT=/path/to/your/pu0/checkpoint.ckpt}"
WHICH="${1:-eval}"

case "$WHICH" in
    tune) START=20000; NUM=50  ;;
    eval) START=20250; NUM=500 ;;
    *) echo "usage: $0 [tune|eval]   (got '$WHICH')" >&2; exit 2 ;;
esac

OUT="${OUT:-/mnt/ai-datastore/finnbar/eventstore_pu0}"

# 25, the dump's own default, unlike pu200's 10. A pu0 event carries ~22k cells against pu200's
# ~117k, so the chunk that had to shrink there costs a fifth as much here. Lower it if the dump
# is killed for memory.
CHUNK="${CHUNK:-25}"

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
