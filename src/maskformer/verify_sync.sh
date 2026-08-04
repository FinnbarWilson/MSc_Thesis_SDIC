#!/bin/bash
# Check that the copies in hepattn_colliderml/ still match the hepattn working tree they came from.
#
# Those files are duplicated rather than imported, which buys the dependency boundary described
# in README.md but costs the usual price of a copy: it can drift. During development the
# authoritative version is the one in hepattn, since that is where the code actually runs, so
# this script exists to make drift visible instead of discovering it at submission.
#
#   ./verify_sync.sh                      # against the default checkout
#   HEPATTN=/path/to/hepattn ./verify_sync.sh
#
# Exit status is 0 when everything matches, 1 otherwise, so it can go in a pre-submission check.
#
# The mirrored files live one directory down, in hepattn_colliderml/, and nothing else does.
# That is the whole reason for the extra level: `ls` answers "which of these files must stay
# byte-identical to upstream" without anyone having to remember the list below.

set -uo pipefail

HEPATTN="${HEPATTN:-$HOME/hepattn}"
SRC="$HEPATTN/src/hepattn/experiments/colliderml"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERE="$ROOT/hepattn_colliderml"

if [ ! -d "$SRC" ]; then
  echo "No hepattn experiment directory at $SRC"
  echo "Set HEPATTN=/path/to/hepattn, or ignore this if you only have the thesis repository."
  exit 1
fi

status=0

check() {
  local rel="$1" src="$2"
  if [ ! -f "$src" ]; then
    echo "  MISSING UPSTREAM  $rel"
    status=1
  elif ! diff -q "$HERE/$rel" "$src" >/dev/null 2>&1; then
    echo "  DIFFERS           $rel"
    status=1
  else
    echo "  ok                $rel"
  fi
}

echo "Comparing $HERE against $SRC"

for f in data.py model.py main.py; do
  check "$f" "$SRC/$f"
done
for f in calo_clustering.yaml overlay_metric_aligned.yaml overlay_long_schedule.yaml; do
  check "configs/$f" "$SRC/configs/$f"
done
for f in __init__.py dump.py format.py geometry.py; do
  check "eval/$f" "$SRC/eval/$f"
done
check "scripts/sweep_pred_threshold.py" "$SRC/scripts/sweep_pred_threshold.py"
for f in calo_clustering.sh calo_dump_eventstore.sh; do
  check "slurm/$f" "$SRC/slurm/$f"
done

# The patch is regenerated rather than diffed: it is a view of the upstream working tree, not a
# copy of a file, so "has it drifted" means "does it still describe the current changes".
echo
if git -C "$HEPATTN" diff --quiet src/hepattn/models/loss.py src/hepattn/models/task.py \
     src/hepattn/callbacks/prediction_writer.py 2>/dev/null; then
  echo "  NOTE: hepattn has no uncommitted changes to the patched files."
  echo "        Either they were committed upstream, or the patch is stale."
  status=1
elif git -C "$HEPATTN" diff src/hepattn/models/loss.py src/hepattn/models/task.py \
       src/hepattn/callbacks/prediction_writer.py 2>/dev/null \
     | diff -q - "$ROOT/hepattn-changes.patch" >/dev/null 2>&1; then
  echo "  ok                hepattn-changes.patch"
else
  echo "  DIFFERS           hepattn-changes.patch  (regenerate: see README)"
  status=1
fi

echo
if [ "$status" -eq 0 ]; then
  echo "In sync."
else
  echo "Out of sync. Re-copy the files listed above, or regenerate the patch with:"
  echo "  git -C $HEPATTN diff src/hepattn/models/loss.py src/hepattn/models/task.py \\"
  echo "      src/hepattn/callbacks/prediction_writer.py > $ROOT/hepattn-changes.patch"
fi
exit "$status"
