#!/usr/bin/env bash
# Everything needed to turn a trained pu200 checkpoint into CLUE numbers, MaskFormer numbers and
# the pu200 figures. Run from the repository root:
#
#   CKPT=<path to pu200 .ckpt> ./scripts/run_pu200_pipeline.sh
#
# WHY THIS EXISTS. The steps are individually documented but the ORDER matters and two of them are
# easy to skip: the CLUE search ranges and the MaskFormer working point are both pu0 MEASUREMENTS
# that config/experiment.yaml says must be re-derived for pu200, not inherited. Doing that by hand
# invites forgetting one and quoting a pu0 threshold on pu200 numbers.
#
# STAGE 1 NEEDS THE GPU, stages 2-5 do not. On 2026-08-09 stage 1 could not run: a 5-epoch training
# job held 49.8 GB and another user 28.0 GB of an 81 GB card, leaving 3.2 GB. Wait for the card, or
# run stage 1 alone when it frees and the rest any time after.
#
# SET dataset.active: pu200 in config/experiment.yaml FIRST. This script refuses to run otherwise
# rather than editing your config: `active` also selects the overrides block and decides that output
# lands in results/pu200/ and figures/pu200/, so flipping it silently is not a kindness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
. setup/paths.sh
PY="$ENV_ANALYSIS/bin/python"

active=$("$PY" -c "import sys; sys.path.insert(0,'.'); from src.config import active_dataset; print(active_dataset())")
if [ "$active" != "pu200" ]; then
    echo "ABORT: dataset.active is '$active'. Set it to pu200 in config/experiment.yaml first," >&2
    echo "       or this run would write pu0 results from pu200 inputs." >&2
    exit 2
fi

# ---------------------------------------------------------------- 1. stores (GPU)
if [ -n "${CKPT:-}" ]; then
    echo "=== [1/5] dumping event stores from $CKPT"
    CKPT="$CKPT" ./src/maskformer/ce_ai_1/dump_store.sh pu200 tune
    CKPT="$CKPT" ./src/maskformer/ce_ai_1/dump_store.sh pu200 eval
else
    echo "=== [1/5] CKPT unset -- skipping the dump and assuming the stores already exist"
fi

# ---------------------------------------------------------------- 2. tune CLUE
# READ THE EDGE WARNINGS THIS PRINTS. rho_c is a local energy density in units of raw cell energy
# and pileup raises the occupancy it is measured over, so the pu0 ranges are very likely the wrong
# box. `tune_subsystem` flags any optimum landing in the outer 5% of its log range; when it does,
# widen that bound in config/experiment.yaml under dataset.pu200.overrides.clue.search and re-run.
# The pu0 ranges needed exactly this treatment once already.
echo "=== [2/5] tuning CLUE on the tune store (80 Optuna trials x 50 events, CPU)"
"$PY" -m scripts.tune_clue

# ---------------------------------------------------------------- 3. MaskFormer working point
# The 0.5 / 0.2 pair in the config is a pu0 measurement. This re-derives it on the pu200 TUNE store,
# so the threshold is not chosen on the events it is later scored over.
echo "=== [3/5] scanning the MaskFormer working point on the tune store"
"$PY" -m scripts.scan_working_points

# ---------------------------------------------------------------- 4. score
echo "=== [4/5] scoring both methods on the eval store -> results/pu200/"
"$PY" -m scripts.score

# ---------------------------------------------------------------- 5. figures
echo "=== [5/5] figures -> figures/pu200/"
"$PY" -m scripts.make_figures

echo
echo "DONE. results/pu200/ and figures/pu200/ are written."
echo "Before quoting anything:"
echo "  - did stage 2 print range-edge warnings? if so the CLUE ranges are still wrong."
echo "  - is the checkpoint in dataset.pu200.overrides.maskformer.checkpoint the one you meant?"
echo "    make_figures warns if it disagrees with the store metadata."
