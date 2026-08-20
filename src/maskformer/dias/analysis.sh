#!/bin/bash -l
#SBATCH --job-name=calo_analysis
# CPU only: everything after the dump is numpy, so this must not occupy a GPU card.
#SBATCH --partition=COMPUTE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Relative to the directory sbatch was run from, so submit from the repository root.
#SBATCH --output=external/slurm_logs/calo_analysis_%j.out
#SBATCH --error=external/slurm_logs/calo_analysis_%j.err

# Turn a dumped event store into CLUE numbers, MaskFormer numbers, the thesis figures and the
# portable figure summary.
#
#   sbatch src/maskformer/dias/analysis.sh            # the active dataset in config/experiment.yaml
#   SKIP_TUNE=1 sbatch src/maskformer/dias/analysis.sh   # reuse the committed clue_parameters.json
#
# The order matters. Both the CLUE parameters and the MaskFormer working point are measurements
# that must be re-derived per dataset rather than inherited, and running the stages by hand
# invites quoting one dataset's threshold over another's numbers.
#
# The container is needed because DIAS is RHEL7 with glibc 2.17, while the pinned conda-forge builds in
# environment.yml (numpy 2.4.6, pandas 3.0.3) need 2.28+. The analysis env is therefore BUILT and
# run inside ~/ubuntu22.sif. It is the same environment environment.yml specifies; the versions are
# pinned because the clustering output depends on them, so this is not the place to improvise.
#
#   apptainer exec --bind $HOME ~/ubuntu22.sif bash setup/install_analysis_env.sh
#
# Where the store is. config/experiment.yaml holds relative paths under external/; CALO_STORE_ROOT relocates them
# without editing the config, keeping one config valid on both machines; the store NAME still comes
# from the config, because that name encodes the window and format version EventStore checks.
set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
. "$REPO/src/maskformer/dias/env.sh"

# Only needed if the stores are not under external/, which is where dump_store.sh puts them.
export CALO_STORE_ROOT="${CALO_STORE_ROOT:-$EXTERNAL/eventstores}"
PY="$ENV_ANALYSIS/bin/python"

echo "host       : $(hostname)"
echo "started    : $(date)"
echo "job        : ${SLURM_JOB_ID:-<interactive>}"
echo "store root : $CALO_STORE_ROOT"
echo

[ -x "$PY" ] || { echo "ABORT: no analysis env at $PY; run setup/install_analysis_env.sh inside the container"; exit 1; }
[ -f "$SIF" ] || { echo "ABORT: no container at $SIF"; exit 1; }

run() { apptainer exec --bind "$REPO" --bind "$DATA_ROOT" "$SIF" "$PY" "$@"; }

echo "=== [0/4] resolved configuration"
run -m scripts.show_config || exit 1

# The CLUE search ranges are the values most likely to be wrong on a new dataset. tune_subsystem
# flags any optimum landing in the outer 5% of its log range; when it does, widen that bound in
# config/experiment.yaml and re-run, because an optimum on a boundary is a truncated search rather
# than a converged one.
if [ "${SKIP_TUNE:-0}" = "1" ]; then
    echo && echo "=== [1/4] SKIP_TUNE=1, keeping the existing clue_parameters.json"
else
    echo && echo "=== [1/4] tuning CLUE on the tune store"
    run -m scripts.tune_clue || exit 1
fi

# The working point is re-derived on the TUNE store, so the threshold is not chosen on the events
# it is later scored over, the same discipline CLUE's parameters get.
echo && echo "=== [2/4] scanning the MaskFormer working point on the tune store"
run -m scripts.scan_working_points || exit 1

# One algorithm per invocation. scripts/score.py takes --algo as a required choice and scores
# exactly one, writing particles_<algo>.parquet and clusters_<algo>.parquet; there is no "score
# everything" mode.
#
# --params is not passed: score.py defaults to this dataset's own results/<ds>/clue_parameters.json,
# which is what stage 1 just wrote. Passing it explicitly was a footgun once results became
# dataset-scoped, and the file documents that.
#
# oracle_resolution is a reference clustering rather than a method under test; it is scored
# through the same entry point so the ceiling is measured by the same code as the methods.
ALGOS="${ALGOS:-maskformer clue oracle_resolution}"
echo && echo "=== [3/4] scoring on the eval store: $ALGOS"
for algo in $ALGOS; do
    echo "--- scoring $algo"
    run -m scripts.score --algo "$algo" || exit 1
done

# Writes figures/thesis/*.pdf|png and results/<ds>/figure_summary.csv. The summary is the small
# committable file the other cluster needs to draw this dataset's column without the per-row tables.
echo && echo "=== [4/4] thesis figures + portable figure summary"
run -m scripts.make_thesis_figures || exit 1

echo
echo "DONE. Before quoting anything:"
echo "  - did stage 1 print range-edge warnings? if so the CLUE ranges are still wrong."
echo "  - commit results/<ds>/figure_summary.csv; that is what the other cluster plots from."
