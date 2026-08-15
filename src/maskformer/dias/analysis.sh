#!/bin/bash -l
#SBATCH --job-name=calo_analysis
# CPU ONLY. Stages 2-5 of the pipeline touch no GPU -- src/clue, src/evaluation, src/plotting and
# src/io import nothing but numpy/scipy/pandas/matplotlib -- so this belongs on COMPUTE and must not
# sit on a GPU card that a training run could be using.
#SBATCH --partition=COMPUTE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_analysis_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_analysis_%j.err

# Turn a dumped event store into CLUE numbers, MaskFormer numbers, the thesis figures and the
# portable figure summary. The DIAS counterpart to scripts/run_pu200_pipeline.sh.
#
#   sbatch src/maskformer/dias/analysis.sh            # the active dataset in config/experiment.yaml
#   SKIP_TUNE=1 sbatch src/maskformer/dias/analysis.sh   # reuse the committed clue_parameters.json
#
# THE ORDER MATTERS AND TWO STEPS ARE EASY TO SKIP. Both the CLUE search ranges and the MaskFormer
# working point are MEASUREMENTS, and config/experiment.yaml says they must be re-derived per
# dataset rather than inherited. Doing it by hand invites forgetting one and quoting one dataset's
# threshold over another's numbers.
#
# WHY THE CONTAINER. DIAS is RHEL7 with glibc 2.17; the pinned conda-forge builds in
# environment.yml (numpy 2.4.6, pandas 3.0.3) need 2.28+. The analysis env is therefore BUILT and
# RUN inside ~/ubuntu22.sif. It is the same env environment.yml specifies -- the versions are
# pinned because the clustering output depends on them, so this is not the place to improvise.
#
#   apptainer exec --bind $HOME ~/ubuntu22.sif bash setup/install_analysis_env.sh
#
# WHERE THE STORE IS. config/experiment.yaml holds ce-ai-1 paths. CALO_STORE_ROOT relocates them
# without editing the config, keeping one config valid on both machines; the store NAME still comes
# from the config, because that name encodes the window and format version EventStore checks.
set -uo pipefail

REPO="${REPO:-/home/xucapfwi/MSc_Thesis_SDIC}"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
. "$REPO/src/maskformer/dias/env.sh"

export CALO_STORE_ROOT="${CALO_STORE_ROOT:-$HOME/eventstores}"
PY="$REPO/external/conda-envs/calo-clustering/bin/python"

echo "host       : $(hostname)"
echo "started    : $(date)"
echo "job        : ${SLURM_JOB_ID:-<interactive>}"
echo "store root : $CALO_STORE_ROOT"
echo

[ -x "$PY" ] || { echo "ABORT: no analysis env at $PY -- run setup/install_analysis_env.sh inside the container"; exit 1; }
[ -f "$SIF" ] || { echo "ABORT: no container at $SIF"; exit 1; }

run() { apptainer exec --bind /home/xucapfwi "$SIF" "$PY" "$@"; }

echo "=== [0/4] resolved configuration"
run -m scripts.show_config || exit 1

# The CLUE search ranges are the values most likely to be wrong on a new dataset. tune_subsystem
# flags any optimum landing in the outer 5% of its log range; when it does, widen that bound in
# config/experiment.yaml and re-run, because an optimum on a boundary is a truncated search rather
# than a converged one.
if [ "${SKIP_TUNE:-0}" = "1" ]; then
    echo && echo "=== [1/4] SKIP_TUNE=1 -- keeping the existing clue_parameters.json"
else
    echo && echo "=== [1/4] tuning CLUE on the tune store"
    run -m scripts.tune_clue || exit 1
fi

# The working point is re-derived on the TUNE store, so the threshold is not chosen on the events
# it is later scored over -- the same discipline CLUE's parameters get.
echo && echo "=== [2/4] scanning the MaskFormer working point on the tune store"
run -m scripts.scan_working_points || exit 1

# ONE ALGORITHM PER INVOCATION. scripts/score.py takes --algo as a required choice and scores
# exactly one, writing particles_<algo>.parquet and clusters_<algo>.parquet; there is no "score
# everything" mode. scripts/run_pu200_pipeline.sh calls a bare `scripts.score` and therefore dies
# with "the following arguments are required: --algo" -- job 48421 hit exactly that, 44 minutes in,
# after the tuning it had just spent 40 of them on. Fixed there too.
#
# --params is NOT passed: score.py defaults to this dataset's own results/<ds>/clue_parameters.json,
# which is what stage 1 just wrote. Passing it explicitly was a footgun once results became
# dataset-scoped, and the file documents that.
#
# The two oracle_* rows are reference clusterings rather than methods under test, and
# maskformer_chained is the interventions-table arm that METHODS_MAIN deliberately excludes from
# the headline figures. All five are scored because results/pu0/ carried all five before.
ALGOS="${ALGOS:-maskformer clue oracle_geometric oracle_resolution maskformer_chained}"
echo && echo "=== [3/4] scoring on the eval store: $ALGOS"
for algo in $ALGOS; do
    echo "--- scoring $algo"
    run -m scripts.score --algo "$algo" || exit 1
done

# Writes figures/thesis/*.pdf|png AND results/<ds>/figure_summary.csv. The summary is the small
# committable file the other cluster needs to draw this dataset's column without the per-row tables.
echo && echo "=== [4/4] thesis figures + portable figure summary"
run -m scripts.make_thesis_figures || exit 1

echo
echo "DONE. Before quoting anything:"
echo "  - did stage 1 print range-edge warnings? if so the CLUE ranges are still wrong."
echo "  - commit results/<ds>/figure_summary.csv; that is what the other cluster plots from."
