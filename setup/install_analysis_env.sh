#!/usr/bin/env bash
# Build the ANALYSIS environment: CLUE, scoring, figures. No GPU, no hepattn.
#
#   ./setup/install_analysis_env.sh
#
# This is the half of the repository an assessor needs: src/clue, src/evaluation, src/plotting and
# src/io import nothing but numpy/scipy/pandas/matplotlib, so the figures regenerate without a GPU
# and without the 300 GB dataset. It is a SEPARATE environment from the training venv on purpose --
# that separation is the repository's central design decision, not an installation detail.
#
# conda rather than a venv, because environment.yml explains why pip cannot do it: CLUEstering
# needs a scikit-learn with no wheel for this python/numpy combination (pip falls back to a source
# build and fails), and the CLUE CPU backends compile against Boost headers.
#
# Everything lands in external/ (gitignored).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$HERE/paths.sh"

mkdir -p "$EXTERNAL"

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

# curl, wget, then python, because THIS SCRIPT IS RUN INSIDE A CONTAINER ON DIAS and
# docker://ubuntu:22.04 is the bare base image -- it ships no curl, and the original
# `curl -fsSL ...` line failed with "curl: command not found" before conda existed to install one.
# On DIAS the whole env must be built inside the container anyway: the host is RHEL7 with glibc
# 2.17 and the pinned conda-forge builds of numpy 2.4 / pandas 3.0 need 2.28+.
#
#   apptainer exec --bind $HOME ~/ubuntu22.sif bash setup/install_analysis_env.sh
fetch() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$1" "$2"
    elif command -v wget >/dev/null 2>&1; then wget -qO "$1" "$2"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$2" "$1"
    else
        echo "ABORT: need one of curl, wget or python3 to download $2" >&2; return 1
    fi
}

if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    echo "=== installing miniforge into $CONDA_ROOT ==="
    fetch "$EXTERNAL/miniforge.sh" "$MINIFORGE_URL"
    bash "$EXTERNAL/miniforge.sh" -b -p "$CONDA_ROOT"
    rm -f "$EXTERNAL/miniforge.sh"
else
    echo "miniforge already at $CONDA_ROOT"
fi

export CONDA_PKGS_DIRS="$EXTERNAL/conda-pkgs"
export CONDA_ENVS_PATH="$EXTERNAL/conda-envs"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

# CLUEstering's wheel build runs in a pip-isolated temp environment with its OWN cmake, which does
# not know the conda prefix and so cannot find Boost there -- it fails with
#     Could NOT find Boost (missing: Boost_INCLUDE_DIR atomic)
# even once libboost is installed. These point that cmake at the env. Measured: with libboost
# installed and these set, the wheel builds; without them it does not.
export BOOST_ROOT="$ENV_ANALYSIS"
export CMAKE_PREFIX_PATH="$ENV_ANALYSIS"
export CMAKE_ARGS="-DBoost_INCLUDE_DIR=$ENV_ANALYSIS/include -DBOOST_ROOT=$ENV_ANALYSIS -DBoost_USE_STATIC_LIBS=OFF"

echo "=== creating calo-clustering from environment.yml ==="
# Full output, not tail: the CLUEstering wheel build is the step that fails here, and its compiler
# error is the only thing that says why. A truncated log costs a whole rebuild to recover.
LOG="$EXTERNAL/install_analysis.log"
if ! "$CONDA_ROOT/bin/conda" env create -f "$REPO_ROOT/environment.yml" --yes > "$LOG" 2>&1; then
    if ! "$CONDA_ROOT/bin/conda" env update -f "$REPO_ROOT/environment.yml" >> "$LOG" 2>&1; then
        echo "FAILED. The compiler error is in $LOG; the relevant part is usually:"
        grep -iE "error:|fatal error|Failed building|CMake Error" "$LOG" | tail -20 || true
        exit 1
    fi
fi

echo
echo "=== verifying ==="
"$ENV_ANALYSIS/bin/python" - <<'PY'
missing = []
for m in ["numpy","scipy","pandas","matplotlib","pyarrow","optuna","sklearn","CLUEstering","fastjet","yaml"]:
    try:
        __import__(m); print(f"  OK      {m}")
    except Exception as e:
        print(f"  MISSING {m}  ({type(e).__name__}: {e})"); missing.append(m)
raise SystemExit(1 if missing else 0)
PY

cat <<EOF

Analysis env ready:  $ENV_ANALYSIS

Use it from the repository root, e.g.

    $ENV_ANALYSIS/bin/python -m scripts.show_config
    $ENV_ANALYSIS/bin/python -m scripts.make_thesis_figures --datasets pu0 pu200 --from-summary

or activate it:

    source $CONDA_ROOT/etc/profile.d/conda.sh
    export CONDA_ENVS_PATH=$EXTERNAL/conda-envs
    conda activate calo-clustering
EOF
