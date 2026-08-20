# Where everything lives. Sourced by the other setup scripts and by
# src/maskformer/ce_ai_1/env.sh; not run directly.
#
# Everything the repository builds goes under external/, which is gitignored. Only the dataset
# lives outside, on a shared datastore.

# Derived rather than hardcoded, so a clone anywhere works.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build artefacts (gitignored). ~12 GB total.
EXTERNAL="$REPO_ROOT/external"
HEPATTN="$EXTERNAL/hepattn"                 # upstream checkout + our patch
VENV_TRAIN="$EXTERNAL/venv-hepattn"         # torch + hepattn, for training and dumping
CONDA_ROOT="$EXTERNAL/miniforge3"           # carries the analysis env
ENV_ANALYSIS="$EXTERNAL/conda-envs/calo-clustering"   # numpy-only: CLUE, scoring, figures

# The dataset, the only thing that lives outside the repository.
DATA_ROOT="${COLLIDERML_DATA:-$EXTERNAL/ColliderML_data}"

# The Comet key, kept outside the repository so it cannot be committed.
COMET_ENV_FILE="${COMET_ENV_FILE:-$HOME/.config/colliderml/comet.env}"
