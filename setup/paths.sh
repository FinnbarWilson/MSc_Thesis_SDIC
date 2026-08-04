# Where everything lives. Sourced by the other setup scripts and by
# src/maskformer/ce_ai_1/env.sh; not run directly.
#
# THE RULE: code and build artefacts live in the repository. Only the dataset lives on
# /mnt/ai-datastore, because that is a shared datastore and 300 GB of parquet is the only thing
# that belongs on it. Everything the repository builds goes under external/, which is gitignored.

# The repository root, derived rather than hardcoded so a clone anywhere works.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build artefacts (gitignored). ~12 GB total.
EXTERNAL="$REPO_ROOT/external"
HEPATTN="$EXTERNAL/hepattn"                 # upstream checkout + our patch
VENV_TRAIN="$EXTERNAL/venv-hepattn"         # torch + hepattn, for training and dumping
CONDA_ROOT="$EXTERNAL/miniforge3"           # carries the analysis env
ENV_ANALYSIS="$EXTERNAL/conda-envs/calo-clustering"   # numpy-only: CLUE, scoring, figures

# The dataset. The ONLY thing outside the repository, and the only thing on the shared store.
DATA_ROOT="${COLLIDERML_DATA:-/mnt/ai-datastore/finnbar/ColliderML_data}"

# The Comet key. Outside the repository so it cannot be committed; in the user's own config
# directory rather than on the shared datastore.
COMET_ENV_FILE="${COMET_ENV_FILE:-$HOME/.config/colliderml/comet.env}"
