#!/bin/bash -l
#SBATCH --job-name=enc_affinity
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/enc_affinity_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/enc_affinity_%j.err

# Ask whether the trained encoder already represents "these two cells share a particle".
#
#   sbatch src/maskformer/dias/probe_encoder_affinity.sh
#
# A forward pass only -- nothing is trained and nothing is written to the event store. The whole
# point is that it needs no retraining: if the encoder's embeddings separate same-particle pairs
# from different-particle pairs better than plain distance does, then affinity-driven chaining is
# available from the checkpoint you already have, and the next step is to store the embeddings.
# If they do not, the relation has to be trained for, and the cheapest way is an auxiliary
# cell-cell affinity head rather than a different architecture.
#
# See the module docstring in probe_encoder_affinity.py for the full argument.

set -euo pipefail

# shellcheck disable=SC1091
. /home/xucapfwi/MSc_Thesis_SDIC/src/maskformer/dias/env.sh

CKPT="${CKPT:-$REPO/external/logs/ColliderML_Calo_Clustering_20260805-T172452/ckpts/epoch=006-val_loss=4.92620.ckpt}"
EVENTS="${EVENTS:-20}"
RADIUS="${RADIUS:-0.06}"

echo "Node:   $(hostname)"
echo "Ckpt:   $CKPT"
echo "Events: $EVENTS   pair radius: $RADIUS m"

preflight_paths
select_gpu
sync_mirror

cd "$EXP_DIR"
run_in_container "$ENV_PYTHON" \
  "$REPO/src/maskformer/dias/probe_encoder_affinity.py" \
  --ckpt "$CKPT" --events "$EVENTS" --radius "$RADIUS"
