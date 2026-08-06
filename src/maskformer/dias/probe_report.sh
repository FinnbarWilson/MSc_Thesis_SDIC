#!/bin/bash -l
#SBATCH --job-name=calo_report
#SBATCH --partition=COMPUTE
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_report_%j.out
#SBATCH --error=/home/xucapfwi/MSc_Thesis_SDIC/external/slurm_logs/calo_report_%j.err

# The last link in the overnight chain: read whatever stores the probe arms produced and write the
# comparison to external/probes/REPORT.txt.
#
# Submitted with --dependency=afterany on all three arms, NOT afterok: an arm that crashes must not
# take the report down with it, because the report is the thing that says which arm crashed.
# compare_probes.py skips a missing or unreadable store and names it.
#
# CPU partition on purpose. Nothing here touches a GPU -- the arms already dumped their stores, and
# everything below is the numpy-only half of the repository.

set -euo pipefail

REPO=/home/xucapfwi/MSc_Thesis_SDIC
OUT="$REPO/external/probes/REPORT.txt"
mkdir -p "$REPO/external/probes"
cd "$REPO"

{
  echo "Probe arm comparison"
  echo "generated $(date)"
  echo
  echo "Jobs:"
  sacct -X -j "${PROBE_JOBS:-}" --format=JobID,JobName%16,State,Elapsed 2>/dev/null || true
  echo
  conda run --no-capture-output -n calo-clustering python src/maskformer/dias/compare_probes.py 100
} > "$OUT" 2>&1 || true

echo "wrote $OUT"
cat "$OUT"
