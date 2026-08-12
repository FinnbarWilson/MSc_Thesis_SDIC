"""Choose the MaskFormer working point, on the tuning window.

    python -m scripts.scan_mf_threshold

Two thresholds turn the model's raw output into clusters:

    mask_threshold    per CELL: is this cell in this query's cluster?
    object_threshold  per QUERY: is this query a real particle at all, or an empty slot?

Both are post-hoc. The event store keeps the mask probabilities down to 0.02, so every point on
this grid is re-derived offline with no GPU and no re-clustering -- unlike CLUE, where each
working point costs a full re-run. That is why this scan is cheap and `scan_working_points` is
not.

Runs on the TUNE store and picks by f1, which is the same window and the same criterion
`scripts.tune_clue` uses for CLUE. That symmetry is the point: neither method may choose its
working point on the events it is reported over, and neither may choose it by a friendlier rule
than the other.

WHAT THE OUTPUT IS FOR. Set `maskformer.mask_threshold` / `.object_threshold` in
config/experiment.yaml from the top row, then re-run `scripts.score`. Also worth reading is how
FLAT the grid is in the mask direction: if f1 barely moves as mask_threshold sweeps its whole
range, the mask head is not what limits the model and the object head is.
"""

import argparse

import pandas as pd

from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.metrics import score_event
from src.io.event_store import EventStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=int, default=0, help="0 = the whole tuning window")
    args = parser.parse_args()

    cfg = settings()
    print(describe())

    store = EventStore(store_path("tune_store"), expect=store_expectations())
    n = len(store) if args.events <= 0 else min(args.events, len(store))
    records = [store[i] for i in range(n)]

    mask_grid = cfg["maskformer"]["mask_scan"]
    object_grid = cfg["maskformer"]["object_scan"]
    wp = cfg["metrics"]["working_points"][0]
    print(f"\n{len(mask_grid)} x {len(object_grid)} grid over {n} tuning events\n")

    rows = []
    for mask_threshold in mask_grid:
        for object_threshold in object_grid:
            parts, clus = [], []
            for record in records:
                label, n_clusters = record.maskformer_labels(
                    mask_threshold=mask_threshold,
                    object_threshold=object_threshold,
                    min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
                )
                p, c, _ = score_event(
                    record,
                    label,
                    n_clusters,
                    algo="maskformer",
                    split_fraction=cfg["metrics"]["split_fraction"],
                    min_overlap=cfg["metrics"]["min_overlap"],
                    min_overlap_frac=cfg["metrics"]["min_overlap_frac"],
                )
                parts.append(p)
                clus.append(c)
            parts = pd.concat(parts, ignore_index=True)
            clus = pd.concat(clus, ignore_index=True)
            eff = float((parts["eff_e"] >= wp).mean()) if len(parts) else 0.0
            pur = float((clus["pur_e"] >= wp).mean()) if len(clus) else 0.0
            rows.append(
                {
                    "mask_threshold": mask_threshold,
                    "object_threshold": object_threshold,
                    "efficiency": eff,
                    "purity": pur,
                    "f1": 2 * eff * pur / max(eff + pur, 1e-12),
                    "clusters_per_event": float(clus.groupby("sample_id").size().mean()) if len(clus) else 0.0,
                    "cells_per_cluster": float(clus["n_hits"].mean()) if len(clus) else 0.0,
                }
            )

    scan = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out = results_dir() / "mf_threshold_scan.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(out, index=False)

    print(scan.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = scan.iloc[0]
    print(f"\nwrote {out}")
    print(f"\nbest: mask_threshold {best.mask_threshold}  object_threshold {best.object_threshold}  f1 {best.f1:.4f}")

    # How much of the grid's variation is the mask axis, and how much the object axis. A flat mask
    # axis says the cell probabilities are already well separated and the object head is the limit.
    at_best_object = scan[scan.object_threshold == best.object_threshold]
    print(
        f"\nat object_threshold {best.object_threshold}, f1 across the whole mask range: "
        f"{at_best_object.f1.min():.4f} - {at_best_object.f1.max():.4f}"
    )
    at_best_mask = scan[scan.mask_threshold == best.mask_threshold]
    print(
        f"at mask_threshold {best.mask_threshold}, f1 across the whole object range: "
        f"{at_best_mask.f1.min():.4f} - {at_best_mask.f1.max():.4f}"
    )
    print("\nSet maskformer.mask_threshold / .object_threshold in config/experiment.yaml, then re-run scripts.score.")


if __name__ == "__main__":
    main()
