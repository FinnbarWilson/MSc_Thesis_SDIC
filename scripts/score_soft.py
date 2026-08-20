"""Score both methods with fractional cell ownership on both sides.

    python -m scripts.score_soft [--limit 50]

`scripts.score` collapses overlapping masks to one winner per cell so CLUE has something it can
express; this does not. Prediction weights are normalised per cell, so a partitioning method
passes through with every weight equal to 1 and is scored by identical code. Expect its
efficiency to be lower here than in the head-to-head: the denominator has changed to the
particle's actual deposited energy. See :mod:`src.evaluation.soft`. Writes
``capability_summary.csv``.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.score import DEFAULT_CLUE_PARAMS
from src.clue.pipeline import SUBSYSTEMS, cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.soft import capability_summary, hard_weights, score_event_soft, sharing_diagnostics
from src.io.event_store import EventStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--params", type=Path, default=None,
                        help="defaults to results/<active dataset>/clue_parameters.json")
    parser.add_argument("--out", type=Path, default=None, help="defaults to results/<active dataset>/")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N events")
    args = parser.parse_args()

    cfg = settings()
    out_dir = args.out or results_dir()
    params_path = args.params or (results_dir() / "clue_parameters.json")
    print(describe())
    store = EventStore(args.store or store_path(), expect=store_expectations())
    print(f"{len(store)} events from {store.root}")

    if params_path.exists():
        tuned = json.loads(params_path.read_text())["subsystems"]
        clue_params = {name: entry["parameters"] for name, entry in tuned.items()}
    else:
        clue_params = dict.fromkeys(SUBSYSTEMS, DEFAULT_CLUE_PARAMS)
        print(f"  clue params  UNTUNED defaults (no {params_path})")

    # MaskFormer divides a contested cell in proportion to mask probabilities; CLUE, being an
    # exclusive partition, is pinned at one claim per cell. That contrast is what is measured.
    algos = ["maskformer", "clue"]

    frames: dict[str, list] = {name: [] for name in algos}
    diagnostics: dict[str, list] = {name: [] for name in algos}
    for i, record in enumerate(store):
        if args.limit and i >= args.limit:
            break

        cluster, cell, weight, n_soft = record.maskformer_soft_masks(
            mask_threshold=cfg["maskformer"]["mask_threshold"],
            object_threshold=cfg["maskformer"]["object_threshold"],
            min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
        )
        frames["maskformer"].append(
            score_event_soft(record, cluster, cell, weight, n_soft, "maskformer",
                             min_overlap_frac=cfg["metrics"]["min_overlap_frac"])
        )
        diagnostics["maskformer"].append(sharing_diagnostics(record, cell, weight))

        label, n_hard = cluster_event(
            record, clue_params, subsystems=tuple(cfg["detectors"]),
            coords=cfg["clue"]["coords"], backend=cfg["clue"]["backend"],
            min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
            link_radius=cfg["clue"].get("link_radius", 0.0),
        )
        c_cluster, c_cell, c_weight = hard_weights(label)
        frames["clue"].append(
            score_event_soft(record, c_cluster, c_cell, c_weight, n_hard, "clue",
                             min_overlap_frac=cfg["metrics"]["min_overlap_frac"])
        )
        diagnostics["clue"].append(sharing_diagnostics(record, c_cell, c_weight))

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for algo, parts in frames.items():
        table = pd.concat(parts, ignore_index=True)
        path = out_dir / f"soft_particles_{algo}.parquet"
        table.to_parquet(path, index=False)
        print(f"  wrote {path}  ({len(table)} rows, {path.stat().st_size / 1e6:.1f} MB)")
        tables.append(table)

    wp = cfg["metrics"]["working_points"][0]
    summary = capability_summary(tables, wp, diagnostics)
    summary.to_csv(out_dir / "capability_summary.csv", index=False)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
