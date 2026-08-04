"""The multi-owner capability study: score both methods with fractional cell ownership.

The head-to-head in ``scripts.score`` collapses MaskFormer's overlapping masks to one winner
per cell, so CLUE has something it can express. This script does not, and scores both methods
against the multi-owner truth with ownership left fractional.

The point is not to arrange an even fight. It is that the exclusive metric suppresses a
capability one method has, and a metric that does that is measuring the format rather than the
clustering. Nothing here favours the model either: prediction weights are normalised per cell,
so a method whose clusters never overlap passes through with every weight equal to 1, and CLUE
is scored by identical code with no special case.

Expect CLUE's efficiency to be *lower* here than in the head-to-head, and do not read that as
the metric penalising it. The denominator has changed from "energy in cells this particle
dominates" -- a target defined by what a partition can express -- to the particle's actual
deposited energy, which neither method's output can move. Scoring the truth partition itself
under this metric gives exactly `exclusive_share` rather than 1, so that shortfall is a
property of the algorithm class, and measuring it is the whole purpose.

    python -m scripts.score_soft
    python -m scripts.score_soft --limit 50
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.clue.pipeline import SUBSYSTEMS, cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.soft import capability_summary, hard_weights, score_event_soft, sharing_diagnostics
from src.io.event_store import EventStore
from scripts.score import DEFAULT_CLUE_PARAMS


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
        clue_params = {name: DEFAULT_CLUE_PARAMS for name in SUBSYSTEMS}
        print(f"  clue params  UNTUNED defaults (no {params_path})")

    # The incidence head is the reason this study exists in the form it does, so it is scored
    # here whenever the store carries it. `maskformer` divides a contested cell in proportion
    # to MASK probabilities, which is the uncalibrated quantity: the measured symptom is that
    # it splits each cell 2.04 ways against truth's 1.22, an over-division that survives every
    # mask threshold up to 0.95 and therefore is not a working-point artefact. Incidence shares
    # are trained against the true energy fractions, so `maskformer_incidence` asks the same
    # capability question with the quantity that was supervised to answer it.
    algos = ["maskformer", "clue"]
    if store[0].has_incidence:
        algos.insert(1, "maskformer_incidence")
    else:
        print("  ! store has no incidence head; skipping maskformer_incidence (re-dump to include it)")

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

        if "maskformer_incidence" in frames:
            i_cluster, i_cell, i_weight, n_inc = record.maskformer_incidence_soft_masks(
                mask_threshold=cfg["maskformer"]["mask_threshold"],
                object_threshold=cfg["maskformer"]["object_threshold"],
                min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
            )
            frames["maskformer_incidence"].append(
                score_event_soft(record, i_cluster, i_cell, i_weight, n_inc, "maskformer_incidence",
                                 min_overlap_frac=cfg["metrics"]["min_overlap_frac"])
            )
            diagnostics["maskformer_incidence"].append(sharing_diagnostics(record, i_cell, i_weight))

        label, n_hard = cluster_event(
            record, clue_params, subsystems=tuple(cfg["detectors"]),
            coords=cfg["clue"]["coords"], backend=cfg["clue"]["backend"],
            min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
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
