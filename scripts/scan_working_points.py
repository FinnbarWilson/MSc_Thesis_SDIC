"""Trace each method's purity/efficiency frontier over its working points.

    python -m scripts.scan_working_points --events 100

A single working point is misleading when the two methods cross near the reporting threshold;
the curve is the comparison to report. The two knobs differ in cost: MaskFormer's thresholds are
post-hoc and re-derived offline, while every CLUE point needs a full re-clustering, so the scan
runs on a subset of events by default. Writes ``wp_scan.parquet``.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.clue.pipeline import cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.metrics import score_event
from src.io.event_store import EventStore

#: CLUE's density threshold is scanned as a multiple of its tuned value rather than over an
#: absolute grid, so the scan follows the tuning rather than having to be re-chosen with it. The
#: MaskFormer grids are read from `maskformer.mask_scan` / `object_scan` in the config.
RHO_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def summarise(particles, clusters, algo, wp, **extra) -> dict:
    """Collapse one working point to the numbers the curve is drawn from."""
    return {
        "algo": algo,
        "efficiency": float((particles["eff_e"] >= wp).mean()) if len(particles) else 0.0,
        "purity": float((clusters["pur_e"] >= wp).mean()) if len(clusters) else 0.0,
        "mean_eff": float(particles["eff_e"].mean()) if len(particles) else 0.0,
        "match_rate": float(particles["matched"].mean()) if len(particles) else 0.0,
        "fake_rate": float((~clusters["matched"]).mean()) if len(clusters) else 0.0,
        "clusters_per_event": len(clusters) / max(particles["sample_id"].nunique(), 1),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=int, default=100)
    # Defaults to the scored store, since the usual job is to show the frontier the reported
    # point sits on. Pass --store tune_store to SELECT a working point, so it is not chosen on
    # the events it will be reported over.
    parser.add_argument("--store", choices=["store", "tune_store"], default="store")
    parser.add_argument("--params", type=Path, default=None,
                        help="defaults to results/<active dataset>/clue_parameters.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to results/<active dataset>/wp_scan.parquet")
    args = parser.parse_args()

    cfg = settings()
    results = results_dir()
    params_path = args.params or (results / "clue_parameters.json")
    out_path = args.out or (results / "wp_scan.parquet")
    print(describe())

    wp = cfg["metrics"]["working_points"][0]
    mask_scan = cfg["maskformer"]["mask_scan"]
    object_scan = cfg["maskformer"]["object_scan"]
    store = EventStore(store_path(args.store), expect=store_expectations())
    records = [store[i] for i in range(min(args.events, len(store)))]
    print(f"scanning working points on {len(records)} events from {args.store} ({store.root})")

    rows = []

    nominal_mask = cfg["maskformer"]["mask_threshold"]
    nominal_object = cfg["maskformer"]["object_threshold"]
    for mask_t in mask_scan:
        for object_t in object_scan:
            p, c = [], []
            for record in records:
                label, n = record.maskformer_labels(mask_t, object_t, cfg["metrics"]["min_cluster_hits"])
                pi, ci, _ = score_event(
                    record, label, n, algo="maskformer",
                    split_fraction=cfg["metrics"]["split_fraction"],
                    min_overlap=cfg["metrics"]["min_overlap"],
                    min_overlap_frac=cfg["metrics"]["min_overlap_frac"],
                )
                p.append(pi)
                c.append(ci)
            rows.append(
                summarise(
                    pd.concat(p, ignore_index=True),
                    pd.concat(c, ignore_index=True),
                    "maskformer",
                    wp,
                    knob=f"mask {mask_t} / obj {object_t}",
                    nominal=(mask_t == nominal_mask and object_t == nominal_object),
                )
            )
            print(f"  maskformer mask={mask_t} obj={object_t}: eff {rows[-1]['efficiency']:.3f} pur {rows[-1]['purity']:.3f}", flush=True)

    tuned = json.loads(params_path.read_text())["subsystems"]
    for scale in RHO_SCALES:
        params = {
            name: {**entry["parameters"], "rho_c_2d": entry["parameters"]["rho_c_2d"] * scale, "rho_c_3d": entry["parameters"]["rho_c_3d"] * scale}
            for name, entry in tuned.items()
        }
        p, c = [], []
        for record in records:
            label, n = cluster_event(
                record, params, subsystems=tuple(cfg["detectors"]), coords=cfg["clue"]["coords"],
                backend=cfg["clue"]["backend"], link_radius=cfg["clue"].get("link_radius", 0.0),
            )
            pi, ci, _ = score_event(
                record, label, n, algo="clue",
                split_fraction=cfg["metrics"]["split_fraction"],
                min_overlap=cfg["metrics"]["min_overlap"],
                min_overlap_frac=cfg["metrics"]["min_overlap_frac"],
            )
            p.append(pi)
            c.append(ci)
        rows.append(
            summarise(
                pd.concat(p, ignore_index=True),
                pd.concat(c, ignore_index=True),
                "clue",
                wp,
                knob=f"rho_c x{scale}",
                nominal=(scale == 1.0),
            )
        )
        print(f"  clue rho_c x{scale}: eff {rows[-1]['efficiency']:.3f} pur {rows[-1]['purity']:.3f}", flush=True)

    scan = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(scan[["algo", "knob", "efficiency", "purity", "mean_eff", "fake_rate", "clusters_per_event"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
