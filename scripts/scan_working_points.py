"""Trace each method's purity/efficiency curve over its working points.

Reporting a single working point is misleading here: the two methods cross at almost exactly
the 0.5 recovery threshold, so which one "wins" is decided by where the threshold is put
rather than by either algorithm. The curve is the honest comparison.

The two knobs are not the same kind of thing. MaskFormer's working point is post-hoc -- the
store keeps its masks down to a probability of 0.02, so any (mask, object) threshold above
that is re-derived offline with no GPU. CLUE's is not: every point needs the algorithm run
again, so the scan is over `rho_c`, the density threshold, which is what actually trades its
efficiency against its purity.

Runs on a subset of events by default; the curve's shape is stable well before 500 events
and each CLUE point costs a full re-clustering.

    python -m scripts.scan_working_points --events 100
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.clue.pipeline import SUBSYSTEMS, cluster_event
from src.config import describe, figures_dir, results_dir, settings, store_expectations, store_path
from src.evaluation.metrics import score_event
from src.io.event_store import EventStore
from src.plotting import figures, style

#: CLUE's density threshold is scanned as a MULTIPLE of its tuned value rather than over an
#: absolute grid, so the scan follows the tuning instead of having to be re-chosen with it --
#: which matters at pu200, where the tuned `rho_c` is expected to land somewhere else entirely.
#: The MaskFormer grids are `maskformer.mask_scan` / `object_scan` in config/experiment.yaml,
#: read from there rather than restated here: they were duplicated once, drifted, and the scan
#: silently reported a different grid from the one the config documents.
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
    parser.add_argument("--params", type=Path, default=None,
                        help="defaults to results/<active dataset>/clue_parameters.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to results/<active dataset>/wp_scan.parquet")
    parser.add_argument("--figure", type=Path, default=None,
                        help="defaults to figures/<active dataset>/working_point_curve")
    args = parser.parse_args()

    cfg = settings()
    results = results_dir()
    params_path = args.params or (results / "clue_parameters.json")
    out_path = args.out or (results / "wp_scan.parquet")
    figure_path = args.figure or (figures_dir() / "working_point_curve")
    print(describe())

    wp = cfg["metrics"]["working_points"][0]
    mask_scan = cfg["maskformer"]["mask_scan"]
    object_scan = cfg["maskformer"]["object_scan"]
    store = EventStore(store_path(), expect=store_expectations())
    records = [store[i] for i in range(min(args.events, len(store)))]
    print(f"scanning working points on {len(records)} events")

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

    # Mark where the reachable corner is, if the reference clusterings have been scored. A
    # bare 0-1 square makes both methods look uniformly poor when most of that square is not
    # available to any algorithm.
    reference = {}
    for name in ("oracle_geometric", "oracle_resolution"):
        p_path = results / f"particles_{name}.parquet"
        c_path = results / f"clusters_{name}.parquet"
        if p_path.exists() and c_path.exists():
            reference[name] = (
                float((pd.read_parquet(p_path, columns=["eff_e"])["eff_e"] >= wp).mean()),
                float((pd.read_parquet(c_path, columns=["pur_e"])["pur_e"] >= wp).mean()),
            )

    style.apply()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figures.working_point_curve(scan, reference=reference, out=figure_path.with_suffix(f".{suffix}"))
    plt.close("all")
    print(f"wrote {figure_path}.pdf / .png")


if __name__ == "__main__":
    main()
