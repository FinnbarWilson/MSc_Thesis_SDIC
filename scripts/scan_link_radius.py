"""Choose CLUE's cross-subsystem linking radius, on the tuning window.

    python -m scripts.scan_link_radius [--events 50]

WHY THIS IS A SEPARATE SCAN AND NOT PART OF `tune_clue`.

`scripts.tune_clue` optimises each subsystem independently, because ECAL and HCAL cell
geometries differ by an order of magnitude and one density radius cannot mean the same thing
in both. `clue.link_radius` is the opposite kind of parameter: it exists only to JOIN
subsystems, so it has no meaning inside a per-subsystem trial and Optuna never sees it there.
It is one scalar, monotonic in its effect, and cheap to scan directly.

WHY IT MATTERS UNDER SHOWER-LEVEL TRUTH. CLUE clusters one subsystem at a time, so a shower
crossing ECAL into HCAL is split by construction. With the truth collapsed to whole showers,
42.2% of targets span a boundary and carry 10.8% of target energy -- so without linking CLUE
cannot represent 42% of the target set, and its efficiency would be capped for a reason
belonging to the harness rather than the algorithm.

WHAT TO READ OFF. Efficiency rises with the radius as split showers are rejoined, then purity
falls as genuinely separate showers get merged. `split_rate` and `merge_rate` show the two
effects directly and are the honest way to pick the knee, rather than maximising f1 alone --
f1 will happily trade a real merge for a spurious efficiency gain. The scan prints both.

Runs on the TUNE store, never the evaluation one, for the same reason the CLUE parameters and
the MaskFormer thresholds are chosen there: a working point selected on the reported events is
not a working point, it is a fit.
"""

import argparse
import json

import pandas as pd

from src.clue.pipeline import cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.metrics import score_event
from src.io.event_store import EventStore

#: Radii in (eta, phi). 0 is the unlinked baseline and must stay in the grid: it is the number
#: the linked result has to beat, and reporting it is what stops the stage from being invisible.
#: The top of the range is deliberately past the point of usefulness -- an ECAL shower is
#: ~0.02-0.05 wide, so 0.20 should already be over-merging, and a scan whose best value sits at
#: its own edge has not found an optimum.
RADII = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)


def summarise(particles: pd.DataFrame, clusters: pd.DataFrame, radius: float, wp: float) -> dict:
    return {
        "link_radius": radius,
        "efficiency": float((particles["eff_e"] >= wp).mean()) if len(particles) else 0.0,
        "purity": float((clusters["pur_e"] >= wp).mean()) if len(clusters) else 0.0,
        "mean_eff": float(particles["eff_e"].mean()) if len(particles) else 0.0,
        "mean_pur": float(clusters["pur_e"].mean()) if len(clusters) else 0.0,
        # Energy-weighted forms, matching `metrics.weighting: energy` in the config. These are the
        # two the knee is read off: splitting should FALL as the radius grows and merging RISE,
        # and the radius to take is where the second starts moving faster than the first.
        "split_rate": float(particles["is_split_e"].mean()) if len(particles) else 0.0,
        "merge_rate": float(clusters["is_merge_e"].mean()) if len(clusters) else 0.0,
        "clusters_per_event": float(clusters.groupby("sample_id").size().mean()) if len(clusters) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=int, default=0, help="0 = the whole tuning window")
    args = parser.parse_args()

    cfg = settings()
    print(describe())

    store = EventStore(store_path("tune_store"), expect=store_expectations())
    n = len(store) if args.events <= 0 else min(args.events, len(store))
    records = [store[i] for i in range(n)]
    print(f"\nscanning {len(RADII)} radii over {n} tuning events\n")

    params_path = results_dir() / "clue_parameters.json"
    if not params_path.exists():
        msg = f"{params_path} does not exist. Run `python -m scripts.tune_clue` first -- the linking radius is chosen on TOP of the tuned per-subsystem parameters, not instead of them."
        raise SystemExit(msg)
    tuned = json.loads(params_path.read_text())
    clue_params = {name: entry["parameters"] for name, entry in tuned["subsystems"].items()}

    wp = cfg["metrics"]["working_points"][0]
    rows = []
    for radius in RADII:
        parts, clus = [], []
        for record in records:
            label, n_clusters = cluster_event(
                record,
                clue_params,
                subsystems=tuple(cfg["detectors"]),
                coords=cfg["clue"]["coords"],
                backend=cfg["clue"]["backend"],
                min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
                link_radius=radius,
            )
            p, c, _ = score_event(
                record,
                label,
                n_clusters,
                algo="clue",
                split_fraction=cfg["metrics"]["split_fraction"],
                min_overlap=cfg["metrics"]["min_overlap"],
                min_overlap_frac=cfg["metrics"]["min_overlap_frac"],
            )
            parts.append(p)
            clus.append(c)
        row = summarise(pd.concat(parts, ignore_index=True), pd.concat(clus, ignore_index=True), radius, wp)
        rows.append(row)
        print(
            f"  r={radius:5.3f}  eff {row['efficiency']:.3f}  pur {row['purity']:.3f}  "
            f"split {row['split_rate']:.3f}  merge {row['merge_rate']:.3f}  "
            f"clusters/ev {row['clusters_per_event']:6.1f}",
            flush=True,
        )

    scan = pd.DataFrame(rows)
    scan["f1"] = 2 * scan.efficiency * scan.purity / (scan.efficiency + scan.purity).clip(lower=1e-12)

    out = results_dir() / "link_radius_scan.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(out, index=False)

    best = scan.loc[scan.f1.idxmax()]
    baseline = scan.loc[scan.link_radius == 0.0].iloc[0]
    print(f"\nwrote {out}")
    print(f"\nunlinked baseline : eff {baseline.efficiency:.3f}  pur {baseline.purity:.3f}  f1 {baseline.f1:.3f}")
    print(f"best f1 at radius : {best.link_radius:.3f}  eff {best.efficiency:.3f}  pur {best.purity:.3f}  f1 {best.f1:.3f}")
    if best.link_radius == max(RADII):
        print("\n  WARNING: the best radius is the largest one scanned, so this is a bound rather than")
        print("  an optimum. Widen RADII and re-run before setting clue.link_radius from it.")
    print("\nSet clue.link_radius in config/experiment.yaml from this, then re-run scripts.score.")


if __name__ == "__main__":
    main()
