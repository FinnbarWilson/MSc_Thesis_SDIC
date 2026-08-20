"""Choose CLUE's cross-subsystem linking radius, on the tuning window.

    python -m scripts.scan_link_radius [--events 50]

``clue.link_radius`` exists only to join subsystems, so it has no meaning inside a per-subsystem
Optuna trial and is scanned directly instead. Read the knee off ``split_rate`` and ``merge_rate``
rather than off f1 alone, which will trade a real merge for a spurious efficiency gain. Runs on
the tune store; writes ``link_radius_scan.parquet``.
"""

import argparse
import json

import pandas as pd

from src.clue.pipeline import cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.metrics import score_event
from src.io.event_store import EventStore

#: Radii in (eta, phi). 0 is the unlinked baseline the linked result has to beat, and the top of
#: the range is deliberately past usefulness, an ECAL shower being ~0.02-0.05 wide, so an
#: optimum sitting on the edge is visible as one.
RADII = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20)


def summarise(particles: pd.DataFrame, clusters: pd.DataFrame, radius: float, wp: float) -> dict:
    return {
        "link_radius": radius,
        "efficiency": float((particles["eff_e"] >= wp).mean()) if len(particles) else 0.0,
        "purity": float((clusters["pur_e"] >= wp).mean()) if len(clusters) else 0.0,
        "mean_eff": float(particles["eff_e"].mean()) if len(particles) else 0.0,
        "mean_pur": float(clusters["pur_e"].mean()) if len(clusters) else 0.0,
        # The knee is read off these two: splitting falls as the radius grows and merging rises,
        # and the radius to take is where the second moves faster than the first.
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
        msg = (f"{params_path} does not exist. Run `python -m scripts.tune_clue` first: the "
               f"linking radius is chosen on top of the tuned per-subsystem parameters, not "
               f"instead of them.")
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
