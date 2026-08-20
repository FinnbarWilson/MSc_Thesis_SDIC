"""Score a clustering against the truth partition and write the long tables.

    python -m scripts.score --algo maskformer
    python -m scripts.score --algo clue
    python -m scripts.score --algo oracle_resolution

Every algorithm comes through here, and the scorer receives a label per cell and nothing else.
Tables land in ``results/<active dataset>/``. ``oracle_resolution`` is a reference clustering
rather than a method under test; see :mod:`src.evaluation.oracle`.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.clue.pipeline import SUBSYSTEMS, cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.metrics import pool, score_event
from src.evaluation.oracle import resolution_labels
from src.io.event_store import EventStore

# Untuned starting point, calibrated against the measured cell energy scale. Replaced by the
# Optuna result once tuning has run.
DEFAULT_CLUE_PARAMS = {
    "d_c_2d": 0.02,
    "rho_c_2d": 1.0e-3,
    "d_o_2d": 0.04,
    "d_c_3d": 0.05,
    "rho_c_3d": 3.0e-3,
    "d_o_3d": 0.10,
    "depth_scale": 50.0,
}


def maskformer_labels(record, cfg):
    return record.maskformer_labels(
        mask_threshold=cfg["maskformer"]["mask_threshold"],
        object_threshold=cfg["maskformer"]["object_threshold"],
        min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
    )


def clue_labels(record, cfg, params):
    return cluster_event(
        record,
        params,
        subsystems=tuple(cfg["detectors"]),
        coords=cfg["clue"]["coords"],
        backend=cfg["clue"]["backend"],
        min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
        link_radius=cfg["clue"].get("link_radius", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--algo",
        choices=["maskformer", "clue", "oracle_resolution"],
        required=True,
    )
    parser.add_argument("--store", type=Path, default=None, help="defaults to dataset.<active>.store")
    parser.add_argument("--params", type=Path, default=None,
                        help="tuned CLUE parameters as JSON; defaults to this dataset's "
                             "results/<active dataset>/clue_parameters.json when it exists")
    parser.add_argument("--out", type=Path, default=None, help="defaults to results/<active dataset>/")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N events")
    parser.add_argument("--tag", default=None, help="suffix for the output files")
    args = parser.parse_args()

    cfg = settings()
    out_dir = args.out or results_dir()
    print(describe())
    store = EventStore(args.store or store_path(), expect=store_expectations())
    print(f"{len(store)} events from {store.root}")
    print(f"  checkpoint  {store.meta['maskformer']['checkpoint'].rsplit('/', 1)[-1]}")
    print(f"  cuts        {store.meta['hit_selection']}  {store.meta['particle_selection']}")

    # Defaults to the active dataset's own parameter file. `tune_clue` records the dataset
    # inside the file, so a set tuned elsewhere is reported rather than used silently.
    params_path = args.params or (results_dir() / "clue_parameters.json")
    if params_path.exists():
        tuned = json.loads(params_path.read_text())
        params_by_subsystem = {name: entry["parameters"] for name, entry in tuned["subsystems"].items()}
        print(f"  clue params  tuned, from {params_path}")
        tuned_on = tuned.get("dataset")
        if tuned_on and tuned_on != cfg["dataset"]["active"]:
            print(f"  ! those parameters were tuned on {tuned_on}, not {cfg['dataset']['active']}. "
                  f"CLUE's density thresholds do not transfer across pileup conditions; "
                  f"run scripts.tune_clue on this dataset.")
    else:
        params_by_subsystem = dict.fromkeys(SUBSYSTEMS, DEFAULT_CLUE_PARAMS)
        if args.algo == "clue":
            print(f"  clue params  untuned defaults (no {params_path}); run scripts.tune_clue "
                  f"before reporting these numbers")

    particles, clusters, events = [], [], []
    for i, record in enumerate(store):
        if args.limit and i >= args.limit:
            break
        if args.algo == "maskformer":
            label, n = maskformer_labels(record, cfg)
        elif args.algo == "clue":
            label, n = clue_labels(record, cfg, params_by_subsystem)
        else:
            label, n = resolution_labels(record, cfg["oracle"]["resolution_fraction"])

        p, c, e = score_event(
            record,
            label,
            n,
            algo=args.algo,
            split_fraction=cfg["metrics"]["split_fraction"],
            min_overlap=cfg["metrics"]["min_overlap"],
            min_overlap_frac=cfg["metrics"]["min_overlap_frac"],
        )
        particles.append(p)
        clusters.append(c)
        events.append(e)

    particles = pd.concat(particles, ignore_index=True)
    clusters = pd.concat(clusters, ignore_index=True)
    events = pool(events)

    tag = args.tag or args.algo
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("particles", particles), ("clusters", clusters), ("events", events)):
        path = out_dir / f"{name}_{tag}.parquet"
        table.to_parquet(path, index=False)
        print(f"  wrote {path}  ({len(table)} rows, {path.stat().st_size / 1e6:.1f} MB)")

    wp = cfg["metrics"]["working_points"][0]
    print(f"\n  eff@{wp} (energy) {(particles['eff_e'] >= wp).mean():.3f}   pur@{wp} (energy) {(clusters['pur_e'] >= wp).mean():.3f}")
    print(f"  eff mean {particles['eff_e'].mean():.3f}   median {particles['eff_e'].median():.3f}"
          f"   |   pur mean {clusters['pur_e'].mean():.3f}   median {clusters['pur_e'].median():.3f}")
    print(f"  split  hits {particles['is_split'].mean():.3f}  energy {particles['is_split_e'].mean():.3f}")
    print(f"  merge  hits {clusters['is_merge'].mean():.3f}  energy {clusters['is_merge_e'].mean():.3f}")
    print(f"  frag {particles['frag_frac'].mean():.3f}   fake {(~clusters['matched']).mean():.3f}"
          f"   (match floor {cfg['metrics']['min_overlap_frac']:g} of the smaller total)")


if __name__ == "__main__":
    main()
