"""Score a clustering against the truth partition and write the long tables.

Both algorithms come through here, and the scorer is never told which is which: it receives
a label per cell and nothing else. That is what makes the head-to-head fair in a way that
inspecting two configs cannot guarantee.

    python -m scripts.score --algo maskformer
    python -m scripts.score --algo maskformer_incidence
    python -m scripts.score --algo clue --params results/clue_parameters.json
    python -m scripts.score --algo oracle_seeded
    python -m scripts.score --algo oracle_resolution

`maskformer` and `maskformer_incidence` are the same checkpoint at the same working point,
differing only in which head decides who owns a contested cell -- the mask head's per-(query,
cell) sigmoid, or the incidence head's softmax over queries, the latter being the one trained
against the true energy fractions. Detection comes from the mask head in both, so the two
claim identical cells and the difference between their rows is the assignment rule alone.
Score both; the comparison between them is a result, not a tuning choice.

The two `oracle_*` algorithms are reference clusterings rather than methods under test; see
src/evaluation/oracle.py for what each one is and how to read it. They come through this same
entry point, and the scorer is no more aware of them than of anything else, which is the
point: a ceiling measured by different code from the methods it bounds is not a ceiling.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.clue.pipeline import SUBSYSTEMS, cluster_event
from src.config import RESULTS_DIR, settings, store_expectations, store_path
from src.evaluation.metrics import pool, score_event
from src.evaluation.oracle import geometric_labels, resolution_labels
from src.io.event_store import EventStore

# Untuned starting point, chosen against the measured cell energy scale rather than guessed;
# see the commentary on `clue.search` in config/experiment.yaml. Replaced by the Optuna
# result once tuning has run.
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


def maskformer_incidence_labels(record, cfg):
    """The same model and the same detected cells, with ownership from the incidence head.

    Scored as a separate algorithm rather than replacing `maskformer`, because the pair is
    the result: both come through this scorer unchanged, so the difference between the two
    rows isolates the assignment rule from everything else about the model.
    """
    mf = cfg["maskformer"]
    return record.maskformer_incidence_labels(
        mask_threshold=mf["mask_threshold"],
        object_threshold=mf["object_threshold"],
        min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
        incidence_floor=mf.get("incidence_floor", 0.0),
        restrict_to_mask=mf.get("incidence_restrict_to_mask", False),
    )


def clue_labels(record, cfg, params):
    return cluster_event(
        record,
        params,
        subsystems=tuple(cfg["detectors"]),
        coords=cfg["clue"]["coords"],
        backend=cfg["clue"]["backend"],
        min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--algo",
        choices=["maskformer", "maskformer_incidence", "clue", "oracle_geometric", "oracle_resolution"],
        required=True,
    )
    parser.add_argument("--store", type=Path, default=None, help="defaults to dataset.<active>.store")
    parser.add_argument("--params", type=Path, default=None, help="tuned CLUE parameters as JSON")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N events")
    parser.add_argument("--tag", default=None, help="suffix for the output files")
    args = parser.parse_args()

    cfg = settings()
    store = EventStore(args.store or store_path(), expect=store_expectations())
    print(f"{len(store)} events from {store.root}")
    print(f"  checkpoint  {store.meta['maskformer']['checkpoint'].rsplit('/', 1)[-1]}")
    print(f"  cuts        {store.meta['hit_selection']}  {store.meta['particle_selection']}")

    if args.params:
        tuned = json.loads(args.params.read_text())["subsystems"]
        params_by_subsystem = {name: entry["parameters"] for name, entry in tuned.items()}
        print(f"  clue params  tuned, from {args.params}")
    else:
        params_by_subsystem = {name: DEFAULT_CLUE_PARAMS for name in SUBSYSTEMS}
        if args.algo == "clue":
            print("  clue params  UNTUNED defaults -- run scripts.tune_clue before reporting these numbers")

    particles, clusters, events = [], [], []
    for i, record in enumerate(store):
        if args.limit and i >= args.limit:
            break
        if args.algo == "maskformer":
            label, n = maskformer_labels(record, cfg)
        elif args.algo == "maskformer_incidence":
            label, n = maskformer_incidence_labels(record, cfg)
        elif args.algo == "clue":
            label, n = clue_labels(record, cfg, params_by_subsystem)
        elif args.algo == "oracle_geometric":
            label, n = geometric_labels(record, depth_weight=cfg["oracle"]["depth_weight"])
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
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("particles", particles), ("clusters", clusters), ("events", events)):
        path = args.out / f"{name}_{tag}.parquet"
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
