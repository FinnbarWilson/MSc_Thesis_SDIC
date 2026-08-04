"""Tune CLUE's parameters on the tuning window, one Optuna study per subsystem.

    python -m scripts.tune_clue

Writes ``results/<active dataset>/clue_parameters.json``, which
``scripts.score --algo clue --params`` reads. Tuning runs on a window disjoint from the
reported events, and :mod:`src.config` refuses to load a configuration where the two overlap
-- tuning CLUE on the events it is reported on would hand it an advantage the MaskFormer does
not have.

**At pu200, read the range warnings before the result.** The search ranges are calibrated
against pu0 cell densities, and `rho_c` is a local energy density that pileup moves. Any
optimum landing in the outer 5% of its log range is reported as a warning, and means the true
optimum is outside the box: widen that range in `dataset.pu200.overrides.clue.search` and
re-run rather than reporting the edge. A baseline tuned against the wrong box is under-tuned,
which is the one way this comparison can be unfair to CLUE.
"""

import argparse
import json
from pathlib import Path

from src.clue.tuning import tune_subsystem
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.io.event_store import EventStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=None, help="defaults to dataset.<active>.tune_store")
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to results/<active dataset>/clue_parameters.json")
    parser.add_argument("--events", type=int, default=0,
                        help="cap the number of tuning events; at pu200 this is the memory knob, "
                             "since every trial re-runs over all of them and they are all held decoded")
    parser.add_argument("--trials", type=int, default=0, help="override clue.optuna_trials")
    parser.add_argument("--storage", default=None,
                        help="optuna storage URL, e.g. sqlite:///results/pu0/clue.db")
    args = parser.parse_args()

    cfg = settings()
    if args.trials:
        cfg["clue"]["optuna_trials"] = args.trials
    out_path = args.out or (results_dir() / "clue_parameters.json")
    print(describe())

    store = EventStore(args.store or store_path("tune_store"), expect=store_expectations())
    # Every Optuna trial re-runs the whole pipeline over every record, so they are decoded once
    # and held. That is the right trade at pu0 and the thing to watch at pu200: the cost is
    # linear in total cells, not in events, so the hit count is what gets printed.
    records = [store[i] for i in range(min(args.events or len(store), len(store)))]
    total_hits = sum(record.n_hits for record in records)
    print(f"tuning on {len(records)} events from {store.root}")
    print(f"  {total_hits:,} cells held in memory ({total_hits / max(len(records), 1):,.0f} per event); "
          f"use --events to cap this")

    # The dataset is recorded in the file, not only in its path, so a parameter set that gets
    # copied or passed with --params still says which pileup condition it was tuned on.
    result = {
        "dataset": cfg["dataset"]["active"],
        "tune_store": str(store.root),
        "coords": cfg["clue"]["coords"],
        "n_tune_events": len(records),
        "subsystems": {},
    }
    for subsystem in cfg["detectors"]:
        print(f"\n=== {subsystem} ===", flush=True)
        params, value = tune_subsystem(records, subsystem, storage_url=args.storage)
        result["subsystems"][subsystem] = {"parameters": params, "objective": value}
        print(f"  best f1 {value:.4f}  " + "  ".join(f"{k}={v:.4g}" for k, v in params.items()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
