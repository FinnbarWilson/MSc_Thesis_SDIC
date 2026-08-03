"""Tune CLUE's parameters on the tuning window, one Optuna study per subsystem.

    python -m scripts.tune_clue

Writes ``results/clue_parameters.json``, which ``scripts.score --algo clue --params`` reads.
Tuning runs on a window disjoint from the reported events, and :mod:`src.config` refuses to
load a configuration where the two overlap -- tuning CLUE on the events it is reported on
would hand it an advantage the MaskFormer does not have.
"""

import argparse
import json
from pathlib import Path

from src.clue.tuning import tune_subsystem
from src.config import RESULTS_DIR, settings, store_expectations, store_path
from src.io.event_store import EventStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=None, help="defaults to dataset.<active>.tune_store")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "clue_parameters.json")
    parser.add_argument("--events", type=int, default=0, help="cap the number of tuning events")
    parser.add_argument("--trials", type=int, default=0, help="override clue.optuna_trials")
    parser.add_argument("--storage", default=None, help="optuna storage URL, e.g. sqlite:///results/clue.db")
    args = parser.parse_args()

    cfg = settings()
    if args.trials:
        cfg["clue"]["optuna_trials"] = args.trials

    store = EventStore(args.store or store_path("tune_store"), expect=store_expectations())
    records = [store[i] for i in range(min(args.events or len(store), len(store)))]
    print(f"tuning on {len(records)} events from {store.root}")

    result = {"coords": cfg["clue"]["coords"], "n_tune_events": len(records), "subsystems": {}}
    for subsystem in cfg["detectors"]:
        print(f"\n=== {subsystem} ===", flush=True)
        params, value = tune_subsystem(records, subsystem, storage_url=args.storage)
        result["subsystems"][subsystem] = {"parameters": params, "objective": value}
        print(f"  best f1 {value:.4f}  " + "  ".join(f"{k}={v:.4g}" for k, v in params.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
