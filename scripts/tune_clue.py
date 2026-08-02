"""Tune the CLUE parameters for every detector collection.

Runs one Optuna study per detector on the tuning subset of the train split, and
writes the best parameter set to ``results/clue_parameters_<coords>_<objective>.json``.
The coordinate system and the objective come from ``config/experiment.yaml``.

Usage::

    python -m scripts.tune_clue
"""

import json

from src.clue.tuning import tune_detector
from src.config import RESULTS_DIR, settings
from src.data.loader import load_calo_events, select_events


def main():
    config = settings()
    coords = config["clue"]["coords"]
    objective = config["clue"]["objective"]
    n_tune_events = config["splits"]["clue_tune_events"]

    print(f"Tuning CLUE on {n_tune_events} events "
          f"(coords={coords}, objective={objective}, "
          f"trials={config['clue']['optuna_trials']} per detector)")

    events = select_events(load_calo_events(), "train").iloc[:n_tune_events]
    storage = f"sqlite:///{RESULTS_DIR / 'clue_studies.db'}"

    best = {}
    for detector in config["detectors"]:
        print(f"\n=== {detector} ===")
        parameters, value = tune_detector(events, detector, storage_url=storage)
        best[detector] = {"parameters": parameters, "objective_value": value}
        print(f"best {objective} = {value:.4f}")
        for name, number in parameters.items():
            print(f"    {name:10s} {number:.6g}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"clue_parameters_{coords}_{objective}.json"
    with open(out_path, "w") as handle:
        json.dump({
            "coords": coords,
            "objective": objective,
            "n_tune_events": n_tune_events,
            "dataset": config["dataset"]["active"],
            "detectors": best,
        }, handle, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
