"""Characterise the clustering problem from the truth alone.

Runs before and independently of any clustering method. Because a cell can hold
energy from more than one particle, no assignment recovers the truth exactly, and
measuring how much energy is shared says how much of the task is soluble at all.
Quoting a method against that ceiling separates the difficulty of the sample from
the quality of the algorithm.

Writes per-particle contamination to ``results/truth_characterisation.parquet``
and the summary to ``results/truth_characterisation.json``.

Usage::

    python -m scripts.run_truth_study
"""

import json

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, settings
from src.data.loader import (hit_energy_mask, load_calo_events, select_events,
                             selection_summary)
from src.data.truth import (footprint_contamination, isolation_column,
                            isolation_flags, reconstructable_particles)


def characterise_events(events, detectors):
    """Per-particle contamination and isolation over a set of events."""
    frames = []
    for index in range(len(events)):
        event = events.iloc[index]
        energy_mask = hit_energy_mask(event)

        for detector in detectors:
            selected = energy_mask & (np.asarray(event["detector"]) == detector)
            if not selected.any():
                continue
            targets = reconstructable_particles(event, hit_mask=selected)
            if targets.empty:
                continue
            contamination = footprint_contamination(event, hit_mask=selected)
            merged = targets.merge(contamination, on="pid", how="inner")
            merged["event"] = index
            merged["detector"] = detector
            frames.append(merged)

    if not frames:
        return pd.DataFrame()
    return isolation_flags(pd.concat(frames, ignore_index=True))


def summarise(characterisation, thresholds):
    """Headline numbers: how much energy is shared, and how many particles are clean."""
    if characterisation.empty:
        return {"n_particles": 0}

    footprint = characterisation["footprint_energy"].sum()
    own = characterisation["own_energy"].sum()
    summary = {
        "n_particles": int(len(characterisation)),
        "shared_energy_fraction": float(1.0 - own / footprint) if footprint else 0.0,
        "median_contamination": float(characterisation["contamination"].median()),
        "isolated_fraction": {},
    }
    for threshold in thresholds:
        column = isolation_column(threshold)
        summary["isolated_fraction"][str(threshold)] = float(characterisation[column].mean())
    return summary


def main():
    config = settings()
    thresholds = config["metrics"]["contamination_thresholds"]

    all_events = load_calo_events()
    counts = selection_summary(all_events)
    print(f"Event selection: {counts['kept']} kept, {counts['rejected']} rejected "
          f"of {counts['total']}")

    events = select_events(all_events, "test")
    print(f"Characterising {len(events)} held-out events")

    characterisation = characterise_events(events, config["detectors"])
    summary = summarise(characterisation, thresholds)
    summary["event_selection"] = counts
    summary["n_events"] = len(events)
    summary["dataset"] = config["dataset"]["active"]

    RESULTS_DIR.mkdir(exist_ok=True)
    characterisation.to_parquet(RESULTS_DIR / "truth_characterisation.parquet")
    with open(RESULTS_DIR / "truth_characterisation.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nparticles            {summary['n_particles']}")
    print(f"shared energy        {summary['shared_energy_fraction']:.4f}")
    print(f"median contamination {summary['median_contamination']:.4f}")
    for threshold, fraction in summary["isolated_fraction"].items():
        print(f"isolated at {threshold:>5s}    {fraction:.4f}")
    print(f"\nWrote {RESULTS_DIR / 'truth_characterisation.json'}")


if __name__ == "__main__":
    main()
