"""Score the tuned CLUE baseline on the held-out test split.

Applies the parameters written by ``scripts.tune_clue`` to events neither the
search nor any model has seen, and reports pooled energy-weighted purity and
efficiency per detector and for the whole event, together with the jet-level
response. These are the numbers that go in the report.

Usage::

    python -m scripts.evaluate_clue
"""

import json

import numpy as np

from src.clue.pipeline import cluster_event
from src.config import RESULTS_DIR, settings
from src.data.loader import hit_energy_mask, load_calo_events, select_events
from src.data.truth import reconstructable_particles
from src.evaluation.jets import (cluster_jets, match_jets, reco_constituents,
                                 truth_constituents)
from src.evaluation.metrics import MetricAccumulator


def load_parameters(coords, objective):
    """Read the tuned parameters, keyed by detector."""
    path = RESULTS_DIR / f"clue_parameters_{coords}_{objective}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m scripts.tune_clue` first."
        )
    with open(path) as handle:
        saved = json.load(handle)
    return {name: entry["parameters"] for name, entry in saved["detectors"].items()}


def evaluate_clusters(events, parameters, config):
    """Pooled cluster metrics for the whole event and per detector."""
    whole_event = MetricAccumulator()
    per_detector = {name: MetricAccumulator() for name in config["detectors"]}

    for index in range(len(events)):
        event = events.iloc[index]
        energy_mask = hit_energy_mask(event)
        merged, detector_results = cluster_event(
            event, parameters, config["detectors"], hit_mask=energy_mask,
            coords=config["clue"]["coords"], backend=config["clue"]["backend"],
        )
        targets = reconstructable_particles(event, hit_mask=energy_mask)
        whole_event.add_event(event, merged, targets, hit_mask=energy_mask)

        for detector, (ids, selected) in detector_results.items():
            local_targets = reconstructable_particles(event, hit_mask=selected)
            per_detector[detector].add_event(event, ids, local_targets, hit_mask=selected)

    return whole_event, per_detector


def evaluate_jets(events, parameters, config):
    """Jet energy response and match efficiency over the split."""
    responses = []
    n_truth_total = 0
    n_matched_total = 0
    min_energy = config["jets"]["min_energy_gev"]

    for index in range(len(events)):
        event = events.iloc[index]
        energy_mask = hit_energy_mask(event)
        merged, _ = cluster_event(
            event, parameters, config["detectors"], hit_mask=energy_mask,
            coords=config["clue"]["coords"], backend=config["clue"]["backend"],
        )

        truth, _ = truth_constituents(event, hit_mask=energy_mask)
        if truth.empty:
            continue
        truth_jets = cluster_jets(truth["eta"], truth["phi"], truth["E"])
        truth_jets = truth_jets[truth_jets["E"] > min_energy].reset_index(drop=True)
        if truth_jets.empty:
            continue
        n_truth_total += len(truth_jets)

        reco = reco_constituents(event, merged)
        if reco.empty:
            continue
        reco_jets = cluster_jets(reco["eta"], reco["phi"], reco["E"])
        matches = match_jets(truth_jets, reco_jets)
        n_matched_total += len(matches)
        responses.extend(matches["response"].tolist())

    if not responses:
        return {"n_truth_jets": n_truth_total, "n_matched": 0}

    values = np.asarray(responses, dtype=float)
    return {
        "n_truth_jets": n_truth_total,
        "n_matched": n_matched_total,
        "match_efficiency": n_matched_total / n_truth_total if n_truth_total else 0.0,
        "response_median": float(np.median(values)),
        "response_mean": float(values.mean()),
        "response_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }


def main():
    config = settings()
    coords = config["clue"]["coords"]
    objective = config["clue"]["objective"]
    working_points = config["metrics"]["working_points"]

    parameters = load_parameters(coords, objective)
    events = select_events(load_calo_events(), "test")
    print(f"Evaluating CLUE on {len(events)} held-out events "
          f"(coords={coords}, tuned for {objective})")

    whole_event, per_detector = evaluate_clusters(events, parameters, config)
    jet_summary = evaluate_jets(events, parameters, config)

    report = {
        "dataset": config["dataset"]["active"],
        "coords": coords,
        "objective": objective,
        "n_events": len(events),
        "whole_event": whole_event.summary(working_points),
        "per_detector": {
            name: accumulator.summary(working_points)
            for name, accumulator in per_detector.items()
        },
        "jets": jet_summary,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"clue_evaluation_{coords}_{objective}.json"
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)

    summary = report["whole_event"]
    print(f"\npurity     {summary['purity']:.4f}")
    print(f"efficiency {summary['efficiency']:.4f}")
    print(f"F1         {summary['f1']:.4f}")
    print(f"targets    {summary['n_targets']}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
