"""Hyperparameter search for the CLUE baseline.

CLUE exposes three parameters per pass, and their best values depend on the cell
pitch, which differs by an order of magnitude between the electromagnetic and
hadronic sections. Each detector collection therefore gets its own search.

Two objectives are available. ``cluster_f1`` maximises the harmonic mean of
energy-weighted purity and efficiency, which is the clustering task itself.
``jet`` maximises truth-to-reconstructed jet energy matching, which is the
physics observable. Both are supported so the effect of the choice can be
measured; whichever produced a reported number must be stated alongside it.

The objective uses exactly the metric defined in :mod:`src.evaluation.metrics`,
so the baseline is tuned toward the quantity it is later scored on.
"""

import optuna

from src.clue.pipeline import cluster_detector
from src.config import settings
from src.data.loader import hit_energy_mask
from src.data.truth import reconstructable_particles
from src.evaluation.jets import energy_match_score
from src.evaluation.metrics import MetricAccumulator

PARAMETER_NAMES = ["d_c_2d", "rho_c_2d", "d_o_2d", "d_c_3d", "rho_c_3d", "d_o_3d", "z_scale"]


def suggest_parameters(trial, detector):
    """Draw one CLUE parameter set for a detector from its search ranges.

    The linking radii are sampled as multiples of their density radius rather
    than independently, because a linking radius smaller than the density radius
    is not meaningful and the two are strongly correlated.

    Args:
        trial: the Optuna trial.
        detector: detector collection name.

    Returns:
        dict: the parameters :func:`src.clue.pipeline.cluster_detector` expects.
    """
    config = settings()["clue"]
    space = config["search"][config["coords"]][detector]
    factor_low, factor_high = config["search"]["outlier_distance_factor"]
    rho_c_3d_low, rho_c_3d_high = config["search"]["rho_c_3d"]
    log_scale = config["coords"] == "etaphi"

    d_c_2d = trial.suggest_float("d_c_2d", *space["d_c_2d"], log=log_scale)
    d_c_3d = trial.suggest_float("d_c_3d", *space["d_c_3d"], log=log_scale)
    return {
        "d_c_2d": d_c_2d,
        "rho_c_2d": trial.suggest_float("rho_c_2d", *space["rho_c_2d"], log=True),
        "d_o_2d": d_c_2d * trial.suggest_float("d_o_2d_factor", factor_low, factor_high),
        "d_c_3d": d_c_3d,
        "rho_c_3d": trial.suggest_float("rho_c_3d", rho_c_3d_low, rho_c_3d_high, log=True),
        "d_o_3d": d_c_3d * trial.suggest_float("d_o_3d_factor", factor_low, factor_high),
        "z_scale": trial.suggest_float("z_scale", *space["z_scale"], log=log_scale),
    }


def score_cluster_f1(events, detector, params):
    """Pooled purity/efficiency F1 for one parameter set on one detector."""
    config = settings()["clue"]
    accumulator = MetricAccumulator()

    for index in range(len(events)):
        event = events.iloc[index]
        energy_mask = hit_energy_mask(event)
        cluster_ids, selected = cluster_detector(
            event, detector, params, hit_mask=energy_mask,
            coords=config["coords"], backend=config["backend"],
        )
        if not selected.any():
            continue
        targets = reconstructable_particles(event, hit_mask=selected)
        accumulator.add_event(event, cluster_ids, targets, hit_mask=selected)

    return accumulator


def score_jet_matching(events, detector, params):
    """Pooled jet energy-matching score for one parameter set on one detector."""
    config = settings()["clue"]
    score_sum = 0.0
    n_truth_jets = 0

    for index in range(len(events)):
        event = events.iloc[index]
        energy_mask = hit_energy_mask(event)
        cluster_ids, selected = cluster_detector(
            event, detector, params, hit_mask=energy_mask,
            coords=config["coords"], backend=config["backend"],
        )
        if not selected.any():
            continue
        score, count = energy_match_score(event, cluster_ids, hit_mask=selected)
        score_sum += score
        n_truth_jets += count

    return score_sum, n_truth_jets


def make_objective(events, detector):
    """Build the Optuna objective for one detector, per the configured target."""
    objective_name = settings()["clue"]["objective"]

    def cluster_objective(trial):
        params = suggest_parameters(trial, detector)
        accumulator = score_cluster_f1(events, detector, params)
        trial.set_user_attr("purity", accumulator.purity())
        trial.set_user_attr("efficiency", accumulator.efficiency())
        trial.set_user_attr("n_targets", accumulator.n_targets)
        return accumulator.f1()

    def jet_objective(trial):
        params = suggest_parameters(trial, detector)
        score_sum, n_truth_jets = score_jet_matching(events, detector, params)
        if n_truth_jets == 0:
            return 0.0
        trial.set_user_attr("n_truth_jets", n_truth_jets)
        return score_sum / n_truth_jets

    if objective_name == "jet":
        return jet_objective
    if objective_name == "cluster_f1":
        return cluster_objective
    raise ValueError(
        f"Unknown clue.objective {objective_name!r}; expected cluster_f1 or jet."
    )


def tune_detector(events, detector, storage_url=None):
    """Run the search for one detector and return its best parameters.

    The sampler is seeded from the experiment file, so repeating a study on the
    same events reproduces the same result.

    Returns:
        tuple: ``(best_parameters, best_value)``.
    """
    config = settings()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["seed"]),
        study_name=f"clue_{config['clue']['coords']}_{config['clue']['objective']}_{detector}",
        storage=storage_url,
        load_if_exists=storage_url is not None,
    )
    study.optimize(make_objective(events, detector), n_trials=config["clue"]["optuna_trials"])

    best = dict(study.best_params)
    best["d_o_2d"] = best["d_c_2d"] * best.pop("d_o_2d_factor")
    best["d_o_3d"] = best["d_c_3d"] * best.pop("d_o_3d_factor")
    return {name: best[name] for name in PARAMETER_NAMES}, study.best_value
