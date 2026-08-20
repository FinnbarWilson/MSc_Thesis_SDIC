"""CLUE's parameter search space and Optuna objective, driven by `scripts.tune_clue`.

The objective calls :func:`~src.evaluation.metrics.score_event`, the same function that
produces the reported tables, so there is no second and cheaper definition of "good" here.

Tuning runs on a window disjoint from the reported one, which :func:`src.config._validate`
enforces. Each subsystem is tuned independently, because ECAL and HCAL cell geometries differ
by an order of magnitude and one density radius cannot mean the same thing in both. Stitching
the per-subsystem parts back together is a separate question with its own parameter,
``clue.link_radius``, which cannot be tuned in this loop and is scanned by
`scripts.scan_link_radius`.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import optuna
import pandas as pd

from src.clue.pipeline import PARAMETER_NAMES, cluster_subsystem
from src.config import active_dataset, clue_search, settings
from src.evaluation.metrics import score_event

SEARCH_PARAMETERS = ("d_c_2d", "rho_c_2d", "d_c_3d", "rho_c_3d", "depth_scale")


class _RestrictedTruth:
    """An EventRecord view whose truth partition is masked to one subsystem.

    A per-subsystem trial should not be penalised for particles it was never shown, so cells
    outside the subsystem are marked unowned. Everything else passes straight through.
    """

    def __init__(self, record, truth_label):
        self._record = record
        self.truth_label = truth_label

    @property
    def truth_deposit(self):
        """Masked to match `truth_label`.

        Without this override, ``__getattr__`` would delegate to the underlying record and
        compute the deposit from the unmasked labels.
        """
        deposit = self._record.truth_deposit.copy()
        deposit[self.truth_label < 0] = 0.0
        return deposit

    def __getattr__(self, name):
        return getattr(self._record, name)


def suggest_parameters(trial: optuna.Trial, subsystem: str, config: Mapping | None = None) -> dict[str, float]:
    """Draw one parameter set for `subsystem` from the configured search ranges.

    The outlier distances are sampled as a multiple of their density radius rather than
    independently, since one below the density radius has no effect at all. Everything is
    sampled logarithmically, the ranges spanning two to three decades.
    """
    config = config or settings()["clue"]
    ranges = clue_search(subsystem)

    params: dict[str, float] = {}
    for name in SEARCH_PARAMETERS:
        low, high = ranges[name]
        params[name] = trial.suggest_float(name, low, high, log=True)

    low, high = config["search"]["outlier_distance_factor"]
    params["d_o_2d"] = params["d_c_2d"] * trial.suggest_float("d_o_2d_factor", low, high)
    params["d_o_3d"] = params["d_c_3d"] * trial.suggest_float("d_o_3d_factor", low, high)
    return params


#: A trial producing more clusters than this per truth particle has shattered the event into
#: near-singletons and cannot win, and the overlap matrix and Hungarian solve both scale with
#: the cluster count, so it is rejected without being scored.
MAX_CLUSTERS_PER_PARTICLE = 8


def score_parameters(
    records: Sequence,
    subsystem: str,
    params: Mapping[str, float],
    config: Mapping | None = None,
    metrics: Mapping | None = None,
) -> dict[str, float]:
    """Run CLUE over `records` for one subsystem and score it with the reporting metric."""
    config = config or settings()["clue"]
    metrics = metrics or settings()["metrics"]
    working_point = metrics["working_points"][0]

    particle_tables, cluster_tables = [], []
    for record in records:
        ids, selected = cluster_subsystem(record, subsystem, params, coords=config["coords"], backend=config["backend"])
        if not selected.any():
            continue

        labels = np.full(record.n_hits, -1, dtype=np.int32)
        clustered = ids >= 0
        n_clusters = 0
        if clustered.any():
            used, compact = np.unique(ids[clustered], return_inverse=True)
            labels[clustered] = compact.astype(np.int32)
            n_clusters = int(used.size)

        if n_clusters > MAX_CLUSTERS_PER_PARTICLE * max(record.n_particles, 1):
            return {"efficiency": 0.0, "purity": 0.0, "f1": 0.0}

        masked = record.truth_label.copy()
        masked[~selected] = -1

        particles, clusters, _ = score_event(
            _RestrictedTruth(record, masked),
            labels,
            n_clusters,
            algo="clue",
            split_fraction=metrics["split_fraction"],
            min_overlap=metrics["min_overlap"],
        )
        # Particles with no cell in this subsystem are not this trial's problem.
        particle_tables.append(particles[particles["n_hits"] > 0])
        cluster_tables.append(clusters)

    if not particle_tables:
        return {"efficiency": 0.0, "purity": 0.0, "f1": 0.0}

    particles = pd.concat(particle_tables, ignore_index=True)
    clusters = pd.concat(cluster_tables, ignore_index=True)
    efficiency = float((particles["eff_e"] >= working_point).mean()) if len(particles) else 0.0
    purity = float((clusters["pur_e"] >= working_point).mean()) if len(clusters) else 0.0
    total = efficiency + purity
    return {
        "efficiency": efficiency,
        "purity": purity,
        "f1": (2 * efficiency * purity / total) if total > 0 else 0.0,
    }


def make_objective(records: Sequence, subsystem: str):
    """Build the Optuna objective for one subsystem."""
    config = settings()["clue"]
    metrics = settings()["metrics"]

    def objective(trial: optuna.Trial) -> float:
        scores = score_parameters(records, subsystem, suggest_parameters(trial, subsystem, config), config, metrics)
        trial.set_user_attr("efficiency", scores["efficiency"])
        trial.set_user_attr("purity", scores["purity"])
        return scores["f1"]

    return objective


def tune_subsystem(records: Sequence, subsystem: str, storage_url: str | None = None) -> tuple[dict[str, float], float]:
    """Search for the best parameters for one subsystem.

    Prints a warning for any optimum landing in the outer 5% of its log range: the true optimum
    is then outside the box, and reporting CLUE at the edge would understate the baseline.

    Args:
        records: the tuning events, decoded once and reused by every trial.
        subsystem: which subsystem to tune.
        storage_url: optional Optuna storage URL, which also enables resuming a study.

    Returns:
        ``(parameters, objective_value)``, the parameters being the seven entries of
        :data:`~src.clue.pipeline.PARAMETER_NAMES` with the outlier distances multiplied out.
    """
    config = settings()["clue"]

    # The dataset is in the study name: with `--storage` and `load_if_exists`, a pu200 run
    # would otherwise resume the pu0 study of the same name and return its trials.
    study = optuna.create_study(
        study_name=f"clue_{active_dataset()}_{config['coords']}_{subsystem}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings()["seed"]),
        storage=storage_url,
        load_if_exists=storage_url is not None,
    )
    study.optimize(make_objective(records, subsystem), n_trials=config["optuna_trials"])

    best = dict(study.best_params)
    params = {name: best[name] for name in SEARCH_PARAMETERS}
    params["d_o_2d"] = params["d_c_2d"] * best["d_o_2d_factor"]
    params["d_o_3d"] = params["d_c_3d"] * best["d_o_3d_factor"]

    # Position within the log range, because that is how the parameters are sampled. Testing
    # near-equality to the endpoint would essentially never fire: the optimiser returns something
    # a few percent inside the bound, which is just as much a sign the range is wrong.
    ranges = clue_search(subsystem)
    for name in SEARCH_PARAMETERS:
        low, high = ranges[name]
        position = np.log(params[name] / low) / np.log(high / low)
        if position < 0.05 or position > 0.95:
            edge = "lower" if position < 0.5 else "upper"
            print(
                f"  WARNING {subsystem}.{name} = {params[name]:.4g} sits in the {edge} 5% of its "
                f"search range [{low:g}, {high:g}]; widen it and re-run before reporting."
            )

    assert set(params) == set(PARAMETER_NAMES)
    return params, float(study.best_value)
