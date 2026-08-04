"""Jet reconstruction and jet-level scoring.

Purity and efficiency score clusters in the abstract; the observable a physics
analysis actually uses is the jet. Both the truth record and a method's output
are turned into jets the same way, so the comparison at jet level is as
controlled as the comparison at cluster level.

Truth jets are built from each particle's total deposited calorimeter energy
rather than its generated momentum, which puts truth and reconstructed jets on
the same energy scale. The per-detector sampling calibration is then a constant
common to both and cancels in the response ratio, so no calibration is needed
for the response to be meaningful.

Clustering uses anti-kt with E-scheme recombination through FastJet.
"""

import numpy as np
import pandas as pd

import fastjet

from src.config import settings
from src.data.loader import ENERGY_COLUMN

TWO_PI = 2.0 * np.pi

JET_COLUMNS = ["E", "pt", "y", "eta", "phi", "n_constituents", "constituents"]
MATCH_COLUMNS = ["truth_index", "reco_index", "E_truth", "E_reco", "response",
                 "delta_r", "delta_y", "delta_phi"]


def delta_phi(first, second):
    """Smallest signed difference between two azimuths, wrapped into (-pi, pi]."""
    return np.mod(first - second + np.pi, TWO_PI) - np.pi


def _to_dataframe(jets):
    """Repackage FastJet PseudoJets as the standard jet DataFrame."""
    return pd.DataFrame({
        "E": [jet.e() for jet in jets],
        "pt": [jet.pt() for jet in jets],
        "y": [jet.rap() for jet in jets],
        "eta": [jet.eta() for jet in jets],
        "phi": [jet.phi_std() for jet in jets],
        "n_constituents": [len(jet.constituents()) for jet in jets],
        "constituents": [[c.user_index() for c in jet.constituents()] for jet in jets],
    })[JET_COLUMNS]


def cluster_jets(eta, phi, energy, radius=None):
    """Cluster massless constituents into anti-kt jets.

    Each constituent becomes a massless four-vector, so ``pt = E / cosh(eta)``.
    Constituents with non-positive energy are dropped.

    Args:
        eta, phi: constituent directions.
        energy: constituent energies in GeV.
        radius: anti-kt radius; defaults to ``jets.radius``.

    Returns:
        pd.DataFrame: one row per jet, sorted by descending energy.
    """
    if radius is None:
        radius = settings()["jets"]["radius"]

    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    energy = np.asarray(energy, dtype=float)
    keep = np.flatnonzero(energy > 0)
    if keep.size == 0:
        return pd.DataFrame(columns=JET_COLUMNS)

    transverse = energy[keep] / np.cosh(eta[keep])
    particles = []
    for position, index in enumerate(keep):
        pseudo_jet = fastjet.PseudoJet(
            transverse[position] * np.cos(phi[index]),
            transverse[position] * np.sin(phi[index]),
            transverse[position] * np.sinh(eta[index]),
            energy[index],
        )
        pseudo_jet.set_user_index(int(index))
        particles.append(pseudo_jet)

    definition = fastjet.JetDefinition(fastjet.antikt_algorithm, radius)
    sequence = fastjet.ClusterSequence(particles, definition)
    return _to_dataframe(fastjet.sorted_by_E(sequence.inclusive_jets()))


def _energy_weighted_directions(codes, energies, x, y, z, n_groups):
    """Energy-weighted centroid direction of each group of cells."""
    total = np.bincount(codes, weights=energies, minlength=n_groups)
    centre_x = np.bincount(codes, weights=energies * x, minlength=n_groups) / total
    centre_y = np.bincount(codes, weights=energies * y, minlength=n_groups) / total
    centre_z = np.bincount(codes, weights=energies * z, minlength=n_groups) / total
    phi = np.arctan2(centre_y, centre_x)
    theta = np.arctan2(np.hypot(centre_x, centre_y), centre_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = -np.log(np.tan(theta / 2.0))
    return total, eta, phi


def truth_constituents(event, hit_mask=None, min_energy_gev=None):
    """One pseudo-particle per truth particle, carrying its deposited energy.

    The direction is the particle's deposit-energy-weighted centroid seen from
    the interaction point. The energy floor removes the large tail of very small
    shower fragments, which would otherwise dominate the constituent count
    without carrying meaningful energy.

    Args:
        event: one event row.
        hit_mask: optional boolean mask over cells.
        min_energy_gev: override for ``jets.constituent_min_energy_gev``.

    Returns:
        tuple: ``(constituents, total_energy)`` where ``constituents`` has
        columns ``pid``, ``E``, ``eta`` and ``phi``, and ``total_energy`` is the
        deposited energy before the floor, so the retained fraction can be
        reported.
    """
    if min_energy_gev is None:
        min_energy_gev = settings()["jets"]["constituent_min_energy_gev"]

    ids = np.asarray(event["contrib_particle_ids"], dtype=object)
    energies = np.asarray(event["contrib_energies"], dtype=object)
    x = np.asarray(event["x"], dtype=float)
    y = np.asarray(event["y"], dtype=float)
    z = np.asarray(event["z"], dtype=float)

    if hit_mask is not None:
        mask = np.asarray(hit_mask)
        ids, energies, x, y, z = ids[mask], energies[mask], x[mask], y[mask], z[mask]
    if len(ids) == 0:
        return pd.DataFrame(columns=["pid", "E", "eta", "phi"]), 0.0

    repeats = np.fromiter((len(entry) for entry in ids), dtype=int)
    flat_pids = np.concatenate([np.asarray(entry) for entry in ids])
    flat_energy = np.concatenate([np.asarray(entry) for entry in energies]).astype(float)

    unique_pids, codes = np.unique(flat_pids, return_inverse=True)
    total, eta, phi = _energy_weighted_directions(
        codes, flat_energy, np.repeat(x, repeats), np.repeat(y, repeats),
        np.repeat(z, repeats), len(unique_pids),
    )

    keep = total > min_energy_gev
    constituents = pd.DataFrame({
        "pid": unique_pids[keep].astype(np.int64),
        "E": total[keep],
        "eta": eta[keep],
        "phi": phi[keep],
    })
    return constituents, float(total.sum())


def reco_constituents(event, cluster_ids):
    """One pseudo-particle per reconstructed cluster.

    Cells the method called noise are simply absent, so energy lost to the
    density threshold shows up directly as a jet response below one rather than
    being silently restored.

    Returns:
        pd.DataFrame with columns ``cluster_id``, ``E``, ``eta`` and ``phi``.
    """
    cluster_ids = np.asarray(cluster_ids)
    selected = cluster_ids >= 0
    if not selected.any():
        return pd.DataFrame(columns=["cluster_id", "E", "eta", "phi"])

    energies = np.asarray(event[ENERGY_COLUMN], dtype=float)[selected]
    unique_ids, codes = np.unique(cluster_ids[selected], return_inverse=True)
    total, eta, phi = _energy_weighted_directions(
        codes, energies,
        np.asarray(event["x"], dtype=float)[selected],
        np.asarray(event["y"], dtype=float)[selected],
        np.asarray(event["z"], dtype=float)[selected],
        len(unique_ids),
    )
    return pd.DataFrame({"cluster_id": unique_ids, "E": total, "eta": eta, "phi": phi})


def match_jets(truth_jets, reco_jets, max_delta_r=None):
    """Match truth jets to reconstructed jets, one to one.

    Truth jets are visited in descending energy, so the hardest jet claims its
    partner first, and each takes the nearest unclaimed reconstructed jet within
    the match cone.

    Args:
        truth_jets, reco_jets: outputs of :func:`cluster_jets`.
        max_delta_r: match cone; defaults to ``jets.match_delta_r``.

    Returns:
        pd.DataFrame: one row per matched pair, with the energy ``response``
        (reconstructed over truth) and the angular separation.
    """
    if max_delta_r is None:
        max_delta_r = settings()["jets"]["match_delta_r"]
    if truth_jets.empty or reco_jets.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    truth_y = truth_jets["y"].to_numpy()
    truth_phi = truth_jets["phi"].to_numpy()
    truth_e = truth_jets["E"].to_numpy()
    reco_y = reco_jets["y"].to_numpy()
    reco_phi = reco_jets["phi"].to_numpy()
    reco_e = reco_jets["E"].to_numpy()

    taken = np.zeros(len(reco_jets), dtype=bool)
    rows = []
    for truth_index in np.argsort(-truth_e):
        d_y = reco_y - truth_y[truth_index]
        d_phi = delta_phi(reco_phi, truth_phi[truth_index])
        separation = np.hypot(d_y, d_phi)
        separation[taken] = np.inf
        reco_index = int(np.argmin(separation))
        if separation[reco_index] > max_delta_r:
            continue
        taken[reco_index] = True
        rows.append((
            int(truth_index), reco_index, truth_e[truth_index], reco_e[reco_index],
            reco_e[reco_index] / truth_e[truth_index], separation[reco_index],
            d_y[reco_index], d_phi[reco_index],
        ))
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def energy_match_score(event, cluster_ids, hit_mask=None, min_jet_energy=None):
    """How well one event's jet energies are reconstructed, for tuning.

    Each matched jet scores ``min(r, 1/r)`` with ``r`` the energy response, which
    is one for a perfect match and falls away symmetrically for under- and
    over-clustering. Scoring symmetrically stops a constant low bias from being
    rewarded, which a bare resolution would allow.

    Returns:
        tuple: ``(score_sum, n_truth_jets)``. Pooling these across events gives
        an objective in [0, 1] that folds together match efficiency, since an
        unmatched truth jet adds nothing to the numerator but one to the
        denominator, and energy response.
    """
    if min_jet_energy is None:
        min_jet_energy = settings()["jets"]["min_energy_gev"]

    truth, _ = truth_constituents(event, hit_mask)
    if truth.empty:
        return 0.0, 0
    truth_jets = cluster_jets(truth["eta"], truth["phi"], truth["E"])
    significant = truth_jets[truth_jets["E"] > min_jet_energy]
    if significant.empty:
        return 0.0, 0

    reco = reco_constituents(event, cluster_ids)
    if reco.empty:
        return 0.0, len(significant)
    reco_jets = cluster_jets(reco["eta"], reco["phi"], reco["E"])

    matches = match_jets(significant.reset_index(drop=True), reco_jets)
    if matches.empty:
        return 0.0, len(significant)
    response = matches["response"].to_numpy()
    return float(np.minimum(response, 1.0 / response).sum()), len(significant)
