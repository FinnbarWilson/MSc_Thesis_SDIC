"""Per-object metrics, emitted as long tables.

:func:`score_event` writes one row per truth particle and one per predicted cluster; every
aggregate in the report is a later selection over those tables.

Metrics come in hit-counted and energy-weighted forms, and the ``_e`` suffixed energy ones are
primary: cell energies span orders of magnitude, and energies are calibrated per subsystem
before anything is summed. Efficiency is taken per truth particle rather than per cluster, so a
particle merged into a neighbour counts as a miss instead of leaving the denominator, and rows
are pooled over events rather than averaged per event.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.matching import contamination, fragmentation, hungarian_match, overlap_matrix

TWO_PI = 2.0 * np.pi

PARTICLE_COLUMNS = [
    "sample_id", "algo", "particle_row", "particle_id", "pdg_id", "particle_class",
    "p_energy", "p_pt", "p_eta", "p_phi",
    "n_hits", "e_dep_calib", "e_dep_raw",
    "dr_min", "n_within_02",
    "matched", "cluster_row",
    "eff_n", "eff_e", "pur_n", "pur_e",
    "e_reco_calib", "response", "response_matched",
    "n_frag", "is_split", "n_frag_e", "is_split_e", "n_touched", "frag_frac",
    "e_lost_noise", "e_lost_other",
]

CLUSTER_COLUMNS = [
    "sample_id", "algo", "cluster_row", "n_hits", "e_raw", "e_calib",
    "matched", "particle_row", "pur_n", "pur_e", "eff_n", "eff_e",
    "n_owners", "is_merge", "n_owners_e", "is_merge_e", "e_unowned", "f_ecal",
]


def delta_phi(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Smallest signed difference between two azimuths, wrapped into (-pi, pi]."""
    return np.mod(first - second + np.pi, TWO_PI) - np.pi


def local_density(eta: np.ndarray, phi: np.ndarray, radius: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """How crowded each particle's neighbourhood is.

    Directions come from the particles' generator momenta rather than their shower centroids,
    so this is a property of the event and not of either algorithm's output.

    Returns:
        ``(dr_min, n_within)``: distance to the nearest other target particle (``inf`` when
        alone) and the number of other target particles within `radius`.
    """
    n = int(eta.size)
    if n == 0:
        return np.empty(0), np.empty(0, dtype=np.int32)
    if n == 1:
        return np.array([np.inf]), np.zeros(1, dtype=np.int32)

    dr = np.hypot(eta[:, None] - eta[None, :], delta_phi(phi[:, None], phi[None, :]))
    np.fill_diagonal(dr, np.inf)
    return dr.min(axis=1), (dr < radius).sum(axis=1).astype(np.int32)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros(np.shape(num), dtype=np.float64), where=np.asarray(den) > 0)


def score_event(
    record,
    pred_label: np.ndarray,
    n_pred: int,
    algo: str,
    split_fraction: float = 0.10,
    min_overlap: float = 0.0,
    min_overlap_frac: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score one event's clustering against its truth partition.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        pred_label: per cell, the predicted cluster index, -1 where unclustered.
        n_pred: number of predicted clusters.
        algo: label carried through to the output tables, e.g. ``"clue"``.
        split_fraction: share of a truth particle a cluster must hold to count towards
            splitting and merging.
        min_overlap: passed to the matcher; pairs at or below it are not matches.
        min_overlap_frac: relative match floor, as a fraction of the smaller of the two
            totals. See :func:`~src.evaluation.matching.hungarian_match`.

    Returns:
        ``(particles, clusters, event)``: one row per truth particle, one row per predicted
        cluster, and one summary dictionary.
    """
    n_truth = int(record.n_particles)
    truth_label = record.truth_label
    pred_label = np.asarray(pred_label)

    # E_ia, calibrated: the energy this particle put in this cell.
    deposit = record.truth_deposit * record.calibration[record.subsystem]
    cell_calib = record.energy_calib

    overlap_n = overlap_matrix(truth_label, pred_label, None, n_truth, n_pred)
    overlap_e = overlap_matrix(truth_label, pred_label, deposit, n_truth, n_pred)

    owned = truth_label >= 0
    clustered = pred_label >= 0
    truth_total_n = np.bincount(truth_label[owned], minlength=n_truth).astype(np.float64)
    truth_total_e = np.bincount(truth_label[owned], weights=deposit[owned], minlength=n_truth)
    pred_total_n = np.bincount(pred_label[clustered], minlength=n_pred).astype(np.float64)
    pred_total_e = np.bincount(pred_label[clustered], weights=cell_calib[clustered], minlength=n_pred)
    pred_total_raw = np.bincount(pred_label[clustered], weights=record.energy[clustered], minlength=n_pred)

    # Matched on calibrated energy overlap, so the pairing follows where the energy went
    # rather than how many low-energy cells coincided.
    match = hungarian_match(
        overlap_e,
        min_overlap=min_overlap,
        truth_total=truth_total_e,
        pred_total=pred_total_e,
        min_overlap_frac=min_overlap_frac,
    )

    # Split and merge in both weightings; the energy pair is the one reported.
    n_frag = fragmentation(overlap_n, truth_total_n, split_fraction)
    n_owners = contamination(overlap_n, truth_total_n, split_fraction)
    n_frag_e = fragmentation(overlap_e, truth_total_e, split_fraction)
    n_owners_e = contamination(overlap_e, truth_total_e, split_fraction)

    # Both `n_frag` forms have a blind spot: a particle spread over more than 1/split_fraction
    # clusters gives no cluster a large enough share, so it counts as zero fragments and is not
    # flagged as split. `frag_frac`, the share outside the largest piece, has no such blind spot
    # and is the one plotted.
    n_touched = (overlap_e > 0).sum(axis=1).astype(np.int32) if overlap_e.size else np.zeros(n_truth, dtype=np.int32)
    best_piece = overlap_e.max(axis=1) if overlap_e.size else np.zeros(n_truth)
    frag_frac = 1.0 - _safe_div(best_piece, truth_total_e)

    matched = np.zeros(n_truth, dtype=bool)
    cluster_row = np.full(n_truth, -1, dtype=np.int64)
    eff_n = np.zeros(n_truth)
    eff_e = np.zeros(n_truth)
    pur_n = np.zeros(n_truth)
    pur_e = np.zeros(n_truth)
    e_reco = np.zeros(n_truth)
    e_matched = np.zeros(n_truth)

    if match.n_matched:
        t, c = match.truth_index, match.pred_index
        matched[t] = True
        cluster_row[t] = c
        eff_n[t] = _safe_div(overlap_n[t, c], truth_total_n[t])
        eff_e[t] = _safe_div(overlap_e[t, c], truth_total_e[t])
        pur_n[t] = _safe_div(overlap_n[t, c], pred_total_n[c])
        pur_e[t] = _safe_div(overlap_e[t, c], pred_total_e[c])
        e_reco[t] = pred_total_e[c]
        e_matched[t] = overlap_e[t, c]

    # Where a particle's energy went when it missed its matched cluster: cells no cluster
    # claimed, against cells another cluster took. Different failures, kept apart.
    lost = owned & ~clustered
    e_lost_noise = np.bincount(truth_label[lost], weights=deposit[lost], minlength=n_truth)
    e_lost_other = np.maximum(truth_total_e - e_matched - e_lost_noise, 0.0)

    dr_min, n_within = local_density(record.particle_eta, record.particle_phi)

    particles = pd.DataFrame({
        "sample_id": np.full(n_truth, record.sample_id, dtype=np.int64),
        "algo": algo,
        "particle_row": np.arange(n_truth, dtype=np.int64),
        "particle_id": record.particle_id,
        "pdg_id": record.particle_pdg_id,
        "particle_class": record.particle_class,
        "p_energy": record.particle_energy,
        "p_pt": record.particle_pt,
        "p_eta": record.particle_eta,
        "p_phi": record.particle_phi,
        "n_hits": truth_total_n.astype(np.int32),
        "e_dep_calib": truth_total_e,
        "e_dep_raw": record.particle_energy_calo_sum,
        "dr_min": dr_min,
        "n_within_02": n_within,
        "matched": matched,
        "cluster_row": cluster_row,
        "eff_n": eff_n,
        "eff_e": eff_e,
        "pur_n": pur_n,
        "pur_e": pur_e,
        "e_reco_calib": e_reco,
        # `response` divides the whole matched cluster by the particle's deposit, so
        # contamination raises it above 1; `response_matched` divides only the shared part and
        # is bounded by 1.
        "response": _safe_div(e_reco, truth_total_e),
        "response_matched": _safe_div(e_matched, truth_total_e),
        "n_frag": n_frag,
        "is_split": n_frag > 1,
        "n_frag_e": n_frag_e,
        "is_split_e": n_frag_e > 1,
        "n_touched": n_touched,
        "frag_frac": frag_frac,
        "e_lost_noise": e_lost_noise,
        "e_lost_other": e_lost_other,
    })[PARTICLE_COLUMNS]

    c_matched = np.zeros(n_pred, dtype=bool)
    c_particle = np.full(n_pred, -1, dtype=np.int64)
    c_pur_n = np.zeros(n_pred)
    c_pur_e = np.zeros(n_pred)
    c_eff_n = np.zeros(n_pred)
    c_eff_e = np.zeros(n_pred)
    if match.n_matched:
        t, c = match.truth_index, match.pred_index
        c_matched[c] = True
        c_particle[c] = t
        c_pur_n[c] = _safe_div(overlap_n[t, c], pred_total_n[c])
        c_pur_e[c] = _safe_div(overlap_e[t, c], pred_total_e[c])
        c_eff_n[c] = _safe_div(overlap_n[t, c], truth_total_n[t])
        c_eff_e[c] = _safe_div(overlap_e[t, c], truth_total_e[t])

    # Cells belonging to no target: real deposits from particles below the pT cut, not noise.
    # They count against purity and never against efficiency, identically for both methods.
    unowned = clustered & ~owned
    e_unowned = np.bincount(pred_label[unowned], weights=cell_calib[unowned], minlength=n_pred)

    in_ecal = clustered & (record.subsystem <= 1)
    e_ecal = np.bincount(pred_label[in_ecal], weights=cell_calib[in_ecal], minlength=n_pred)

    clusters = pd.DataFrame({
        "sample_id": np.full(n_pred, record.sample_id, dtype=np.int64),
        "algo": algo,
        "cluster_row": np.arange(n_pred, dtype=np.int64),
        "n_hits": pred_total_n.astype(np.int32),
        "e_raw": pred_total_raw,
        "e_calib": pred_total_e,
        "matched": c_matched,
        "particle_row": c_particle,
        "pur_n": c_pur_n,
        "pur_e": c_pur_e,
        "eff_n": c_eff_n,
        "eff_e": c_eff_e,
        "n_owners": n_owners,
        "is_merge": n_owners > 1,
        "n_owners_e": n_owners_e,
        "is_merge_e": n_owners_e > 1,
        "e_unowned": e_unowned,
        "f_ecal": _safe_div(e_ecal, pred_total_e),
    })[CLUSTER_COLUMNS]

    event = {
        "sample_id": int(record.sample_id),
        "algo": algo,
        "n_hits": int(record.n_hits),
        "n_particles": n_truth,
        "n_particles_untruncated": int(record.n_particles_untruncated),
        "truncated": bool(record.truncated),
        "n_clusters": int(n_pred),
        "n_matched": match.n_matched,
        "n_unmatched_truth": int(match.unmatched_truth.size),
        "n_unmatched_pred": int(match.unmatched_pred.size),
        "e_calo_calib": float(record.event_energy_calib),
        "e_on_target_calib": float(record.event_energy_on_target_calib),
        "e_clustered_calib": float(cell_calib[clustered].sum()),
    }

    return particles, clusters, event


def pool(rows: Sequence[Mapping]) -> pd.DataFrame:
    """Concatenate per-event summary dictionaries into one table."""
    return pd.DataFrame(list(rows))
