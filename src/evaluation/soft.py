"""Scoring with fractional cell ownership on both the truth and the prediction side.

The head-to-head in :mod:`src.evaluation.metrics` runs on an exclusive partition, which is the
right comparison for a partitioning algorithm against a mask-based one but discards real
information on both sides. This module measures what that costs.

Three properties keep the result method-independent. Prediction weights are normalised per cell,
so a method whose clusters never overlap arrives with every weight equal to 1. A particle and a
cluster overlap in a cell by ``E_i * min(t_ai, w_ci)``, taking the minimum rather than a product
because the two are shares of one divisible quantity. And the denominator is the particle's
actual deposited energy, which neither method's output can move.

Expect the truth partition itself to score ``exclusive_share`` here rather than 1, so
``exclusive_share`` is a ceiling on soft efficiency for the whole partitioning class. Report soft
efficiency with :func:`sharing_diagnostics` beside it: a mask probability is a membership output,
not a trained estimate of a cell's energy fraction.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.matching import hungarian_match

SOFT_PARTICLE_COLUMNS = [
    "sample_id", "algo", "particle_row", "particle_id", "pdg_id",
    "p_energy", "p_pt", "p_eta", "p_phi",
    "e_dep_multi", "e_dep_exclusive", "exclusive_share", "no_exclusive_cell",
    "dr_min", "n_within_02",
    "matched", "cluster_row", "eff_e", "pur_e", "n_shared_cells",
]


def hard_weights(pred_label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Present an exclusive labelling in the fractional form, every claim carrying weight 1.

    A partitioning method goes through this rather than a separate code path, so both methods
    reach the scorer identically.
    """
    cells = np.flatnonzero(np.asarray(pred_label) >= 0)
    return np.asarray(pred_label)[cells].astype(np.int64), cells.astype(np.int64), np.ones(cells.size)


def _cell_join(
    left_cell: np.ndarray, right_cell: np.ndarray, n_hits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of entries falling in the same cell, for two cell-sorted arrays.

    Built with repeats rather than a loop over cells, of which there are tens of thousands per
    event.
    """
    right_per_cell = np.bincount(right_cell, minlength=n_hits)
    repeats = right_per_cell[left_cell]
    if repeats.sum() == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    left_index = np.repeat(np.arange(left_cell.size), repeats)
    starts = np.concatenate([[0], np.cumsum(right_per_cell)[:-1]])
    offset = np.arange(int(repeats.sum())) - np.repeat(np.cumsum(repeats) - repeats, repeats)
    right_index = starts[left_cell[left_index]] + offset
    return left_index, right_index


def score_event_soft(
    record,
    pred_cluster: np.ndarray,
    pred_cell: np.ndarray,
    pred_weight: np.ndarray,
    n_pred: int,
    algo: str,
    min_overlap_frac: float = 0.05,
) -> pd.DataFrame:
    """Score one event with fractional ownership on both the truth and the prediction side.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        pred_cluster, pred_cell, pred_weight: the prediction as fractional claims. Use
            :func:`hard_weights` for a partitioning method and
            :meth:`~src.io.event_store.EventRecord.maskformer_soft_masks` for overlapping
            masks.
        n_pred: number of predicted clusters.
        algo: label carried into the output table.
        min_overlap_frac: relative match floor, as in the exclusive scoring.

    Returns:
        One row per truth particle.
    """
    n_truth = int(record.n_particles)
    calib = record.calibration[record.subsystem]
    cell_energy = record.energy * calib

    # Truth as fractional claims, straight from the multi-owner CSR.
    truth_particle = np.repeat(np.arange(n_truth), np.diff(record.truth_indptr)).astype(np.int64)
    truth_cell = record.truth_indices.astype(np.int64)
    truth_share = record.truth_incidence.astype(np.float64)

    order = np.argsort(truth_cell, kind="stable")
    truth_particle, truth_cell, truth_share = truth_particle[order], truth_cell[order], truth_share[order]
    order = np.argsort(pred_cell, kind="stable")
    pred_cluster, pred_cell, pred_weight = (
        np.asarray(pred_cluster)[order], np.asarray(pred_cell)[order], np.asarray(pred_weight)[order],
    )

    truth_total = np.bincount(truth_particle, weights=truth_share * cell_energy[truth_cell], minlength=n_truth)
    pred_total = np.bincount(pred_cluster, weights=pred_weight * cell_energy[pred_cell], minlength=n_pred)

    overlap = np.zeros((n_truth, n_pred), dtype=np.float64)
    shared_cells = np.zeros(n_truth, dtype=np.int64)
    if n_truth and n_pred and truth_cell.size and pred_cell.size:
        left, right = _cell_join(truth_cell, pred_cell, record.n_hits)
        if left.size:
            # The energy both sides agree belongs to this (particle, cluster) in this cell.
            # min rather than a product: the shares are portions of one divisible quantity,
            # not independent probabilities, so their intersection is the smaller share.
            contribution = cell_energy[truth_cell[left]] * np.minimum(
                truth_share[left], pred_weight[right]
            )
            flat = truth_particle[left] * n_pred + pred_cluster[right]
            overlap = np.bincount(flat, weights=contribution, minlength=n_truth * n_pred).reshape(n_truth, n_pred)
            shared_cells = np.bincount(truth_particle[left], minlength=n_truth).astype(np.int64)

    match = hungarian_match(
        overlap, truth_total=truth_total, pred_total=pred_total, min_overlap_frac=min_overlap_frac
    )

    matched = np.zeros(n_truth, dtype=bool)
    cluster_row = np.full(n_truth, -1, dtype=np.int64)
    eff = np.zeros(n_truth)
    pur = np.zeros(n_truth)
    if match.n_matched:
        t, c = match.truth_index, match.pred_index
        matched[t] = True
        cluster_row[t] = c
        eff[t] = np.divide(overlap[t, c], truth_total[t], out=np.zeros(t.size), where=truth_total[t] > 0)
        pur[t] = np.divide(overlap[t, c], pred_total[c], out=np.zeros(c.size), where=pred_total[c] > 0)

    # The exclusive share says how much of each particle a partitioning method could reach at
    # all; `no_exclusive_cell` flags those for which that is nothing.
    exclusive = np.zeros(n_truth)
    owned = record.truth_label >= 0
    if owned.any():
        # Calibrated, like every other energy here. `truth_deposit` is raw, and the two differ
        # by the ~40x sampling factor, and mixing them scales the share by 1/40.
        deposit = record.truth_deposit * calib
        exclusive = np.bincount(record.truth_label[owned], weights=deposit[owned], minlength=n_truth)

    from src.evaluation.metrics import local_density

    dr_min, n_within = local_density(record.particle_eta, record.particle_phi)

    return pd.DataFrame({
        "sample_id": np.full(n_truth, record.sample_id, dtype=np.int64),
        "algo": algo,
        "particle_row": np.arange(n_truth, dtype=np.int64),
        "particle_id": record.particle_id,
        "pdg_id": record.particle_pdg_id,
        "p_energy": record.particle_energy,
        "p_pt": record.particle_pt,
        "p_eta": record.particle_eta,
        "p_phi": record.particle_phi,
        "e_dep_multi": truth_total,
        "e_dep_exclusive": exclusive,
        "exclusive_share": np.divide(exclusive, truth_total, out=np.zeros(n_truth), where=truth_total > 0),
        "no_exclusive_cell": exclusive <= 0.0,
        "dr_min": dr_min,
        "n_within_02": n_within,
        "matched": matched,
        "cluster_row": cluster_row,
        "eff_e": eff,
        "pur_e": pur,
        "n_shared_cells": shared_cells,
    })[SOFT_PARTICLE_COLUMNS]


def _effective_claims(cell: np.ndarray, weight: np.ndarray, n_hits: int) -> float:
    """Mean perplexity of each cell's division: how many ways it is effectively split.

    ``exp(-sum p log p)`` over a cell's normalised weights. A raw count of claims depends on how
    many claims a method emits rather than on how it divides a cell; perplexity does not, so it
    is the figure to use when comparing methods that emit different numbers of claims. Compare
    against the truth value computed the same way, not against 1.
    """
    cell = np.asarray(cell)
    weight = np.asarray(weight, dtype=np.float64)
    if cell.size == 0:
        return 0.0
    total = np.bincount(cell, weights=weight, minlength=n_hits)
    share = weight / np.maximum(total[cell], 1e-30)
    entropy = -np.bincount(cell, weights=share * np.log(np.maximum(share, 1e-30)), minlength=n_hits)
    touched = np.bincount(cell, minlength=n_hits) > 0
    return float(np.exp(entropy[touched]).mean()) if touched.any() else 0.0


def sharing_diagnostics(record, pred_cell: np.ndarray, pred_weight: np.ndarray | None = None) -> dict:
    """How often each side says a cell is shared, in raw and perplexity-weighted form.

    The soft score folds together two questions: whether a method can represent a shared cell,
    and whether it divides one in the right proportions. These numbers separate them. A method
    over-claiming cells has its per-cell weights cut by the normalisation, so its soft efficiency
    can land below its exclusive efficiency.

    Args:
        pred_cell: cell index of each predicted claim.
        pred_weight: claim weights; when given, the perplexity-based pair is added.
    """
    claims = np.bincount(np.asarray(pred_cell), minlength=record.n_hits)
    owners = np.bincount(record.truth_indices, minlength=record.n_hits)
    out = {
        "claims_per_cell": float(claims[claims > 0].mean()) if (claims > 0).any() else 0.0,
        "shared_cell_frac": float((claims[claims > 0] > 1).mean()) if (claims > 0).any() else 0.0,
        "truth_owners_per_cell": float(owners[owners > 0].mean()) if (owners > 0).any() else 0.0,
        "truth_shared_frac": float((owners[owners > 0] > 1).mean()) if (owners > 0).any() else 0.0,
    }

    # The k-independent pair, for comparing methods that emit different numbers of claims.
    if pred_weight is not None:
        out["eff_claims_per_cell"] = _effective_claims(pred_cell, pred_weight, record.n_hits)
        out["truth_eff_owners_per_cell"] = _effective_claims(
            record.truth_indices, record.truth_incidence, record.n_hits
        )
    return out


def capability_summary(
    tables: Sequence[pd.DataFrame],
    working_point: float = 0.5,
    diagnostics: dict[str, list[dict]] | None = None,
) -> pd.DataFrame:
    """Collapse the soft tables to one row per method.

    ``impossible_recovered`` is the share of particles owning no cell exclusively that a method
    nonetheless reconstructs above `working_point`. A partitioning method has no cell of its own
    to award them, so its score there is near zero as a property of the algorithm class.

    Args:
        tables: per-event outputs of :func:`score_event_soft`.
        working_point: containment a particle must reach to count.
        diagnostics: per-method lists of :func:`sharing_diagnostics` outputs, averaged in.
    """
    rows = []
    for table in tables:
        for algo, group in table.groupby("algo", observed=True):
            impossible = group[group["no_exclusive_cell"]]
            shared = pd.DataFrame((diagnostics or {}).get(str(algo), [])).mean(numeric_only=True)
            rows.append({
                "algo": str(algo),
                "n_particles": len(group),
                "claims_per_cell": float(shared.get("claims_per_cell", float("nan"))),
                "truth_owners_per_cell": float(shared.get("truth_owners_per_cell", float("nan"))),
                "eff_claims_per_cell": float(shared.get("eff_claims_per_cell", float("nan"))),
                "truth_eff_owners_per_cell": float(shared.get("truth_eff_owners_per_cell", float("nan"))),
                "eff_soft": float((group["eff_e"] >= working_point).mean()),
                "eff_soft_mean": float(group["eff_e"].mean()),
                "pur_soft": float((group["pur_e"] >= working_point).mean()),
                "pur_soft_mean": float(group["pur_e"].mean()),
                "match_rate": float(group["matched"].mean()),
                "n_impossible": int(len(impossible)),
                "impossible_recovered": float((impossible["eff_e"] >= working_point).mean()) if len(impossible) else 0.0,
                "impossible_matched": float(impossible["matched"].mean()) if len(impossible) else 0.0,
            })
    return pd.DataFrame(rows)
