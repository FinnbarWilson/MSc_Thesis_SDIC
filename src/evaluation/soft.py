"""Scoring against the multi-owner truth, with fractional cell ownership on both sides.

The head-to-head in :mod:`src.evaluation.metrics` runs on an exclusive partition: one owner
per cell in the truth, one cluster per cell in the prediction. That is the right call for
comparing a partitioning algorithm against a mask-based one *on the task they share*, and it
is what makes the numbers in the main figures fair. But it is not the whole question, and it
throws away something real on both sides of the comparison:

*   The truth genuinely is multi-owner. A cell struck by three particles holds three
    contributions, and the exclusive partition hands it wholly to the largest. That discards
    ~17% of (particle, cell) associations, and about 0.4% of target particles own no cell
    exclusively at all -- they are structurally unrecoverable by any partitioning method,
    and the main tables can only ever report them as missed.
*   MaskFormer's masks may overlap; it can say a cell belongs to two particles.
    :meth:`~src.io.event_store.EventRecord.maskformer_labels` deliberately collapses that to
    one winner so CLUE has something it can compete against. Which means the head-to-head
    measures the model with its distinguishing capability switched off.

So this module scores the same events with ownership left fractional. **The aim is not to even
the fight.** It is that a metric which suppresses one method's distinguishing capability is
measuring the format rather than the clustering. What makes the result fair is not that both
methods score well but that both are measured against the same, method-independent target:

*   Prediction weights are normalised per cell. An algorithm whose clusters never overlap has
    every weight equal to 1, so CLUE arrives here unchanged and is scored by identical code
    with no special case.
*   The overlap of a particle and a cluster in a cell is ``E_i * min(t_ai, w_ci)`` -- the
    energy both sides agree belongs there. ``min`` rather than a product because the two are
    shares of one divisible quantity, not independent probabilities.
*   The denominator is the particle's **actual deposited energy**, a property of the event
    that neither method's output can move. The exclusive metric instead measures against the
    energy in cells the particle dominates, which is a target defined by what a partition
    happens to be able to express.

That last point has a consequence worth stating plainly, because it looks like unfairness and
is not. Score the truth partition as a prediction and its soft efficiency is *exactly*
``exclusive_share`` -- the fraction of the particle's energy in cells it dominates -- not 1.
So ``exclusive_share`` is a hard ceiling on soft efficiency for the entire partitioning class,
and CLUE's soft efficiency will sit **below** its exclusive efficiency. The shortfall is not a
penalty applied to CLUE; it is the measurement of the energy that only overlapping masks can
reach, which is the quantity this module exists to report.

The size is bounded and worth knowing in advance: the exclusive partition already retains
~94% of each particle's own energy, so the aggregate effect is a few per cent. The number to
watch is the tail -- what fraction of the particles that *no* partitioning method can recover
does a mask-based one actually find.

One honest caveat on the weights, and it turns out to be a finding rather than a caveat. A mask
probability is a *membership* output; the model was never trained to predict what fraction of a
cell's energy is a given particle's. So a low soft score could mean "cannot represent sharing"
or "divides sharing badly", and :func:`sharing_diagnostics` exists to separate them.

Measured, it is the second, and it is not fixable by tuning. The model divides each cell 2.08
ways against the truth's 1.22, and ``scripts/scan_soft_threshold.py`` shows that no mask
threshold closes the gap: at 0.95, where masks are already shedding genuine cells, it still
divides 1.85 ways. The overlap is made of confident claims, not marginal ones. That follows
from the architecture -- an element-wise sigmoid per (query, cell) rather than a softmax over
queries, so nothing in the loss constrains one cell's claims to sum to anything. The model is
never taught what a cell's division should add up to.

The practical consequence: quote soft efficiency with the sharing diagnostics beside it, and
read the shortfall as a statement about mask calibration rather than about whether the
architecture can represent shared cells. It plainly can -- that is what the recovery of
otherwise-unreachable particles demonstrates.
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
    """Present an exclusive labelling in the same fractional form as an overlapping one.

    Every claim has weight 1, which is what a partition means: the cell is wholly one
    cluster's. Passing CLUE through this rather than through a separate code path is what
    guarantees the two methods are scored identically here.
    """
    cells = np.flatnonzero(np.asarray(pred_label) >= 0)
    return np.asarray(pred_label)[cells].astype(np.int64), cells.astype(np.int64), np.ones(cells.size)


def _cell_join(
    left_cell: np.ndarray, right_cell: np.ndarray, n_hits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of entries that fall in the same cell, for two cell-sorted arrays.

    Both sides have only a handful of entries per cell -- a few truth contributors, a few
    accepting queries -- so the join is small. It is built with repeats rather than a Python
    loop over cells because there are tens of thousands of them per event.
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

    # The exclusive share is the diagnostic that makes this table worth keeping: it says how
    # much of each particle a partitioning method could even in principle have reached, and
    # `no_exclusive_cell` flags the particles for which that is nothing at all.
    exclusive = np.zeros(n_truth)
    owned = record.truth_label >= 0
    if owned.any():
        # Calibrated, like every other energy here. `truth_deposit` is raw, and the two
        # differ by the ~40x sampling factor, so mixing them silently scales the share by
        # 1/40 and makes it look as though a partition reaches almost none of its particle.
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


def sharing_diagnostics(record, pred_cell: np.ndarray) -> dict:
    """How often each side says a cell is shared, which is what makes the soft numbers legible.

    Without this the soft efficiency looks like a bug. A method whose masks overlap twice as
    often as the truth does has its per-cell weights halved by the normalisation, so the
    energy it can be credited with in a contested cell is capped well below the particle's
    actual share and its soft efficiency lands *below* its exclusive efficiency.

    That is the metric working, not failing -- over-claiming a cell is a real error and should
    cost something. But it does mean the soft score folds together two different questions:
    whether a method can represent a shared cell at all, and whether it divides it in the
    right proportions. These two numbers separate them, and they are worth reporting next to
    any soft efficiency. Note that mask probability is a *membership* output, not a trained
    estimate of energy fraction, so some of the miscalibration is a property of what the model
    was asked to predict rather than of how well it predicts it.
    """
    claims = np.bincount(np.asarray(pred_cell), minlength=record.n_hits)
    owners = np.bincount(record.truth_indices, minlength=record.n_hits)
    return {
        "claims_per_cell": float(claims[claims > 0].mean()) if (claims > 0).any() else 0.0,
        "shared_cell_frac": float((claims[claims > 0] > 1).mean()) if (claims > 0).any() else 0.0,
        "truth_owners_per_cell": float(owners[owners > 0].mean()) if (owners > 0).any() else 0.0,
        "truth_shared_frac": float((owners[owners > 0] > 1).mean()) if (owners > 0).any() else 0.0,
    }


def capability_summary(
    tables: Sequence[pd.DataFrame],
    working_point: float = 0.5,
    diagnostics: dict[str, list[dict]] | None = None,
) -> pd.DataFrame:
    """Collapse the soft tables to the comparison this study exists to make.

    `impossible_recovered` is the headline. It is the share of structurally unrecoverable
    particles -- those owning no cell exclusively -- that a method nonetheless reconstructs
    above the working point. A partitioning method has no cell of its own to award them, so
    its score there is near zero as a property of the algorithm class rather than of its
    tuning: measured at 0.007 for CLUE against 0.534 for overlapping masks. The residual is
    not quite zero because a cluster built around some *other* particle can contain cells this
    one contributed to sub-dominantly, and that energy is credited.
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
