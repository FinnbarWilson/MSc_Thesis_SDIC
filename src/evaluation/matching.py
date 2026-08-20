"""Matching predicted clusters to truth particles.

Nothing here knows which method produced a clustering: it takes a label per cell and the
event's truth partition, and returns the correspondence the metrics are computed from. Both
CLUE and the MaskFormer are scored through it.

The assignment is global and one-to-one, solved with ``scipy.optimize.linear_sum_assignment``.
It is rectangular, so it yields ``min(n_truth, n_pred)`` pairs; truth particles left over are
inefficiencies and predicted clusters left over are fakes, and both are returned.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchResult:
    """A one-to-one assignment between truth particles and predicted clusters."""

    truth_index: np.ndarray
    pred_index: np.ndarray
    overlap: np.ndarray
    unmatched_truth: np.ndarray
    unmatched_pred: np.ndarray

    @property
    def n_matched(self) -> int:
        return int(self.truth_index.size)


def overlap_matrix(
    truth_label: np.ndarray,
    pred_label: np.ndarray,
    weight: np.ndarray | None,
    n_truth: int,
    n_pred: int,
) -> np.ndarray:
    """Weight shared between every (truth particle, predicted cluster) pair.

    Args:
        truth_label: per cell, the owning particle's index, -1 where no target owns it.
        pred_label: per cell, the predicted cluster index, -1 where unclustered.
        weight: per cell weight to accumulate. ``None`` counts cells, which gives the
            hit-based metrics; pass the particle's own calibrated deposit for the
            energy-weighted ones.
        n_truth: number of truth particles.
        n_pred: number of predicted clusters.

    Returns:
        ``(n_truth, n_pred)`` float64. Cells owned or claimed by nobody contribute to no pair,
        but still count towards the totals the caller divides by, which is how unclustered
        energy costs efficiency and sub-threshold deposits cost purity.
    """
    if n_truth == 0 or n_pred == 0:
        return np.zeros((n_truth, n_pred), dtype=np.float64)

    both = (truth_label >= 0) & (pred_label >= 0)
    if not both.any():
        return np.zeros((n_truth, n_pred), dtype=np.float64)

    flat = truth_label[both].astype(np.int64) * n_pred + pred_label[both].astype(np.int64)
    values = None if weight is None else np.asarray(weight, dtype=np.float64)[both]
    counts = np.bincount(flat, weights=values, minlength=n_truth * n_pred)
    return counts.reshape(n_truth, n_pred)


def hungarian_match(
    overlap: np.ndarray,
    min_overlap: float = 0.0,
    truth_total: np.ndarray | None = None,
    pred_total: np.ndarray | None = None,
    min_overlap_frac: float = 0.0,
) -> MatchResult:
    """Assign truth particles to predicted clusters, maximising total shared weight.

    Args:
        overlap: ``(n_truth, n_pred)`` from :func:`overlap_matrix`.
        min_overlap: pairs at or below this are reported unmatched. The default of 0 stops a
            global optimum pairing a particle with a wholly disjoint cluster to fill out the
            assignment.
        truth_total, pred_total: row and column totals, required only when
            `min_overlap_frac` is non-zero.
        min_overlap_frac: relative floor. A pair survives only if it shares at least this
            fraction of ``min(truth total, cluster total)``. Taking the minimum keeps the test
            symmetric: a small cluster wholly inside a large particle still matches, a large
            cluster brushing a small one does not.

    Returns:
        A :class:`MatchResult`.

    Raises:
        ValueError: if `min_overlap_frac` is non-zero and either total is missing.
    """
    n_truth, n_pred = overlap.shape
    if n_truth == 0 or n_pred == 0:
        return MatchResult(
            truth_index=np.empty(0, dtype=np.int64),
            pred_index=np.empty(0, dtype=np.int64),
            overlap=np.empty(0, dtype=np.float64),
            unmatched_truth=np.arange(n_truth),
            unmatched_pred=np.arange(n_pred),
        )

    rows, cols = linear_sum_assignment(overlap, maximize=True)
    values = overlap[rows, cols]
    keep = values > min_overlap

    if min_overlap_frac > 0.0:
        if truth_total is None or pred_total is None:
            msg = "min_overlap_frac requires both truth_total and pred_total"
            raise ValueError(msg)
        floor = min_overlap_frac * np.minimum(
            np.asarray(truth_total, dtype=np.float64)[rows],
            np.asarray(pred_total, dtype=np.float64)[cols],
        )
        keep &= values >= floor

    matched_truth = rows[keep]
    matched_pred = cols[keep]
    return MatchResult(
        truth_index=matched_truth,
        pred_index=matched_pred,
        overlap=values[keep],
        unmatched_truth=np.setdiff1d(np.arange(n_truth), matched_truth, assume_unique=False),
        unmatched_pred=np.setdiff1d(np.arange(n_pred), matched_pred, assume_unique=False),
    )


def fragmentation(overlap: np.ndarray, truth_total: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Per truth particle, how many clusters hold at least `fraction` of it; >1 means split.

    Counted from the same overlap matrix that drove the matching, so the split rate cannot
    drift from the efficiency. The weighting is the caller's: pass the hit-counted pair or the
    energy pair. Both are computed in :func:`~src.evaluation.metrics.score_event`.
    """
    if overlap.size == 0:
        return np.zeros(overlap.shape[0], dtype=np.int32)
    threshold = fraction * np.asarray(truth_total, dtype=np.float64)[:, None]
    return (overlap >= np.maximum(threshold, np.finfo(np.float64).tiny)).sum(axis=1).astype(np.int32)


def contamination(overlap: np.ndarray, truth_total: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Per cluster, how many truth particles put at least `fraction` of themselves in it.

    More than one means the cluster merged several particles. The threshold is a fraction of
    each truth particle's total, not of the cluster's, so a cluster swallowing a large particle
    and a tenth of a small neighbour has merged them however its own energy divides.
    """
    if overlap.size == 0:
        return np.zeros(overlap.shape[1], dtype=np.int32)
    threshold = fraction * np.asarray(truth_total, dtype=np.float64)[:, None]
    return (overlap >= np.maximum(threshold, np.finfo(np.float64).tiny)).sum(axis=0).astype(np.int32)
