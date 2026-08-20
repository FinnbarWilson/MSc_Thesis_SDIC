"""The resolution reference: a ceiling the reported efficiencies can be read against.

Feeding the truth partition back in as a prediction scores exactly 1 by construction, so a
meaningful ceiling has to come from an information constraint instead.
:func:`resolution_labels` merges target particles whose showers share too much of each other's
energy to be separable, then clusters the merged set perfectly. Efficiency is ~1 by
construction, so the number to read is the purity.

Unowned cells, the sub-threshold deposits, go to the nearest merged object rather than being
dropped, so this reference carries the same contamination burden the real methods do. It is
scored through :func:`~src.evaluation.metrics.score_event` like any other algorithm.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

#: Owners per cell considered when measuring overlap. Cells carry a long tail of tiny
#: contributors, and one outside the top few cannot hold the `fraction` of itself that
#: :func:`resolution_labels` requires, so this bounds the pair enumeration for free.
MAX_OWNERS_PER_CELL = 4


@dataclass(frozen=True)
class ParticleGeometry:
    """Per truth particle: total calibrated deposit, shower direction, and shower depth."""

    energy: np.ndarray
    direction: np.ndarray  # (n_particles, 3) unit vectors
    depth: np.ndarray  # (n_particles,) energy-weighted fractional depth in [0, 1]


def _cell_directions(record) -> np.ndarray:
    """Unit vector from the interaction point to each cell."""
    xyz = np.stack([record.x, record.y, record.z], axis=1).astype(np.float64)
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.maximum(norm, 1e-12)


def _cell_depths(record) -> np.ndarray:
    """Fractional depth of each cell, in [0, 1] across the event's radial extent.

    Distance from the interaction point rather than a layer index, because a layer is 5.05 mm
    in ECAL and 51 mm in HCAL and an index-based metric would be ten times more permissive in
    one than the other.
    """
    radius = np.linalg.norm(np.stack([record.x, record.y, record.z], axis=1).astype(np.float64), axis=1)
    if radius.size == 0:
        return radius
    low, high = float(radius.min()), float(radius.max())
    return (radius - low) / max(high - low, 1e-12)


def _multiowner_entries(record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the multi-owner CSR into ``(particle, cell, calibrated energy)`` triples.

    The multi-owner truth, not the exclusive partition: overlap is the question here, and the
    partition has already discarded it.
    """
    counts = np.diff(record.truth_indptr)
    rows = np.repeat(np.arange(record.n_particles), counts)
    cols = record.truth_indices.astype(np.int64)
    calib = record.calibration[record.subsystem]
    energy = record.truth_incidence.astype(np.float64) * (record.energy[cols] * calib[cols])
    return rows, cols, energy


def particle_geometry(record) -> ParticleGeometry:
    """True shower axis and deposited energy per particle, from the multi-owner truth.

    The direction is the deposit-weighted mean of the cell unit vectors, renormalised.
    Averaging vectors rather than eta and phi is what keeps a shower straddling ``phi = +/-pi``
    from being handed a direction on the opposite side of the detector.
    """
    n = int(record.n_particles)
    if n == 0:
        return ParticleGeometry(np.zeros(0), np.zeros((0, 3)), np.zeros(0))

    rows, cols, energy = _multiowner_entries(record)
    unit = _cell_directions(record)
    cell_depth = _cell_depths(record)

    total = np.bincount(rows, weights=energy, minlength=n)
    accum = np.zeros((n, 3), dtype=np.float64)
    for axis in range(3):
        accum[:, axis] = np.bincount(rows, weights=energy * unit[cols, axis], minlength=n)
    depth = np.bincount(rows, weights=energy * cell_depth[cols], minlength=n)
    depth = np.divide(depth, total, out=np.zeros(n), where=total > 0)

    norm = np.linalg.norm(accum, axis=1, keepdims=True)
    # A particle with no deposit gets an arbitrary finite direction; it owns no cell, so
    # nothing is assigned to it and the choice cannot affect any metric.
    direction = np.where(norm > 1e-12, accum / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 1.0]))
    return ParticleGeometry(energy=total, direction=direction, depth=depth)


def unresolvable_groups(record, fraction: float = 0.5) -> np.ndarray:
    """Group truth particles that share too much of each other's energy to be separable.

    Two particles are joined when the energy they deposit in the same cells, taken as
    ``sum_i min(E_ia, E_ib)``, is at least `fraction` of the smaller one's total.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        fraction: how much of the smaller particle must be shared before the pair is
            considered inseparable.

    Returns:
        Per particle, its group index. Particles in a group of one are separable.
    """
    n = int(record.n_particles)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    rows, cols, energy = _multiowner_entries(record)
    totals = np.bincount(rows, weights=energy, minlength=n)

    # Regroup particle-major entries into cell-major runs, largest depositor first.
    order = np.lexsort((-energy, cols))
    rows, cols, energy = rows[order], cols[order], energy[order]

    starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
    sizes = np.diff(np.r_[starts, cols.size])
    rank_within_cell = np.arange(cols.size) - np.repeat(starts, sizes)
    keep = rank_within_cell < MAX_OWNERS_PER_CELL
    rows, cols, energy = rows[keep], cols[keep], energy[keep]

    starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
    sizes = np.diff(np.r_[starts, cols.size])

    left, right, shared = [], [], []
    for start, size in zip(starts[sizes > 1], sizes[sizes > 1], strict=True):
        block = slice(start, start + size)
        owners, deposits = rows[block], energy[block]
        i, j = np.triu_indices(size, k=1)
        left.append(owners[i])
        right.append(owners[j])
        shared.append(np.minimum(deposits[i], deposits[j]))

    if not left:
        return np.arange(n, dtype=np.int64)

    left = np.concatenate(left)
    right = np.concatenate(right)
    shared = np.concatenate(shared)

    # One pair can coincide in many cells; sum over them before applying the threshold.
    pair_shared = coo_matrix((shared, (left, right)), shape=(n, n)).tocsr()
    pair_shared = pair_shared + pair_shared.T
    pair_shared.eliminate_zeros()

    pl, pr = pair_shared.nonzero()
    values = np.asarray(pair_shared[pl, pr]).ravel()
    smaller = np.minimum(totals[pl], totals[pr])
    joined = values >= fraction * np.maximum(smaller, 1e-30)
    if not joined.any():
        return np.arange(n, dtype=np.int64)

    graph = coo_matrix(
        (np.ones(joined.sum()), (pl[joined], pr[joined])), shape=(n, n)
    ).tocsr()
    _, groups = connected_components(graph, directed=False)
    return groups.astype(np.int64)


def resolution_labels(record, fraction: float = 0.5) -> tuple[np.ndarray, int]:
    """Perfect clustering of a truth set from which inseparable particles have been merged.

    Owned cells go to their particle's group; unowned cells go to the nearest group axis, so
    this reference carries the same contamination burden the real methods do.

    Returns:
        ``(label, n_groups)`` with -1 where the event has no particles.
    """
    groups = unresolvable_groups(record, fraction)
    n = int(record.n_particles)
    if n == 0 or record.n_hits == 0:
        return np.full(record.n_hits, -1, dtype=np.int32), 0

    used, compact = np.unique(groups, return_inverse=True)
    n_groups = int(used.size)

    label = np.full(record.n_hits, -1, dtype=np.int32)
    owned = record.truth_label >= 0
    label[owned] = compact[record.truth_label[owned]].astype(np.int32)

    if (~owned).any():
        geometry = particle_geometry(record)
        axes = np.zeros((n_groups, 3), dtype=np.float64)
        for axis in range(3):
            axes[:, axis] = np.bincount(
                compact, weights=geometry.energy * geometry.direction[:, axis], minlength=n_groups
            )
        norm = np.linalg.norm(axes, axis=1, keepdims=True)
        axes = np.where(norm > 1e-12, axes / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 1.0]))

        _, nearest = cKDTree(axes).query(_cell_directions(record)[~owned], k=1)
        label[~owned] = nearest.astype(np.int32)

    return label, n_groups
