"""The resolution reference, so the reported numbers have something to be measured against.

An efficiency of 0.55 is uninterpretable on its own. It could mean the method is poor, or it
could mean the task is nearly saturated and 0.6 is all anyone gets. A reader otherwise cannot
tell whether the gap to 1.0 is headroom or physics.

Note first why the obvious ceiling is not one. Feeding the truth partition back in as a
prediction scores exactly 1 by construction -- ``tests/test_scorer_identity.py`` asserts it --
so "perfect" is trivially achievable and says nothing. A meaningful ceiling has to come from
an *information* constraint: what could an algorithm achieve that does not get to read the
truth label off each cell?

An earlier version of this module also built a **geometric** ceiling -- perfect shower axes
and particle count, every cell assigned to the nearest axis. It was dropped from the analysis
because it is not a ceiling: it is optimistic in knowing the true count and axes, and it
bounds only spatial clustering as a class, which is CLUE's class and not MaskFormer's. The
thesis reports the resolution reference alone.

**Resolution** (:func:`resolution_labels`). Asks how many target particles are physically
inseparable to begin with. Particles whose showers share the same cells, carrying a large
fraction of each, are merged into one object; every cell is then
assigned perfectly within that merged set. Efficiency is ~1 by construction, so the number to
read is the **purity**, which is a genuine ceiling: no exclusive-partition algorithm can be
purer than the detector's own granularity allows. Unowned cells -- the sub-threshold deposits
that are 46% of the calorimeter energy -- are assigned to the nearest merged object rather
than dropped, because dropping them would hand this reference a contamination advantage that
neither real method enjoys and quietly inflate the ceiling.

It is scored through :func:`~src.evaluation.metrics.score_event` like any other algorithm, by
the same code path, so it lands in the same tables and plots on the same axes.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

#: Owners per cell considered when measuring which particles overlap. Cells are shared by a
#: long tail of tiny contributors; keeping the largest few bounds the pair enumeration at
#: negligible cost to the result, since a contributor outside the top few cannot hold the
#: `fraction` of itself that :func:`resolution_labels` requires.
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

    Distance from the interaction point is the depth coordinate, rather than a layer index.
    A layer means a different physical thickness in each subsystem -- 5.05 mm in ECAL against
    51 mm in HCAL -- so one step of layer index is not one step of depth, and a metric built
    on the index would be ten times more permissive in one subsystem than the other. Distance
    is the same quantity everywhere and needs no per-subsystem bookkeeping.

    That it is not "depth into the calorimeter" for an endcap cell does not matter here: the
    coordinate is only ever compared between cells at similar angles, where the difference in
    distance from the origin *is* the difference in depth along the shower axis.
    """
    radius = np.linalg.norm(np.stack([record.x, record.y, record.z], axis=1).astype(np.float64), axis=1)
    if radius.size == 0:
        return radius
    low, high = float(radius.min()), float(radius.max())
    return (radius - low) / max(high - low, 1e-12)


def _multiowner_entries(record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the multi-owner CSR into ``(particle, cell, calibrated energy)`` triples.

    The *multi-owner* truth is used rather than the exclusive partition on purpose. Overlap
    is the whole question here, and the exclusive partition has already thrown it away by
    handing each cell to a single winner.
    """
    counts = np.diff(record.truth_indptr)
    rows = np.repeat(np.arange(record.n_particles), counts)
    cols = record.truth_indices.astype(np.int64)
    calib = record.calibration[record.subsystem]
    energy = record.truth_incidence.astype(np.float64) * (record.energy[cols] * calib[cols])
    return rows, cols, energy


def particle_geometry(record) -> ParticleGeometry:
    """True shower axis and deposited energy per particle, from the multi-owner truth.

    The direction is the deposit-weighted mean of the cell unit vectors, renormalised. Doing
    it on vectors rather than by averaging eta and phi is what keeps a shower straddling
    ``phi = +/-pi`` from being handed a direction on the opposite side of the detector.
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
    # A particle with no recorded deposit gets an arbitrary but finite direction; it owns no
    # cell, so nothing is ever assigned to it and the choice cannot affect any metric.
    direction = np.where(norm > 1e-12, accum / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 1.0]))
    return ParticleGeometry(energy=total, direction=direction, depth=depth)


def unresolvable_groups(record, fraction: float = 0.5) -> np.ndarray:
    """Group truth particles that share too much of each other's energy to be separable.

    Two particles are joined when the energy they deposit *in the same cells* is at least
    `fraction` of the smaller one's total. The shared energy of a pair is taken as
    ``sum_i min(E_ia, E_ib)``, which is the part of the two showers that genuinely coincides
    rather than merely the cells they both touch.

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

    Owned cells go to their particle's group. Unowned cells -- real deposits from particles
    below the pT cut -- go to the nearest group axis, so this reference carries the same
    contamination burden the real methods do and its purity is a ceiling rather than a
    flattering artefact.
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
