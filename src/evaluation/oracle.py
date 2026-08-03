"""Reference clusterings, so the reported numbers have something to be measured against.

An efficiency of 0.31 is uninterpretable on its own. It could mean the method is poor, or it
could mean the task is nearly saturated and 0.35 is all anyone gets. Nothing in the figures
distinguished those two readings, which is the single largest gap in the analysis: a reader
cannot tell whether the gap to 1.0 is headroom or physics.

Note first why the obvious ceiling is not one. Feeding the truth partition back in as a
prediction scores exactly 1 by construction -- ``tests/test_scorer_identity.py`` asserts it --
so "perfect" is trivially achievable and says nothing. A meaningful ceiling has to come from
an *information* constraint: what could an algorithm achieve that does not get to read the
truth label off each cell? Two are built here, bracketing the answer from different sides.

**Geometric** (:func:`geometric_labels`). An idealised algorithm handed the two things every
clustering method must otherwise infer -- the true number of particles and the true shower
axis of each -- which then assigns every cell to the nearest axis in angle and depth. Its one
free parameter, the relative weight of depth, is *scanned and maximised* rather than guessed:
a ceiling left at an arbitrary setting is not a ceiling but one arbitrary algorithm.

This is the ceiling for **spatial clustering as a class**, which is the class CLUE belongs to
and MaskFormer does not. That gives it a use no aggregate number has: if a learned model
exceeds the best achievable by perfect-seeded geometry, it is demonstrably using something
beyond geometry, and if it does not, the gap says how much is left on the table before
architecture is the limiting factor.

Read it as a bound on that class and not as a universal one -- it remains *optimistic* in
knowing the true particle count and axes. An earlier version used angle alone, which was an
unfair reference for a depth-aware method like CLUE; depth is worth more than expected here,
lifting the ceiling from 0.530 to 0.591 mean efficiency.

**Resolution** (:func:`resolution_labels`). Comes at it from the other side and asks how many
target particles are physically inseparable to begin with. Particles whose showers share the
same cells, carrying a large fraction of each, are merged into one object; every cell is then
assigned perfectly within that merged set. Efficiency is ~1 by construction, so the number to
read is the **purity**, which is a genuine ceiling: no exclusive-partition algorithm can be
purer than the detector's own granularity allows. Unowned cells -- the sub-threshold deposits
that are 46% of the calorimeter energy -- are assigned to the nearest merged object rather
than dropped, because dropping them would hand this reference a contamination advantage that
neither real method enjoys and quietly inflate the ceiling.

Both are scored through :func:`~src.evaluation.metrics.score_event` like any other algorithm,
by the same code path, so they land in the same tables and plot on the same axes.
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


#: Depth weights scanned when building the geometric ceiling. A ceiling is allowed -- in fact
#: obliged -- to take the best value of its own free parameter: leaving it at an arbitrary
#: setting would understate the bound and flatter everything measured against it.
#:
#: The range brackets the measured optimum of 0.8 on both sides, which matters here for the
#: same reason it does in CLUE's tuning: a first pass over [0, 0.8] returned the optimum
#: pressed against its upper bound, which means the box was wrong rather than that the answer
#: was 0.8. Extending upward found a genuine turnover (0.591 at 0.8, 0.570 at 1.6, 0.403 at
#: 25.6), so the optimum is now interior and the ceiling is not an artefact of the range.
DEPTH_WEIGHT_SCAN = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)


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


def _assignment_space(directions: np.ndarray, depths: np.ndarray, depth_weight: float) -> np.ndarray:
    """Stack angle and depth into one space a Euclidean nearest-neighbour query can search.

    Chordal distance on the unit sphere is monotonic in opening angle, so the first three
    columns make a Euclidean query an angular one with no phi wraparound to special-case. The
    fourth adds depth at a chosen weight; at ``depth_weight = 0`` the result is the pure
    angular assignment.

    Depth turns out to be worth roughly as much as angle: the measured optimum is 0.8 against
    a unit-vector angular term, and it lifts the ceiling from 0.530 to 0.591 mean efficiency.
    That is a larger contribution than one might expect, and it is the reason the angle-only
    version was an unfair reference for CLUE, which uses depth.

    It does not keep rising, though, and the reason is worth knowing. A shower is a ray from
    the interaction point, not a point, so beyond about 1.0 the depth term starts cutting
    showers across their own radial extent rather than separating neighbours, and the ceiling
    falls away steadily -- 0.570 at 1.6, down to 0.403 at 25.6.
    """
    if depth_weight == 0.0:
        return directions
    return np.column_stack([directions, depth_weight * depths])


def geometric_labels(
    record,
    geometry: ParticleGeometry | None = None,
    depth_weight: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Assign every cell to the nearest true shower axis in angle and depth.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        geometry: precomputed axes, to avoid recomputing them across a weight scan.
        depth_weight: relative weight of the depth coordinate. Use
            :func:`best_depth_weight` to choose it rather than guessing.

    Returns:
        ``(label, n_clusters)`` in the same form every other method returns, so the result
        goes straight into :func:`~src.evaluation.metrics.score_event`.
    """
    geometry = geometry or particle_geometry(record)
    n = int(record.n_particles)
    if n == 0 or record.n_hits == 0:
        return np.full(record.n_hits, -1, dtype=np.int32), 0

    tree = cKDTree(_assignment_space(geometry.direction, geometry.depth, depth_weight))
    _, nearest = tree.query(
        _assignment_space(_cell_directions(record), _cell_depths(record), depth_weight), k=1
    )
    return nearest.astype(np.int32), n


def best_depth_weight(
    records, weights=DEPTH_WEIGHT_SCAN, working_point: float = 0.5
) -> tuple[float, dict[float, float]]:
    """Pick the depth weight that maximises the ceiling, over a small scan.

    A ceiling that has not been maximised over its own free parameters is not a ceiling, it is
    one arbitrary algorithm; scanning is what earns the name. The criterion is the mean energy
    efficiency, which is continuous and so does not depend on where a working point is put.

    Args:
        records: a sequence of event records, normally a subset -- the optimum is stable long
            before the full evaluation window.
        weights: depth weights to try.
        working_point: unused by the criterion, accepted so callers can pass it uniformly.

    Returns:
        ``(best_weight, {weight: mean efficiency})``.
    """
    from src.evaluation.metrics import score_event

    scores: dict[float, float] = {}
    for weight in weights:
        total = 0.0
        count = 0
        for record in records:
            geometry = particle_geometry(record)
            label, n = geometric_labels(record, geometry, depth_weight=weight)
            particles, _, _ = score_event(record, label, n, algo="geometric")
            total += float(particles["eff_e"].sum())
            count += len(particles)
        scores[weight] = total / max(count, 1)
    return max(scores, key=lambda w: scores[w]), scores


#: Kept so existing callers and tables that name the angle-only version still resolve.
seeded_labels = geometric_labels


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
