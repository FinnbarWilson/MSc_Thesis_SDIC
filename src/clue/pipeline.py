"""The classical baseline: a three-stage CLUE pipeline over calorimeter cells.

Running CLUE once in three dimensions tends to merge showers that overlap in depth, so the
pipeline follows the layered strategy used for high-granularity calorimeter reconstruction:

  1. cluster within each detector layer, in 2D (:func:`cluster_subsystem`);
  2. reduce each layer cluster to a centroid and cluster the centroids in 3D, linking the
     per-layer pieces of one shower into a trackster (:func:`cluster_subsystem`);
  3. link clusters ACROSS sub-detectors (:func:`link_across_subsystems`), because stages 1-2
     run per sub-detector and a shower crossing ECAL into HCAL is otherwise split by
     construction. Its one parameter, `clue.link_radius`, is chosen by
     `scripts.scan_link_radius`.

Both passes call the same CLUEstering library, which is possible only because it accepts an
arbitrary number of dimensions. Nothing here modifies the algorithm: the parameters are
CLUE's own density radius, density threshold and linking radius, and the periodic distance
metric is a CLUEstering feature.

Input comes from the event store, which is what makes the comparison fair: these are
literally the cells the network was given, not a hit set re-derived from the raw files.

Coordinates are either projective ``(eta, phi)`` or Cartesian ``(x, y)`` in metres. The
angular choice suits the detector geometry, since a shower keeps an approximately constant
angular size as it propagates outward while its transverse extent grows with radius. It
requires phi to be treated as periodic so a shower straddling +/-pi is not split in two --
see :func:`_run_clue` for the part of that which is easy to get wrong.
"""

from collections.abc import Mapping, Sequence

import numpy as np

import CLUEstering as clue

TWO_PI = 2.0 * np.pi

#: Order matches the event store's ``cell_subsystem`` codes.
SUBSYSTEMS: tuple[str, ...] = ("ecb", "ece", "hcb", "hce")
SUBSYSTEM_CODE: Mapping[str, int] = {name: i for i, name in enumerate(SUBSYSTEMS)}

PARAMETER_NAMES = ("d_c_2d", "rho_c_2d", "d_o_2d", "d_c_3d", "rho_c_3d", "d_o_3d", "depth_scale")


def _run_clue(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    weights: np.ndarray,
    params: Mapping[str, float],
    suffix: str,
    coords: str,
    backend: str,
    depth: np.ndarray | None = None,
) -> np.ndarray:
    """Run one CLUE pass and return its cluster labels.

    Args:
        coord_a, coord_b: the two transverse clustering coordinates.
        weights: cell energies in GeV.
        params: holds ``d_c_<suffix>``, ``rho_c_<suffix>`` and ``d_o_<suffix>``.
        suffix: ``"2d"`` or ``"3d"``, selecting which parameters to use.
        coords: ``"etaphi"`` or ``"xy"``.
        backend: CLUEstering compute backend.
        depth: optional third coordinate, for the linking pass.

    Returns:
        Cluster id per input point, -1 for points called noise.
    """
    clusterer = clue.clusterer(params[f"d_c_{suffix}"], params[f"rho_c_{suffix}"], params[f"d_o_{suffix}"])

    rows = [coord_a, coord_b] if depth is None else [coord_a, coord_b, depth]
    data = np.array([*rows, weights])

    if coords == "etaphi":
        # BOTH of these are required, and that is the trap. `choose_metric` changes the
        # distance function; `wrapped_coords` wraps the tile grid CLUE uses to find
        # candidate neighbours. With the metric alone, points near +pi and -pi land in
        # non-adjacent tiles and are never compared, so the periodic distance never gets the
        # chance to fire: a blob straddling the boundary still comes back as two clusters.
        # Verified in tests/test_clue_periodic.py.
        #
        # read_data resets `self.wrapped` to all-zero every time it is called, so the flags
        # must be passed here rather than set beforehand.
        periods = [0.0, TWO_PI] if depth is None else [0.0, TWO_PI, 0.0]
        wrapped = [0, 1] if depth is None else [0, 1, 0]
        clusterer.read_data(data, wrapped_coords=wrapped)
        clusterer.choose_metric("periodic_euclidean", parameters=periods)
    else:
        clusterer.read_data(data)

    clusterer.run_clue(backend=backend)
    return clusterer.output_df["cluster_ids"].to_numpy()


def _layer_clusters(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    weights: np.ndarray,
    layer: np.ndarray,
    params: Mapping[str, float],
    coords: str,
    backend: str,
) -> np.ndarray:
    """Cluster each detector layer separately and return a label per cell.

    The layer index comes from the event store, where it was calibrated once against a
    frozen geometry. That matters more than it looks: a layer is a plane of constant \\|z\\|
    only in the endcaps. The barrels are 16-fold staved, so one layer spans a range of
    radii, and splitting on z there would carve the detector into hundreds of meaningless
    bands instead of its 48 (ECAL) or 36 (HCAL) real layers.

    Labels are made unique across layers by the caller, since CLUE restarts its numbering
    for every call.
    """
    labels = np.full(weights.size, -1, dtype=np.int64)
    order = np.argsort(layer, kind="stable")
    # Splitting on an integer layer index is exact; the old float-z comparison was correct
    # only by accident of the coordinate never being recomputed.
    boundaries = np.flatnonzero(np.diff(layer[order])) + 1
    for group in np.split(order, boundaries):
        if group.size:
            labels[group] = _run_clue(coord_a[group], coord_b[group], weights[group], params, "2d", coords, backend)
    return labels


def _layer_centroids(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    layer: np.ndarray,
    weights: np.ndarray,
    group: np.ndarray,
    depth_scale: float,
    coords: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce each layer cluster to one energy-weighted centroid.

    The third coordinate is the layer *index* divided by ``depth_scale``, not a physical
    depth. Using the index keeps the depth axis meaning the same thing in every subsystem,
    which a length cannot: ECAL layers are 5.05 mm apart and HCAL layers 51 mm, so a single
    linking radius expressed in metres would be ten times more permissive in one than the
    other.

    Azimuth is averaged circularly, because a plain mean over a cluster straddling +/-pi
    would land on the opposite side of the detector.
    """
    n_groups = int(group.max()) + 1
    total = np.bincount(group, weights=weights, minlength=n_groups)
    safe = np.maximum(total, np.finfo(np.float64).tiny)

    centre_a = np.bincount(group, weights=weights * coord_a, minlength=n_groups) / safe
    if coords == "etaphi":
        sin_b = np.bincount(group, weights=weights * np.sin(coord_b), minlength=n_groups)
        cos_b = np.bincount(group, weights=weights * np.cos(coord_b), minlength=n_groups)
        centre_b = np.arctan2(sin_b, cos_b)
    else:
        centre_b = np.bincount(group, weights=weights * coord_b, minlength=n_groups) / safe

    depth = np.zeros(n_groups)
    depth[group] = layer
    return centre_a, centre_b, depth / depth_scale, total


def cluster_subsystem(
    record,
    subsystem: str,
    params: Mapping[str, float],
    coords: str = "etaphi",
    backend: str = "cpu serial",
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster one calorimeter subsystem of one event.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        subsystem: one of :data:`SUBSYSTEMS`.
        params: the seven entries of :data:`PARAMETER_NAMES`.
        coords: ``"etaphi"`` or ``"xy"``.
        backend: CLUEstering compute backend.

    Returns:
        ``(cluster_ids, selected)``, both over all cells of the event. ``cluster_ids`` is -1
        for noise and for cells outside this subsystem; ``selected`` marks the cells this
        call considered.
    """
    cluster_ids = np.full(record.n_hits, -1, dtype=np.int64)
    # Selection by subsystem code. The previous version compared the detector column against
    # collection *names* like "ECalBarrelCollection", but the column holds uint8 detector
    # ids, so the comparison was always False and CLUE returned no clusters at all.
    selected = record.subsystem == SUBSYSTEM_CODE[subsystem]
    if not selected.any():
        return cluster_ids, selected

    if coords == "etaphi":
        coord_a, coord_b = record.eta()[selected], record.phi()[selected]
    else:
        coord_a, coord_b = record.x[selected].astype(np.float64), record.y[selected].astype(np.float64)

    weights = record.energy[selected].astype(np.float64)
    layer = record.layer[selected].astype(np.int64)

    labels = _layer_clusters(coord_a, coord_b, weights, layer, params, coords, backend)
    keep = labels >= 0
    if not keep.any():
        return cluster_ids, selected

    # One group per (layer, layer-local cluster) pair.
    _, group = np.unique(np.stack([layer[keep], labels[keep]]), axis=1, return_inverse=True)
    group = np.asarray(group).ravel()

    centre_a, centre_b, depth, total = _layer_centroids(
        coord_a[keep], coord_b[keep], layer[keep], weights[keep], group, params["depth_scale"], coords
    )
    trackster = _run_clue(centre_a, centre_b, total, params, "3d", coords, backend, depth=depth)

    positions = np.flatnonzero(selected)[keep]
    cluster_ids[positions] = trackster[group]
    return cluster_ids, selected


def link_across_subsystems(record, label: np.ndarray, radius: float) -> np.ndarray:
    """Union clusters in DIFFERENT subsystems whose energy centroids are within `radius`.

    WHY THIS EXISTS. `cluster_event` runs CLUE once per subsystem, so a shower crossing the
    ECAL/HCAL boundary is split by construction. Under the per-Geant-particle truth that cost
    little, because a target was a single shower fragment and 24.6% of them spanned a boundary.
    Under shower-level truth a target is the whole shower, and the figure is **42.2%** of
    targets and 10.8% of target energy (measured on 50 pu0 events). Leaving it would cap CLUE's
    efficiency for reasons belonging to the harness rather than the algorithm -- the same
    unfairness that collapsing the truth removed from the model's side.

    This is the pipeline's own 2D->3D trick applied one level up: reduce each cluster to a
    centroid, then link centroids. It joins only clusters from different subsystems, because
    clusters within one have already been offered to CLUE's own linking pass and rejecting
    that decision here would be second-guessing the algorithm rather than completing it.

    Centroids are energy-weighted in (eta, phi) with phi wrapped, since the boundary being
    crossed is radial (ECAL to HCAL) or in eta (barrel to endcap), and a shower crossing it
    keeps its angular position. Linking is transitive via union-find, so an ecb-hcb-hce chain
    becomes one cluster.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        label: per cell, cluster index, -1 for unclustered.
        radius: maximum centroid separation in (eta, phi). 0 disables and returns `label`.

    Returns:
        A relabelled copy; ids are NOT compacted, which `cluster_event` does afterwards.
    """
    if radius <= 0 or not (label >= 0).any():
        return label

    n = int(label.max()) + 1
    energy = record.energy_calib
    eta, phi = record.eta(), record.phi()

    clustered = label >= 0
    idx = label[clustered]
    weight = np.bincount(idx, weights=energy[clustered], minlength=n)
    if not (weight > 0).any():
        return label

    safe = np.where(weight > 0, weight, 1.0)
    cen_eta = np.bincount(idx, weights=energy[clustered] * eta[clustered], minlength=n) / safe
    # phi is periodic, so average the unit vector rather than the angle: a cluster straddling
    # +-pi would otherwise land its centroid at 0, on the opposite side of the detector.
    sin_p = np.bincount(idx, weights=energy[clustered] * np.sin(phi[clustered]), minlength=n) / safe
    cos_p = np.bincount(idx, weights=energy[clustered] * np.cos(phi[clustered]), minlength=n) / safe
    cen_phi = np.arctan2(sin_p, cos_p)

    # The subsystem a cluster belongs to. cluster_event builds each cluster from one subsystem,
    # so the first cell's code identifies it.
    sub = np.full(n, -1, dtype=np.int64)
    sub[idx] = record.subsystem[clustered]

    live = np.flatnonzero(weight > 0)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # n is a few hundred per event, so the pairwise pass is cheaper than building a tree.
    d_eta = cen_eta[live][:, None] - cen_eta[live][None, :]
    d_phi = np.mod(cen_phi[live][:, None] - cen_phi[live][None, :] + np.pi, TWO_PI) - np.pi
    close = np.hypot(d_eta, d_phi) <= radius
    cross = sub[live][:, None] != sub[live][None, :]
    rows, cols = np.nonzero(np.triu(close & cross, k=1))

    for a, b in zip(live[rows], live[cols], strict=True):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    out = label.copy()
    roots = np.array([find(i) for i in range(n)])
    out[clustered] = roots[idx]
    return out


def cluster_event(
    record,
    params_by_subsystem: Mapping[str, Mapping[str, float]],
    subsystems: Sequence[str] = SUBSYSTEMS,
    coords: str = "etaphi",
    backend: str = "cpu serial",
    min_cluster_hits: int = 1,
    link_radius: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Cluster every subsystem and merge the results into one labelling.

    Cluster ids are offset per subsystem so they stay unique across the event. On its own that
    is bookkeeping and does not link a shower across a subsystem boundary, so a hadron showering
    through ECAL into HCAL yields at least two clusters. `link_radius` is what closes that gap;
    at the default of 0 it is disabled and the behaviour is exactly what produced the pre-2026-08
    results. See :func:`link_across_subsystems` for why it is no longer optional under
    shower-level truth.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        params_by_subsystem: tuned parameters keyed by subsystem name.
        subsystems: which subsystems to process.
        coords: ``"etaphi"`` or ``"xy"``.
        backend: CLUEstering compute backend.
        min_cluster_hits: drop clusters smaller than this before relabelling.
        link_radius: (eta, phi) radius for joining clusters ACROSS subsystems. 0 disables.

    Returns:
        ``(labels, n_clusters)`` with labels compacted to ``0..n_clusters-1`` and -1 for
        unclustered cells.
    """
    merged = np.full(record.n_hits, -1, dtype=np.int64)
    offset = 0

    for subsystem in subsystems:
        if subsystem not in params_by_subsystem:
            continue
        ids, _ = cluster_subsystem(record, subsystem, params_by_subsystem[subsystem], coords=coords, backend=backend)
        clustered = ids >= 0
        if clustered.any():
            merged[clustered] = ids[clustered] + offset
            offset = int(merged.max()) + 1

    # Before min_cluster_hits, so the size cut sees the linked cluster rather than the pieces.
    merged = link_across_subsystems(record, merged, link_radius)

    if min_cluster_hits > 1 and (merged >= 0).any():
        counts = np.bincount(merged[merged >= 0])
        too_small = np.flatnonzero(counts < min_cluster_hits)
        if too_small.size:
            merged[np.isin(merged, too_small)] = -1

    # Compact so cluster ids are contiguous, which the scorer's overlap matrix assumes.
    if not (merged >= 0).any():
        return merged.astype(np.int32), 0
    used, compact = np.unique(merged[merged >= 0], return_inverse=True)
    out = np.full(record.n_hits, -1, dtype=np.int32)
    out[merged >= 0] = compact.astype(np.int32)
    return out, int(used.size)
