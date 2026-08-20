"""The classical baseline: a three-stage CLUE pipeline over calorimeter cells.

Running CLUE once in three dimensions merges showers that overlap in depth, so the pipeline
follows the layered strategy used for high-granularity calorimeter reconstruction:

  1. cluster within each detector layer, in 2D;
  2. reduce each layer cluster to a centroid and cluster the centroids in 3D, linking one
     shower's per-layer pieces into a trackster (both stages :func:`cluster_subsystem`);
  3. link clusters across sub-detectors (:func:`link_across_subsystems`), since stages 1-2 run
     per sub-detector and a shower crossing ECAL into HCAL is otherwise split.

Nothing here modifies the algorithm: the parameters are CLUE's own, and the periodic metric is a
CLUEstering feature. Input comes from the event store, so these are the cells the network saw.

Coordinates are projective ``(eta, phi)`` or Cartesian ``(x, y)`` in metres. The angular choice
suits the geometry, a shower keeping roughly constant angular size as it propagates outward, and
requires phi to be treated as periodic.
"""

from collections.abc import Mapping, Sequence

import CLUEstering as clue
import numpy as np

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
        depth: optional third coordinate, for the 3D pass.

    Returns:
        Cluster id per input point, -1 for points called noise.
    """
    clusterer = clue.clusterer(params[f"d_c_{suffix}"], params[f"rho_c_{suffix}"], params[f"d_o_{suffix}"])

    rows = [coord_a, coord_b] if depth is None else [coord_a, coord_b, depth]
    data = np.array([*rows, weights])

    if coords == "etaphi":
        # Both calls are required. `choose_metric` changes the distance function, while
        # `wrapped_coords` wraps the tile grid CLUE uses to find candidate neighbours. Without
        # the second, points near +pi and -pi land in non-adjacent tiles and are never compared,
        # so a blob straddling the boundary still comes back as two clusters. The failure is
        # silent; tests/test_clue_periodic.py pins both halves.
        #
        # read_data resets `self.wrapped`, so the flags go to it rather than being set first.
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

    The layer index comes from the event store, where it was calibrated against a frozen
    geometry: a layer is a plane of constant \\|z\\| only in the endcaps, the barrels being
    16-fold staved. Labels are made unique across layers by the caller, since CLUE restarts its
    numbering on every call.
    """
    labels = np.full(weights.size, -1, dtype=np.int64)
    order = np.argsort(layer, kind="stable")
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

    The third coordinate is the layer index divided by ``depth_scale``, not a physical depth,
    so the depth axis means the same thing in every subsystem; ECAL layers are 5.05 mm apart
    and HCAL layers 51 mm, and a radius in metres would be ten times more permissive in one.

    Azimuth is averaged circularly, since a plain mean over a cluster straddling +/-pi would
    land on the opposite side of the detector.
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
    """Union clusters in *different* subsystems whose energy centroids are within `radius`.

    :func:`cluster_event` runs CLUE once per subsystem, so a shower crossing the ECAL/HCAL
    boundary is split by construction; under shower-level truth a large share of targets span
    one, and without this stage CLUE could not represent them at all. Clusters within one
    subsystem are left alone, having already been offered to CLUE's own 3D pass.

    Centroids are energy-weighted in (eta, phi) with phi wrapped, and linking is transitive via
    union-find, so an ecb-hcb-hce chain becomes one cluster.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        label: per cell, cluster index, -1 for unclustered.
        radius: maximum centroid separation in (eta, phi). 0 disables and returns `label`.

    Returns:
        A relabelled copy. Ids are not compacted; `cluster_event` does that afterwards.
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
    # Circular mean: a cluster straddling +/-pi would otherwise land its centroid at 0, on the
    # opposite side of the detector.
    sin_p = np.bincount(idx, weights=energy[clustered] * np.sin(phi[clustered]), minlength=n) / safe
    cos_p = np.bincount(idx, weights=energy[clustered] * np.cos(phi[clustered]), minlength=n) / safe
    cen_phi = np.arctan2(sin_p, cos_p)

    # Each cluster is built from one subsystem, so any of its cells identifies it.
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

    Cluster ids are offset per subsystem so they stay unique across the event; that is
    bookkeeping only, and `link_radius` is what actually joins a shower across a boundary.

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

    # Before min_cluster_hits, so the size cut sees the linked cluster and not its pieces.
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
