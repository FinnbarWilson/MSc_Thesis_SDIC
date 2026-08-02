"""The classical baseline: a two-stage CLUE pipeline over calorimeter cells.

Running CLUE once in three dimensions tends to merge showers that overlap in
depth, so the pipeline follows the two-stage strategy used for high-granularity
calorimeter reconstruction: cluster within each detector layer first, reduce each
layer cluster to a single centroid, then cluster the centroids in three
dimensions to link the per-layer pieces of one shower into a trackster.

Both passes call the same CLUEstering library, which is possible only because it
accepts an arbitrary number of dimensions. Nothing here modifies the algorithm:
the parameters are CLUE's own density radius, density threshold and linking
radius, and the periodic distance metric is a CLUEstering feature.

Coordinates are either Cartesian ``(x, y)`` in mm or projective ``(eta, phi)``.
The angular choice suits the detector geometry, since a shower keeps an
approximately constant angular size as it propagates outward while its
transverse extent in mm grows with radius. It requires phi to be treated as
periodic so a shower straddling +/-pi is not split in two.
"""

import numpy as np
import pandas as pd

import CLUEstering as clue

TWO_PI = 2.0 * np.pi


def cartesian_to_eta_phi(x, y, z):
    """Convert Cartesian coordinates in mm to pseudorapidity and azimuth."""
    phi = np.arctan2(y, x)
    theta = np.arctan2(np.hypot(x, y), z)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = -np.log(np.tan(theta / 2.0))
    return eta, phi


def _clustering_coordinates(event, hit_mask, coords):
    """The two transverse coordinates CLUE clusters on, for the selected cells."""
    x = np.asarray(event["x"], dtype=float)[hit_mask]
    y = np.asarray(event["y"], dtype=float)[hit_mask]
    z = np.asarray(event["z"], dtype=float)[hit_mask]
    if coords == "etaphi":
        return cartesian_to_eta_phi(x, y, z)
    return x, y


def _run_clue(coord_a, coord_b, weights, params, suffix, coords, backend):
    """Run one two-dimensional CLUE pass and return its cluster labels.

    Args:
        coord_a, coord_b: the two clustering coordinates.
        weights: cell energies in GeV.
        params: parameter dictionary holding ``d_c_<suffix>``, ``rho_c_<suffix>``
            and ``d_o_<suffix>``.
        suffix: ``"2d"`` or ``"3d"``, selecting which parameters to use.
        coords: ``"etaphi"`` or ``"xy"``.
        backend: CLUEstering compute backend.

    Returns:
        np.ndarray: cluster id per input point, -1 for points called noise.
    """
    clusterer = clue.clusterer(
        params[f"d_c_{suffix}"], params[f"rho_c_{suffix}"], params[f"d_o_{suffix}"]
    )
    clusterer.read_data(np.array([coord_a, coord_b, weights]))
    if coords == "etaphi":
        clusterer.choose_metric("periodic_euclidean", parameters=[0.0, TWO_PI])
    clusterer.run_clue(backend=backend)
    return clusterer.output_df["cluster_ids"].values


def _layer_clusters(coord_a, coord_b, weights, layer_z, params, coords, backend):
    """Cluster each detector layer separately and return a label per cell.

    Cells are grouped by their discrete ``z``, which is what defines a layer, and
    CLUE is run within each group. Labels are made unique across layers by the
    caller, since CLUE restarts its numbering for every call.
    """
    labels = np.full(len(weights), -1, dtype=int)
    order = np.argsort(layer_z, kind="stable")
    for layer in np.split(order, np.flatnonzero(np.diff(layer_z[order])) + 1):
        labels[layer] = _run_clue(
            coord_a[layer], coord_b[layer], weights[layer],
            params, "2d", coords, backend,
        )
    return labels


def _layer_centroids(frame, z_scale, coords):
    """Reduce each layer cluster to one energy-weighted centroid.

    The third coordinate of the centroid is the layer's ``z`` divided by
    ``z_scale``. That division is a unit conversion rather than a tuning trick:
    without it the depth axis is in mm while the two angular axes are of order
    one, and a single density radius cannot mean anything in both.

    Azimuth is averaged circularly, because a plain mean over a cluster straddling
    +/-pi would land on the opposite side of the detector.
    """
    group = frame["layer_cluster"].to_numpy()
    weights = frame["weight"].to_numpy()
    n_groups = int(group.max()) + 1
    total = np.bincount(group, weights=weights, minlength=n_groups)

    if coords == "etaphi":
        centre_a = np.bincount(group, weights=weights * frame["coord_a"].to_numpy(),
                               minlength=n_groups) / total
        sin_phi = np.bincount(group, weights=weights * np.sin(frame["coord_b"]),
                              minlength=n_groups)
        cos_phi = np.bincount(group, weights=weights * np.cos(frame["coord_b"]),
                              minlength=n_groups)
        centre_b = np.arctan2(sin_phi, cos_phi)
    else:
        centre_a = np.bincount(group, weights=weights * frame["coord_a"].to_numpy(),
                               minlength=n_groups) / total
        centre_b = np.bincount(group, weights=weights * frame["coord_b"].to_numpy(),
                               minlength=n_groups) / total

    depth = np.empty(n_groups)
    depth[group] = frame["layer_z"].to_numpy()

    return pd.DataFrame({
        "layer_cluster": np.arange(n_groups),
        "coord_a": centre_a,
        "coord_b": centre_b,
        "depth": depth / z_scale,
        "weight": total,
    })


def _link_centroids(centroids, params, coords, backend):
    """Cluster the layer centroids in three dimensions into tracksters."""
    clusterer = clue.clusterer(params["d_c_3d"], params["rho_c_3d"], params["d_o_3d"])
    clusterer.read_data(np.array([
        centroids["coord_a"].to_numpy(),
        centroids["coord_b"].to_numpy(),
        centroids["depth"].to_numpy(),
        centroids["weight"].to_numpy(),
    ]))
    if coords == "etaphi":
        clusterer.choose_metric("periodic_euclidean", parameters=[0.0, TWO_PI, 0.0])
    clusterer.run_clue(backend=backend)
    return clusterer.output_df["cluster_ids"].values


def cluster_detector(event, detector, params, hit_mask=None, coords="etaphi",
                     backend="cpu serial"):
    """Cluster one detector collection of one event.

    Args:
        event: one event row.
        detector: detector collection name.
        params: ``d_c_2d``, ``rho_c_2d``, ``d_o_2d``, ``d_c_3d``, ``rho_c_3d``,
            ``d_o_3d`` and ``z_scale``.
        hit_mask: optional boolean mask over cells, applied on top of the
            detector selection. Used to pass the shared energy threshold in.
        coords: ``"etaphi"`` or ``"xy"``.
        backend: CLUEstering compute backend.

    Returns:
        tuple: ``(cluster_ids, selected)`` both of length equal to the number of
        cells in the event. ``cluster_ids`` is -1 for noise and for cells outside
        this detector; ``selected`` marks the cells this call considered.
    """
    n_hits = len(event["x"])
    cluster_ids = np.full(n_hits, -1, dtype=int)

    selected = np.asarray(event["detector"]) == detector
    if hit_mask is not None:
        selected = selected & np.asarray(hit_mask)
    if not selected.any():
        return cluster_ids, selected

    coord_a, coord_b = _clustering_coordinates(event, selected, coords)
    weights = np.asarray(event["total_energy"], dtype=float)[selected]
    layer_z = np.asarray(event["z"], dtype=float)[selected]

    labels = _layer_clusters(coord_a, coord_b, weights, layer_z, params, coords, backend)
    frame = pd.DataFrame({
        "coord_a": coord_a,
        "coord_b": coord_b,
        "layer_z": layer_z,
        "weight": weights,
        "label": labels,
        "position": np.flatnonzero(selected),
    })
    frame = frame[frame["label"] != -1]
    if frame.empty:
        return cluster_ids, selected

    frame["layer_cluster"] = frame.groupby(["layer_z", "label"]).ngroup()
    centroids = _layer_centroids(frame, params["z_scale"], coords)
    centroids["trackster"] = _link_centroids(centroids, params, coords, backend)

    trackster_of = dict(zip(centroids["layer_cluster"], centroids["trackster"]))
    cluster_ids[frame["position"].to_numpy()] = frame["layer_cluster"].map(trackster_of).to_numpy()
    return cluster_ids, selected


def cluster_event(event, params_by_detector, detectors, hit_mask=None,
                  coords="etaphi", backend="cpu serial"):
    """Cluster every detector collection and merge them into one labelling.

    Cluster ids are offset per detector so that they stay unique across the
    event. This is bookkeeping only: it does not link a shower across detector
    boundaries. Jets rejoin an electromagnetic shower with its hadronic
    continuation at the jet level instead.

    Args:
        event: one event row.
        params_by_detector: tuned parameters keyed by detector name.
        detectors: detector collections to process.
        hit_mask: optional boolean mask over cells.
        coords: ``"etaphi"`` or ``"xy"``.
        backend: CLUEstering compute backend.

    Returns:
        tuple: ``(cluster_ids, per_detector)`` where ``cluster_ids`` covers the
        whole event and ``per_detector`` maps each detector to its own
        ``(cluster_ids, selected)`` pair for per-detector scoring.
    """
    merged = np.full(len(event["x"]), -1, dtype=int)
    per_detector = {}
    offset = 0

    for detector in detectors:
        if detector not in params_by_detector:
            continue
        ids, selected = cluster_detector(
            event, detector, params_by_detector[detector],
            hit_mask=hit_mask, coords=coords, backend=backend,
        )
        per_detector[detector] = (ids, selected)

        clustered = ids != -1
        if clustered.any():
            merged[clustered] = ids[clustered] + offset
            offset = int(merged.max()) + 1

    return merged, per_detector
