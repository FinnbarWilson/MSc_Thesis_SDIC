"""Extend each cluster along its own shower axis, using the core to estimate the direction.

WHY THIS ANGLE
--------------
The geometric ceiling in `results/` reaches 0.608 overall and 0.295 above 20 GeV by assigning every
cell to the nearest TRUE shower axis. That is the best any purely geometric assignment can do, and
it is far above where either real method sits — so the axis is evidently powerful information. No
algorithm has the true axis, but MaskFormer supplies something close: its core is the one thing it
gets right, and a core is a handful of the shower's most energetic cells, which is exactly what
points along the shower.

The failure this targets is longitudinal and it is severe. Measured on the epoch-6 checkpoint, a
particle above 10 GeV deposits over a **0.418 m** depth span and its matched cluster covers
**0.011 m** of that — 3%. The mask is essentially flat in depth: a two-dimensional patch at one
layer, cut out of a three-dimensional shower. Neither chaining nor a wider position encoding moved
that. Growing explicitly ALONG the radial direction is the one operation aimed straight at it.

WHAT IT DOES
------------
For each seed cluster, take the energy-weighted centroid of its cells as a direction from the
interaction point, then claim unclaimed cells that lie within `road_width` of that ray in angle,
regardless of how deep they are. A shower is a narrow road pointing away from the origin, so the
test is "is this cell on my ray", not "is this cell near my cells".

THE OBVIOUS RISK, STATED UP FRONT
---------------------------------
At pu0 the median distance between neighbouring particles is 0.054 and only 0.008 for the energetic
ones, while showers are 0.08-0.23 wide. Two particles in a jet core share a ray to within far less
than any usable `road_width`, so a road will collect a neighbour's cells as readily as its own.
Expect efficiency up and purity down, and expect the trade to be worse than plain chaining unless
the depth argument dominates. Contested cells go to the nearest ray, which is the least-bad
tie-break available without truth.
"""

from __future__ import annotations

import numpy as np


def axis_labels(
    record,
    base_label: np.ndarray,
    n_base: int,
    road_width: float = 0.03,
    min_core_cells: int = 2,
    min_cluster_hits: int = 1,
) -> tuple[np.ndarray, int]:
    """Grow each seed cluster along its own outward ray into unclaimed cells.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        base_label: seed cluster per cell, -1 where unclaimed. Seeds are never overwritten.
        n_base: number of seed clusters.
        road_width: angular half-width of the road, in (eta, phi) units. A cell joins if it is
            within this of a cluster's ray and no other ray is closer.
        min_core_cells: clusters with fewer cells than this do not define a usable direction and
            are left as they are.
        min_cluster_hits: clusters smaller than this are dropped afterwards.

    Returns:
        ``(label, n_clusters)``, in the same form as the store's own labellers.
    """
    label = np.asarray(base_label).copy()
    if n_base == 0 or label.size == 0:
        return label, n_base

    eta = record.eta()
    phi = record.phi()
    energy = np.asarray(record.energy_calib, dtype=np.float64)

    # One ray per cluster: the energy-weighted direction of its core. Energy weighting rather than
    # a plain mean because the core's own outliers are the cells most likely to have been picked up
    # from a neighbour, and they carry little energy.
    axis_eta = np.full(n_base, np.nan)
    axis_phi = np.full(n_base, np.nan)
    for c in range(n_base):
        cells = np.flatnonzero(label == c)
        if cells.size < min_core_cells:
            continue
        w = energy[cells]
        if w.sum() <= 0:
            continue
        axis_eta[c] = (eta[cells] * w).sum() / w.sum()
        ref = phi[cells][np.argmax(w)]
        d = np.arctan2(np.sin(phi[cells] - ref), np.cos(phi[cells] - ref))
        axis_phi[c] = ref + (d * w).sum() / w.sum()

    live = np.flatnonzero(np.isfinite(axis_eta))
    unclaimed = np.flatnonzero(label < 0)
    if live.size == 0 or unclaimed.size == 0:
        return label, n_base

    # Distance from every unclaimed cell to every ray. n_unclaimed x n_live is ~14k x 900 at pu0,
    # about 12M floats -- large but transient, and far simpler than a spatial index over rays.
    d_eta = eta[unclaimed][:, None] - axis_eta[live][None, :]
    d_phi = np.arctan2(
        np.sin(phi[unclaimed][:, None] - axis_phi[live][None, :]),
        np.cos(phi[unclaimed][:, None] - axis_phi[live][None, :]),
    )
    dist = np.hypot(d_eta, d_phi)

    best = dist.argmin(axis=1)
    best_d = dist[np.arange(len(unclaimed)), best]
    take = best_d <= road_width
    label[unclaimed[take]] = live[best[take]].astype(np.int32)

    if min_cluster_hits > 1:
        counts = np.bincount(label[label >= 0], minlength=n_base)
        label[np.isin(label, np.flatnonzero(counts < min_cluster_hits))] = -1

    return label, n_base
