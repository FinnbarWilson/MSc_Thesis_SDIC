"""Re-partition every cell by density flow, seeded on MaskFormer's cores.

THE IDEA, AND WHY IT IS NOT THE SAME AS CHAINING
------------------------------------------------
`src.postproc.chain` grows the predicted clusters into cells nobody claimed. That leaves the other
failure untouched: measured on the epoch-6 checkpoint, a particle above 20 GeV has 41.7% of its
energy in unclaimed cells but a further **31.3% inside clusters belonging to something else**, split
across 4.9 fragments. Chaining cannot reach that energy by construction, because it never reassigns
a cell that already has an owner.

This module reassigns all of them. Every cell — claimed or not — attaches to its nearest DENSER
neighbour, that neighbour to its own, and the chain is followed until it reaches an anchor. Anchors
are MaskFormer's cluster cores. So the partition is decided by the density landscape, exactly as in
CLUE, while the question of *how many objects there are and where* is answered by the network.

That division of labour is the point. The measurements say each method is good at a different half:

* MaskFormer knows the objects. It finds a confident ~6-cell core for essentially every particle
  (86% of >20 GeV particles get a matched cluster) but cannot grow one to the size of a shower.
* CLUE knows the cells. Its chaining reaches p99 = 263 cells against MaskFormer's 42, but it has to
  *guess* the objects from density peaks, and it misses 36% of high-energy particles outright.

Merging fragments after the fact was the obvious alternative and the geometry rules it out: a
particle's own sibling fragments sit a median 0.048 apart while genuinely different particles sit
0.008 apart, so the nearest cluster to a fragment is usually somebody else. Dissolving the fragments
into a fresh partition sidesteps that problem rather than trying to solve it.

WHAT TO WATCH
-------------
This can *lose* to plain chaining, and the reason would be informative rather than a bug. Anchoring
on every MaskFormer cluster anchors the over-segmentation too — 753 clusters per event against 538
true particles — so a shower split across five queries provides five anchors, and density flow will
happily divide it five ways again. `anchor_min_energy` exists to suppress the weakest of them.
Judge on efficiency AND purity together, as always.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

K_NEIGHBOURS = 48


def _nearest_denser(xyz, rho, link_distance, tree):
    """Index of the nearest cell with strictly greater density, or -1 if there is none in range.

    `cKDTree.query` returns neighbours already sorted by distance, so the first one that is denser
    IS the nearest denser one and no second sort is needed.
    """
    n = len(xyz)
    dist, idx = tree.query(xyz, k=min(K_NEIGHBOURS, n), distance_upper_bound=link_distance)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    valid = np.isfinite(dist) & (idx < n)
    safe = np.clip(idx, 0, n - 1)
    denser = valid & (rho[safe] > rho[:, None])
    has = denser.any(axis=1)
    first = np.where(has, denser.argmax(axis=1), 0)
    ndn = np.where(has, safe[np.arange(n), first], -1)
    return ndn


def _follow(ndn: np.ndarray) -> np.ndarray:
    """Root of each cell's pointer chain, by pointer jumping (log-depth, no Python loop per cell)."""
    root = np.where(ndn >= 0, ndn, np.arange(len(ndn)))
    for _ in range(int(np.ceil(np.log2(max(len(ndn), 2)))) + 1):
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt
    return root


def flow_labels(
    record,
    base_label: np.ndarray,
    n_base: int,
    link_distance: float = 0.05,
    density_radius: float | None = None,
    anchor_min_energy: float = 0.0,
    min_cluster_hits: int = 1,
) -> tuple[np.ndarray, int]:
    """Reassign every cell to a MaskFormer core by following the density gradient.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        base_label: per cell, the seed cluster index, -1 where unclaimed. Used only to locate the
            cores; the output partition is built from scratch.
        n_base: number of seed clusters.
        link_distance: neighbour radius in metres for the density-gradient step.
        density_radius: radius for the local density. Defaults to ``link_distance``.
        anchor_min_energy: drop seed clusters holding less calibrated energy than this before
            anchoring, so the weakest fragments do not each claim a share of a shower. 0 keeps all.
        min_cluster_hits: clusters smaller than this are dropped, matching the store's convention.

    Returns:
        ``(label, n_clusters)``, in the same form as the store's own labellers.
    """
    label_in = np.asarray(base_label)
    if n_base == 0 or label_in.size == 0:
        return np.full(record.n_hits, -1, dtype=np.int32), 0

    xyz = np.column_stack([record.x, record.y, record.z]).astype(np.float64)
    energy = np.asarray(record.energy_calib, dtype=np.float64)
    tree = cKDTree(xyz)

    radius = density_radius or link_distance
    rho = np.array([energy[i].sum() for i in tree.query_ball_point(xyz, r=radius)])

    # One anchor per seed cluster: its densest cell. Using the densest rather than the
    # energy-weighted centroid keeps the anchor ON a real cell, which matters because the pointer
    # chains have to terminate somewhere that exists.
    anchor_cell = np.full(n_base, -1, dtype=np.int64)
    for c in range(n_base):
        cells = np.flatnonzero(label_in == c)
        if cells.size == 0:
            continue
        if anchor_min_energy > 0 and energy[cells].sum() < anchor_min_energy:
            continue
        anchor_cell[c] = cells[np.argmax(rho[cells])]

    live = np.flatnonzero(anchor_cell >= 0)
    if live.size == 0:
        return np.full(record.n_hits, -1, dtype=np.int32), 0

    ndn = _nearest_denser(xyz, rho, link_distance, tree)

    # Anchors terminate their chains. Without this a chain would climb straight past a core to the
    # global density maximum of the jet and every cell in the neighbourhood would land in one
    # cluster -- which is exactly the failure CLUE controls with rho_c and d_c.
    ndn[anchor_cell[live]] = -1

    root = _follow(ndn)

    cluster_of_cell = np.full(record.n_hits, -1, dtype=np.int64)
    cluster_of_cell[anchor_cell[live]] = live
    label = cluster_of_cell[root].astype(np.int32)

    # Compact, so downstream code sees 0..n-1 as it does from maskformer_labels.
    if min_cluster_hits > 1:
        counts = np.bincount(label[label >= 0], minlength=n_base)
        label[np.isin(label, np.flatnonzero(counts < min_cluster_hits))] = -1
    used = np.unique(label[label >= 0])
    if used.size == 0:
        return np.full(record.n_hits, -1, dtype=np.int32), 0
    remap = np.full(n_base, -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    out = np.where(label >= 0, remap[np.clip(label, 0, n_base - 1)], -1).astype(np.int32)
    return out, int(used.size)
