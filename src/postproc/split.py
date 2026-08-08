"""Split over-merged clusters at their local energy maxima.

WHY, AND WHY IT IS NOT AN UNDO OF `merge.py`
--------------------------------------------
After merging and chaining, **48.6% of assigned cells sit in the wrong particle's cluster**, and
82.1% of those cells have a SINGLE contributing particle -- the truth is unambiguous and the
assignment is simply wrong. Only ~12% is genuinely shared energy. So the misassignment is mostly
recoverable, and it is caused by growth: chaining wins the >= 0.5 metric by claiming cells
indiscriminately and being right often enough, which necessarily swallows neighbours.

Every operation in this package so far has made clusters BIGGER -- merge joins them, chain grows
them. Nothing has ever split one. This is the missing inverse, and it is the standard move in
production calorimeter reconstruction (ATLAS topological clusters are grown by significance and then
split wherever a cluster contains several local energy maxima).

It is not an undo of merging because it uses an INDEPENDENT criterion. `merge.py` joins clusters on
the mask head's cross-claims -- the model's opinion about which fragments are one shower. This
splits on the energy topology of the cells themselves. A pipeline that merged and split on the same
signal would oscillate; merging on model evidence and splitting on topology is the same separation
of concerns that lets ATLAS and Pandora run both without fighting.

The physical claim is simple, and it is the usable form of "a shower branches": a shower has one
core. A cluster with two well-separated cores is two showers.

HOW
---
Steepest ascent, then basin suppression:

1. Each cell points at its most energetic neighbour within `link_distance` **that belongs to the
   same cluster**. A cell with no more energetic neighbour is a local maximum.
2. Following those pointers assigns every cell to a basin -- a watershed on the energy landscape.
3. Basins whose peak is weaker than `min_peak_frac` of the cluster's strongest peak are noise rather
   than a second shower, so they are folded back into the nearest surviving basin.
4. A cluster with one surviving basin is returned untouched.

WHAT TO EXPECT
--------------
Purity up, efficiency down, because splitting is the opposite trade to growing. It is only a gain if
the efficiency it costs is smaller than the purity it buys, which is a question for the
efficiency-purity curve rather than for either number alone.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.postproc.merge import cells_by_cluster


def split_labels(
    record,
    label: np.ndarray,
    n: int,
    link_distance: float = 0.04,
    min_peak_frac: float = 0.25,
    min_peak_energy: float = 0.0,
    min_cluster_hits: int = 1,
) -> tuple[np.ndarray, int]:
    """Split each cluster at its local energy maxima.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        label: per cell, the cluster index, -1 where unclaimed.
        n: number of clusters.
        link_distance: neighbour radius in metres defining "adjacent" for the ascent.
        min_peak_frac: a second maximum must reach this fraction of the cluster's strongest peak to
            count as a separate shower. Lower splits more.
        min_peak_energy: absolute floor on a peak's calibrated energy, in GeV. 0 disables it.
        min_cluster_hits: sub-clusters smaller than this are dropped.

    Returns:
        ``(label, n_clusters)``, in the same form as the store's own labellers.
    """
    lab = np.asarray(label).copy()
    if n == 0 or lab.size == 0:
        return lab, n

    xyz = np.column_stack([record.x, record.y, record.z]).astype(np.float64)
    energy = np.asarray(record.energy_calib, dtype=np.float64)
    tree = cKDTree(xyz)

    # --- step 1: steepest-ascent pointers, confined within each cluster ------------------------
    neighbours = tree.query_ball_point(xyz, r=link_distance)
    ptr = np.arange(len(xyz))
    for i, idx in enumerate(neighbours):
        if lab[i] < 0 or not idx:
            continue
        cand = np.asarray(idx)
        cand = cand[lab[cand] == lab[i]]
        if cand.size == 0:
            continue
        best = cand[np.argmax(energy[cand])]
        if energy[best] > energy[i]:
            ptr[i] = best

    # --- step 2: follow to the basin root -------------------------------------------------------
    root = ptr.copy()
    for _ in range(int(np.ceil(np.log2(max(len(ptr), 2)))) + 1):
        nxt = root[root]
        if np.array_equal(nxt, root):
            break
        root = nxt

    # --- step 3: suppress weak basins ------------------------------------------------------------
    out = np.full(len(lab), -1, dtype=np.int64)
    next_id = 0
    cells = cells_by_cluster(lab, n)
    for c in range(n):
        k = cells[c]
        if k.size == 0:
            continue
        peaks, inverse = np.unique(root[k], return_inverse=True)
        if peaks.size == 1:
            out[k] = next_id
            next_id += 1
            continue
        peak_e = energy[peaks]
        strong = (peak_e >= min_peak_frac * peak_e.max()) & (peak_e >= min_peak_energy)
        if strong.sum() <= 1:
            out[k] = next_id
            next_id += 1
            continue
        # Weak basins join the nearest strong peak -- nearest in space, not in energy, because a
        # weak maximum is a fluctuation on somebody's flank and belongs to whichever shower it sits
        # on rather than to whichever is brightest.
        strong_idx = np.flatnonzero(strong)
        weak_idx = np.flatnonzero(~strong)
        target = np.arange(peaks.size)
        if weak_idx.size:
            d = np.linalg.norm(xyz[peaks[weak_idx]][:, None, :] - xyz[peaks[strong_idx]][None, :, :], axis=2)
            target[weak_idx] = strong_idx[d.argmin(axis=1)]
        new_ids = np.full(peaks.size, -1, dtype=np.int64)
        for s in strong_idx:
            new_ids[s] = next_id
            next_id += 1
        out[k] = new_ids[target[inverse]]

    if min_cluster_hits > 1 and next_id:
        counts = np.bincount(out[out >= 0], minlength=next_id)
        out[np.isin(out, np.flatnonzero(counts < min_cluster_hits))] = -1

    used = np.unique(out[out >= 0])
    if used.size == 0:
        return np.full(len(lab), -1, dtype=np.int32), 0
    remap = np.full(next_id, -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return np.where(out >= 0, remap[np.clip(out, 0, next_id - 1)], -1).astype(np.int32), int(used.size)
