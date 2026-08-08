"""Decide which of several overlapping showers owns a contested cell, instead of taking the nearest.

WHY THIS IS THE REMAINING PROBLEM
---------------------------------
The energy budget after merging and chaining, per particle above 20 GeV (100 events):

    in its matched cluster   0.309
    unclaimed by anything    0.086
    in ANOTHER cluster       0.605

Coverage is finished -- 8.6% unclaimed, against CLUE's 17.8% and the raw model's 41.0%. Growing
harder cannot win more than that 8.6%. Sixty percent of a high-energy shower now sits in a cluster
belonging to a different particle, so every remaining point is an ATTRIBUTION point: not "should
this cell be claimed" but "by which of the showers reaching it".

`chain.py` answers that with the nearest already-claimed neighbour, which is close to a coin flip
where it matters: in a jet core the median distance to the next particle is 0.008 while a
high-energy shower spans 0.233, so several showers are always within reach. That tie-break is the
single crudest decision left in the pipeline, and it is applied to the majority of the energy.

WHAT THIS DOES INSTEAD
----------------------
For each (cell, candidate cluster) pair it learns whether that cluster's dominant particle is the
cell's true exclusive owner, from features a calorimeter actually offers:

* how far the cell sits from the cluster's core, in angle and in depth SEPARATELY -- showers are
  long and narrow, so one combined distance throws away the asymmetry that distinguishes "further
  along my own shower" from "sideways into someone else's"
* the cluster's core energy and the cell's own energy, since a big shower legitimately reaches
  further than a small one and a bright cell far from a weak core probably is not its
* the mask head's probability for that (cell, cluster) pair, which was the dominant feature in the
  merge classifier and is the one signal that is not geometric

Trained on the tune window, applied to the eval window, exactly as `merge.py` is.

THE HONEST PRIOR
----------------
The resolution ceiling says ~half of clusters unavoidably contain more than one target particle,
and 4.2% of cells carry 20.8% of the energy with several genuine contributors. A share of that 60.5%
is not misattributed at all -- it is shared energy that an exclusive answer cannot represent, and no
classifier recovers it. `scripts/score_soft.py` measures that part; this measures the rest.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from src.postproc.merge import cells_by_cluster

FEATURES = (
    "d_ang",         # angular offset from the cluster core
    "d_rad",         # depth offset, metres -- kept separate from the angular one on purpose
    "d_cell",        # 3D distance to the nearest cell already in the cluster
    "log_core_e",    # the cluster's energy: a bright shower reaches further
    "log_cell_e",    # the cell's own energy
    "e_frac",        # cell energy relative to the cluster core
    "mask_p",        # the mask head's probability for this (cell, cluster) pair
    "core_size",
)


def _cluster_cores(record, label, n, cells):
    eta, phi, r = record.eta(), record.phi(), record.r()
    e = np.asarray(record.energy_calib, dtype=np.float64)
    cen = np.full((n, 3), np.nan)
    tot = np.zeros(n)
    size = np.zeros(n, dtype=int)
    for c in range(n):
        k = cells[c]
        if k.size == 0 or e[k].sum() <= 0:
            continue
        w = e[k]
        ref = phi[k][np.argmax(w)]
        d = np.arctan2(np.sin(phi[k] - ref), np.cos(phi[k] - ref))
        cen[c] = [(eta[k] * w).sum() / w.sum(), ref + (d * w).sum() / w.sum(), (r[k] * w).sum() / w.sum()]
        tot[c] = w.sum()
        size[c] = k.size
    return cen, tot, size


def candidate_cells(record, label, n, reach=0.08, max_candidates=5):
    """(cell, cluster) pairs worth deciding: unclaimed cells with clusters within `reach`.

    Only unclaimed cells are offered. Reassigning cells the model already placed is what made
    `flow.py` score 0.267 against a 0.377 baseline, and that lesson is load-bearing here: this
    replaces the chaining TIE-BREAK, not the seeds.
    """
    cells = cells_by_cluster(label, n)
    cen, tot, size = _cluster_cores(record, label, n, cells)
    live = np.flatnonzero(np.isfinite(cen[:, 0]))
    unclaimed = np.flatnonzero(label < 0)
    if live.size == 0 or unclaimed.size == 0:
        return np.empty((0, 2), dtype=int), cells, cen, tot, size

    eta, phi = record.eta(), record.phi()
    tree = cKDTree(np.column_stack([cen[live, 0], cen[live, 1]]))
    k = min(max_candidates, live.size)
    dist, idx = tree.query(np.column_stack([eta[unclaimed], phi[unclaimed]]), k=k)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    ok = np.isfinite(dist) & (dist <= reach) & (idx < live.size)
    rows = np.repeat(unclaimed, k)[ok.ravel()]
    cols = live[np.clip(idx, 0, live.size - 1)].ravel()[ok.ravel()]
    return np.column_stack([rows, cols]).astype(int), cells, cen, tot, size


def cell_features(record, label, n, pairs, soft, cells, cen, tot, size):
    if pairs.size == 0:
        return np.empty((0, len(FEATURES)))
    eta, phi, r = record.eta(), record.phi(), record.r()
    e = np.asarray(record.energy_calib, dtype=np.float64)
    xyz = np.column_stack([record.x, record.y, record.z]).astype(np.float64)
    cell_i, clus = pairs[:, 0], pairs[:, 1]

    d_ang = np.hypot(eta[cell_i] - cen[clus, 0],
                     np.arctan2(np.sin(phi[cell_i] - cen[clus, 1]), np.cos(phi[cell_i] - cen[clus, 1])))
    d_rad = np.abs(r[cell_i] - cen[clus, 2])

    d_cell = np.full(len(pairs), np.inf)
    trees = {}
    for c in np.unique(clus):
        k = cells[c]
        if k.size == 0:
            continue
        trees[c] = cKDTree(xyz[k])
    for c, tree in trees.items():
        sel = clus == c
        d_cell[sel] = tree.query(xyz[cell_i[sel]], k=1)[0]

    mask_p = np.zeros(len(pairs))
    if soft is not None:
        mask_p = np.asarray(soft[cell_i, clus]).ravel()

    return np.column_stack([
        d_ang, d_rad, d_cell,
        np.log10(np.maximum(tot[clus], 1e-12)), np.log10(np.maximum(e[cell_i], 1e-12)),
        e[cell_i] / np.maximum(tot[clus], 1e-12), mask_p, size[clus],
    ])


def cell_truth(record, label, n, pairs, cells):
    """1 where the candidate cluster's dominant particle is this cell's true exclusive owner."""
    tl = np.asarray(record.truth_label)
    fac = np.divide(record.energy_calib, np.maximum(record.energy, 1e-12))
    dep = record.truth_deposit * fac
    owner = np.full(n, -1)
    for c in range(n):
        k = cells[c]
        k = k[tl[k] >= 0]
        if k.size:
            owner[c] = int(np.bincount(tl[k], weights=dep[k]).argmax())
    return ((owner[pairs[:, 1]] >= 0) & (owner[pairs[:, 1]] == tl[pairs[:, 0]])).astype(int)


def attribute_labels(label, n, pairs, scores, min_score=0.0):
    """Award each contested cell to its best-scoring candidate cluster."""
    out = np.asarray(label).copy()
    if pairs.size == 0:
        return out, n
    order = np.lexsort((-scores, pairs[:, 0]))
    cell_sorted = pairs[order, 0]
    first = np.empty(cell_sorted.size, dtype=bool)
    first[0] = True
    first[1:] = cell_sorted[1:] != cell_sorted[:-1]
    best = order[first]
    take = scores[best] >= min_score
    out[pairs[best, 0][take]] = pairs[best, 1][take].astype(out.dtype)
    return out, n
