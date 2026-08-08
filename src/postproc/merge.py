"""Merge the fragments a single shower was split across, using the mask head's own cross-claims.

WHY THIS, AFTER THE OTHERS FAILED
---------------------------------
Three facts from the epoch-6 checkpoint set this up.

1. A particle above 20 GeV has 31.3% of its energy inside clusters belonging to something else,
   spread over 4.9 fragments. `chain.py` cannot reach it -- it only fills unclaimed cells -- and
   merging fragments with truth as a guide takes E > 20 efficiency from 0.183 to 0.294, past CLUE's
   0.224. The energy is there and the assignment is what loses it.
2. The two largest fragments of a > 10 GeV particle hold **36% and 16.5%** of its energy. Merging
   just those two clears the 0.5 threshold on its own.
3. Geometry cannot decide which pairs to merge. A particle's own sibling fragments sit a median
   0.048 apart while genuinely different particles sit 0.008 apart -- the nearest cluster to a
   fragment is usually somebody else's. Any proximity rule merges the wrong things first.

So the merge decision needs a signal that is not geometric, and there is one. Measured over 1,982
pairs each, the mask probability query A assigns to the cells of another cluster:

    its own other fragment      mean 0.271   median 0.093   60.3% have any signal
    a different particle        mean 0.205   median 0.000   42.4% have any signal

Weak on the means and clean on the medians. Too weak to threshold on alone, which is why this
learns a small classifier over that feature plus geometry and energy rather than hand-tuning a cut.

WHY IT SHOULD BE SAFER THAN `flow.py` WAS
-----------------------------------------
Re-partitioning every cell by density scored 0.267 against a 0.377 baseline -- worse than doing
nothing -- because it discarded the core assignments the network gets right. Merging changes no cell
assignment at all. It only relabels which cluster a group of cells belongs to, so a wrong merge
costs purity on those clusters and nothing else, while the cores stay exactly where the model put
them. That is the property that separated chaining's success from flow's failure.

TRAINING DISCIPLINE
-------------------
The classifier is fitted on the TUNE window and applied to the eval window, the same separation
CLUE's Optuna parameters already get. Fitting it on the events it is scored over would make every
downstream number meaningless, so `fit_merge_model` refuses no store and it is the caller's job to
pass the tune one -- `scripts/tune_merge.py` is the entry point that does it correctly.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

FEATURES = (
    "cross_ab",      # mask probability A gives to B's cells
    "cross_ba",      # and the reverse; a shared shower should show it both ways
    "cross_max",
    "d_ang",         # centroid separation in (eta, phi)
    "d_rad",         # centroid separation in depth, metres
    "d_min_cell",    # closest approach of any two cells, metres -- adjacency, not centroid distance
    "e_ratio",       # smaller/larger calibrated energy
    "log_e_sum",
    "n_ratio",       # smaller/larger cell count
)


def cells_by_cluster(label: np.ndarray, n: int) -> list[np.ndarray]:
    """Cell indices for each cluster, grouped in one pass.

    The obvious ``[np.flatnonzero(label == c) for c in range(n)]`` is O(n_clusters x n_cells) --
    750 x 24,000 = 18 million comparisons per event here, and it was the hotspot that made the
    first version of this module take ~14 s per event. One argsort does the same work.
    """
    claimed = np.flatnonzero(label >= 0)
    if claimed.size == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n)]
    order = claimed[np.argsort(label[claimed], kind="stable")]
    counts = np.bincount(label[claimed], minlength=n)
    return np.split(order, np.cumsum(counts)[:-1])


def _cluster_geometry(record, label, n, cells=None):
    eta, phi, r = record.eta(), record.phi(), record.r()
    e = np.asarray(record.energy_calib, dtype=np.float64)
    cen = np.full((n, 3), np.nan)
    tot = np.zeros(n)
    size = np.zeros(n, dtype=int)
    cells = cells if cells is not None else cells_by_cluster(label, n)
    for c in range(n):
        k = cells[c]
        if k.size == 0:
            continue
        w = e[k]
        if w.sum() <= 0:
            continue
        ref = phi[k][np.argmax(w)]
        dphi = np.arctan2(np.sin(phi[k] - ref), np.cos(phi[k] - ref))
        cen[c] = [(eta[k] * w).sum() / w.sum(), ref + (dphi * w).sum() / w.sum(), (r[k] * w).sum() / w.sum()]
        tot[c] = w.sum()
        size[c] = k.size
    return cen, tot, size


def candidate_pairs(record, label, n, max_ang=0.15, max_pairs_per_cluster=6):
    """Cluster pairs close enough to be worth asking about.

    Not all pairs: an event has ~750 clusters and 280k pairs, almost all of them absurd. Restricting
    to the nearest few by centroid keeps the training set balanced and the inference cheap, at the
    cost of never merging two fragments that are far apart -- which the 0.048 median separation says
    is a safe trade.
    """
    cells = cells_by_cluster(label, n)
    cen, tot, size = _cluster_geometry(record, label, n, cells)
    live = np.flatnonzero(np.isfinite(cen[:, 0]))
    if live.size < 2:
        return np.empty((0, 2), dtype=int), cen, tot, size, cells
    tree = cKDTree(np.column_stack([cen[live, 0], cen[live, 1]]))
    pairs = set()
    k = min(max_pairs_per_cluster + 1, live.size)
    dist, idx = tree.query(np.column_stack([cen[live, 0], cen[live, 1]]), k=k)
    for i, row in enumerate(np.atleast_2d(idx)):
        for j, d in zip(row, np.atleast_2d(dist)[i], strict=False):
            if j == i or not np.isfinite(d) or d > max_ang:
                continue
            a, b = int(live[i]), int(live[j])
            pairs.add((min(a, b), max(a, b)))
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2), cen, tot, size, cells


def pair_features(record, label, n, pairs, soft, cen, tot, size, cells=None):
    """Feature matrix for `pairs`, in the order given by :data:`FEATURES`."""
    if pairs.size == 0:
        return np.empty((0, len(FEATURES)))
    xyz = np.column_stack([record.x, record.y, record.z]).astype(np.float64)
    cells = cells if cells is not None else cells_by_cluster(label, n)
    trees = {}

    rows = []
    for a, b in pairs:
        ca, cb = cells[a], cells[b]
        cross_ab = float(np.asarray(soft[cb, a].todense()).mean()) if soft is not None and cb.size else 0.0
        cross_ba = float(np.asarray(soft[ca, b].todense()).mean()) if soft is not None and ca.size else 0.0
        d_ang = float(np.hypot(cen[a, 0] - cen[b, 0],
                               np.arctan2(np.sin(cen[a, 1] - cen[b, 1]), np.cos(cen[a, 1] - cen[b, 1]))))
        d_rad = float(abs(cen[a, 2] - cen[b, 2]))
        if ca.size and cb.size:
            # Cache the per-cluster tree: a cluster appears in up to six pairs, and rebuilding its
            # tree each time was pure repeat work.
            if a not in trees:
                trees[a] = cKDTree(xyz[ca])
            d_min = float(trees[a].query(xyz[cb], k=1)[0].min())
        else:
            d_min = np.inf
        lo, hi = (tot[a], tot[b]) if tot[a] <= tot[b] else (tot[b], tot[a])
        nlo, nhi = (size[a], size[b]) if size[a] <= size[b] else (size[b], size[a])
        rows.append([
            cross_ab, cross_ba, max(cross_ab, cross_ba), d_ang, d_rad, d_min,
            lo / max(hi, 1e-12), np.log10(max(tot[a] + tot[b], 1e-12)), nlo / max(nhi, 1),
        ])
    return np.asarray(rows, dtype=np.float64)


def pair_truth(record, label, n, pairs, cells=None):
    """1 where both clusters are dominated by the same truth particle."""
    tl = np.asarray(record.truth_label)
    fac = np.divide(record.energy_calib, np.maximum(record.energy, 1e-12))
    dep = record.truth_deposit * fac
    cells = cells if cells is not None else cells_by_cluster(label, n)
    owner = np.full(n, -1)
    for c in range(n):
        k = cells[c]
        k = k[tl[k] >= 0]
        if k.size:
            owner[c] = int(np.bincount(tl[k], weights=dep[k]).argmax())
    return np.array([1 if (owner[a] >= 0 and owner[a] == owner[b]) else 0 for a, b in pairs], dtype=int)


def apply_merges(label, n, pairs, decision):
    """Union the clusters flagged for merging, and compact the ids."""
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), d in zip(pairs, decision, strict=False):
        if d:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(n)])
    out = np.where(label >= 0, roots[np.clip(label, 0, n - 1)], -1)
    used = np.unique(out[out >= 0])
    if used.size == 0:
        return np.full_like(label, -1), 0
    remap = np.full(n, -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return np.where(out >= 0, remap[np.clip(out, 0, n - 1)], -1).astype(np.int32), int(used.size)
