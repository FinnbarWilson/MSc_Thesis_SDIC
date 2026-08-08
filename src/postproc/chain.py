"""Grow MaskFormer's clusters outwards by chaining, the way CLUE does.

WHY THIS EXISTS, measured rather than assumed
---------------------------------------------
MaskFormer's mask head decides membership one cell at a time: the mask logit is a dot product
between a query and a single cell, so a cell either resembles the query or it does not. There is no
mechanism by which a cell can join because it is adjacent to a cell that joined. CLUE's membership
is the opposite -- each cell attaches to its nearest denser neighbour, that one attaches to its own,
and a cell far from the core joins through a *path*.

Measured on the epoch-6 checkpoint, that difference is exactly what the scores look like. The
matched cluster holds ~6 cells whatever the particle's true size, spanning ~0.06 in angle and ~1 cm
of depth against showers that reach 0.23 and 0.42 m; and 35% of a high-energy particle's energy sits
in cells that no query gives even 2% probability, so no working point recovers it (checked to
mask 0.02 / object 0.001). Five training-side interventions failed to move any of this --
`src/maskformer/HIGH_ENERGY_STATUS.md` has them.

So this module does not try to fix the model. It takes the model's clusters as SEEDS, which is what
the mask head is good at, and adds the one thing it structurally cannot do.

WHAT IT DOES
------------
Multi-source growth over a neighbour graph, in three steps:

1. Build the graph once per event: every pair of cells within `link_distance` metres of each other,
   in real 3D space. Cartesian rather than (eta, phi, layer) so that phi wrapping cannot create a
   false gap at +-pi, and so the metric means the same thing in the barrel and the endcaps.
2. Seed it with the predicted clusters, unchanged.
3. Repeatedly hand each still-unclaimed cell the label of its nearest already-claimed neighbour,
   until nothing changes or `max_rounds` is reached.

Two properties are deliberate. **Seeds are never overwritten** -- growth only fills unclaimed cells,
so the cores the network found survive intact and this can only add. And when two clusters grow
towards each other they meet and stop, because each cell is claimed once by whichever seed reaches
it first: the unclaimed space ends up partitioned between the seeds rather than duplicated.

WHAT TO EXPECT, AND THE HONEST CEILING
--------------------------------------
Merging each particle's own fragments with truth as a guide took E > 20 GeV efficiency from 0.183 to
0.294, against CLUE's 0.224. That is an ORACLE -- it knew which fragments belonged together. This
does not, so treat 0.294 as an upper bound that will not be reached, not a forecast.

Expect purity to fall. Growing into contested cells is precisely the trade CLUE already makes
(purity 0.251 against MaskFormer's 0.274), and 4.2% of cells here carry 20.8% of the energy with
more than one contributing particle. Judge this on efficiency AND purity together, never on
efficiency alone -- a chainer that swallows the event scores wonderfully on one and uselessly on the
other.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def _local_density(xyz: np.ndarray, energy: np.ndarray, radius: float, tree: cKDTree) -> np.ndarray:
    """Calibrated energy within `radius` of each cell -- CLUE's rho, on the same neighbour scale.

    Density is what tells two overlapping showers apart. Purely geometric growth attaches a cell to
    whichever cluster happens to be nearest, which in a jet core (median neighbour separation 0.008
    against a 0.23 shower radius at E > 20 GeV) is close to a coin flip. Following the density
    gradient instead means a contested cell flows towards the core it actually sits on.
    """
    neighbours = tree.query_ball_point(xyz, r=radius)
    rho = np.empty(len(xyz))
    for i, idx in enumerate(neighbours):
        rho[i] = energy[idx].sum()
    return rho


def _soft_claim_matrix(record, mask_threshold, object_threshold, min_cluster_hits, floor):
    """Sparse [cell x cluster] of the model's own mask probabilities, down to `floor`.

    THIS IS THE INFORMATION `maskformer_labels` DESTROYS. The mask head emits an independent
    probability per (query, cell) and the store keeps every one above 0.02; thresholding at 0.5
    collapses that to a binary owner and throws the rest away. A cell the model gives 0.3 to one
    query and 0.05 to another is not "unclaimed" -- it is a cell with an opinion about which shower
    it belongs to, and that opinion is what growth needs when geometry is ambiguous.

    THE CLUSTER NUMBERING IS THE WHOLE DIFFICULTY, and getting it wrong is silent. Both
    `maskformer_labels` and `maskformer_soft_masks` compact their cluster ids with
    ``np.unique(rows)`` over the queries THEY accept, and they accept different sets: at 0.5/0.2
    one event gave 889 clusters, at 0.02/0.2 it gave 960. Indexing a soft matrix built at 0.02 with
    a label id built at 0.5 therefore reads a different query's probability, and the first version
    of this module did exactly that -- it produced bit-identical results to no soft masks at all,
    which is what gave it away. So the query selection is replicated here from
    ``EventRecord.maskformer_labels`` to recover ITS numbering, and the low-floor probabilities are
    then mapped onto it.
    """
    from src.io.event_store import logit_code_for_threshold, probability_for_logit_code

    n_q = int(record.mf_valid_prob.size)
    if n_q == 0 or record.mf_indices.size == 0:
        return None
    all_rows = np.repeat(np.arange(n_q), np.diff(record.mf_indptr))

    # --- replicate the seeding step's query selection, to recover its cluster ids --------------
    code = logit_code_for_threshold(mask_threshold)
    keep = (record.mf_logit_u8 >= code) & (record.mf_valid_prob[all_rows] >= object_threshold)
    if not keep.any():
        return None
    rows, cols, codes = all_rows[keep], record.mf_indices[keep], record.mf_logit_u8[keep]
    order = np.lexsort((codes, cols))
    rows, cols = rows[order], cols[order]
    last = np.empty(cols.size, dtype=bool)
    last[-1] = True
    last[:-1] = cols[1:] != cols[:-1]
    rows, cols = rows[last], cols[last]
    if min_cluster_hits > 1:
        counts = np.bincount(rows, minlength=n_q)
        big = counts[rows] >= min_cluster_hits
        rows = rows[big]
        if rows.size == 0:
            return None
    used = np.unique(rows)
    query_to_label = np.full(n_q, -1, dtype=np.int64)
    query_to_label[used] = np.arange(used.size)

    # --- the soft claims, restricted to those same queries -------------------------------------
    soft_keep = (record.mf_logit_u8 >= logit_code_for_threshold(floor)) & (
        record.mf_valid_prob[all_rows] >= object_threshold
    )
    if not soft_keep.any():
        return None
    labels = query_to_label[all_rows[soft_keep]]
    ok = labels >= 0
    if not ok.any():
        return None
    prob = probability_for_logit_code(record.mf_logit_u8[soft_keep][ok])
    cells = record.mf_indices[soft_keep][ok]
    return sp.csr_matrix((prob, (cells, labels[ok])), shape=(record.n_hits, used.size))


def chain_labels(
    record,
    base_label: np.ndarray,
    n_base: int,
    link_distance: float = 0.03,
    max_rounds: int = 32,
    min_cluster_hits: int = 1,
    density_guided: bool = False,
    density_radius: float | None = None,
    use_soft_masks: bool = False,
    soft_floor: float = 0.02,
    object_threshold: float = 0.2,
    affinity=None,
) -> tuple[np.ndarray, int]:
    """Grow ``base_label`` outwards into unclaimed cells by single-linkage chaining.

    Args:
        record: an :class:`~src.io.event_store.EventRecord`.
        base_label: per cell, the seed cluster index, -1 where unclaimed.
        n_base: number of seed clusters.
        link_distance: two cells are neighbours if they are within this many METRES of each
            other in 3D. This is a cell-to-cell linking scale, not a cluster radius: chaining
            reaches far by taking many small steps, so it should sit near the cell spacing.
        max_rounds: cap on growth rounds. Each round advances the frontier by one link, so this
            bounds how far a cluster can reach at ``max_rounds * link_distance``. It exists to stop
            a runaway in a dense event, not as a tuning knob.
        min_cluster_hits: clusters smaller than this are dropped and their cells returned to -1,
            matching the convention in :mod:`src.io.event_store`.
        density_guided: only link a cell to a neighbour at least as dense as itself, so growth
            follows the energy gradient outwards from cores instead of spreading isotropically.
            This is CLUE's nearest-denser-neighbour rule, applied to the network's seeds.
        density_radius: radius in metres for the local density. Defaults to ``link_distance``.
        use_soft_masks: when several clusters compete for a cell, award it to the one the MASK
            HEAD gives the highest probability, falling back to geometry where the model has no
            opinion. Costs nothing at inference -- the probabilities are already in the store.
        soft_floor: lowest mask probability to read. The store keeps 0.02; below that there is
            nothing recorded.
        object_threshold: object-head cut applied when reading the soft masks, so the same
            queries are eligible here as in the seeding step.
        affinity: optional callable ``(cell_idx, neighbour_idx) -> score`` over parallel index
            arrays, higher meaning "more likely the same particle". When given it REPLACES the
            nearest-neighbour tie-break: among the claimed neighbours in range, a cell joins the
            one scoring highest rather than the one that is closest.

            This is the crudest decision in the pipeline and it governs most of the misassigned
            energy -- 48.6% of assigned cells end up in the wrong cluster, and 82.1% of those have a
            single unambiguous owner, so they are wrong rather than ambiguous. Distance is a weak
            discriminator here: a learned classifier over geometry alone reaches AUC 0.742 on cell
            pairs, and adding the encoder's embeddings takes it to 0.817, against 0.670 for raw
            distance. Passing that classifier in is the point of this argument.

    Returns:
        ``(label, n_clusters)`` in the same form as the store's own labellers, so this drops
        straight into ``scripts.score``.
    """
    label = np.asarray(base_label).copy()
    if n_base == 0 or label.size == 0:
        return label, n_base

    xyz = np.column_stack([record.x, record.y, record.z]).astype(np.float64)

    # One tree for the whole event. `query_ball_point` on just the frontier each round would be
    # cheaper in principle, but the frontier is most of the event by round three and rebuilding a
    # tree per round costs more than querying a static one.
    tree = cKDTree(xyz)

    rho = None
    if density_guided:
        rho = _local_density(xyz, record.energy_calib, density_radius or link_distance, tree)

    soft = (
        _soft_claim_matrix(record, 0.5, object_threshold, min_cluster_hits, soft_floor)
        if use_soft_masks else None
    )

    for _ in range(max_rounds):
        unclaimed = np.flatnonzero(label < 0)
        if unclaimed.size == 0:
            break

        # Nearest claimed cell to each unclaimed cell. Query k neighbours and take the first that
        # is claimed: k=8 is enough that a cell surrounded by unclaimed cells simply waits for a
        # later round rather than being resolved wrongly now.
        dist, idx = tree.query(xyz[unclaimed], k=8, distance_upper_bound=link_distance)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)

        # scipy marks "no neighbour within the bound" with index == n and distance == inf.
        valid = np.isfinite(dist) & (idx < len(xyz))
        neighbour_label = np.where(valid, label[np.clip(idx, 0, len(xyz) - 1)], -1)
        claimed = neighbour_label >= 0
        if not claimed.any():
            break

        if rho is not None:
            # Grow from dense to sparse only: a cell may join a neighbour at least as dense as
            # itself, never one that is thinner. Without this, growth spreads isotropically into
            # the gaps between showers and the two sides meet wherever they happen to collide.
            denser = np.zeros_like(claimed)
            np.copyto(denser, rho[np.clip(idx, 0, len(xyz) - 1)] >= rho[unclaimed][:, None], where=valid)
            claimed = claimed & denser
            if not claimed.any():
                break

        # Among the claimed neighbours of each cell, take the closest one. Masking the distance of
        # unclaimed neighbours to inf makes argmin pick the right column.
        masked = np.where(claimed, dist, np.inf)
        best = masked.argmin(axis=1)
        rows = np.arange(len(unclaimed))

        if affinity is not None:
            # Score every (cell, claimed neighbour) candidate and take the argmax instead. Scoring
            # only the candidates -- rather than all pairs -- keeps this the same cost as the
            # distance version plus one model call per round.
            cand_cell = np.repeat(unclaimed, idx.shape[1])
            cand_nb = np.clip(idx, 0, len(xyz) - 1).ravel()
            flat_ok = claimed.ravel()
            score = np.full(flat_ok.shape, -np.inf)
            if flat_ok.any():
                score[flat_ok] = affinity(cand_cell[flat_ok], cand_nb[flat_ok])
            best = score.reshape(claimed.shape).argmax(axis=1)

        if soft is not None:
            # The model's own opinion, where it has one. For each (cell, candidate cluster) pair
            # look up the mask probability and take the argmax instead of the nearest. Cells with
            # no recorded probability for any candidate score 0 everywhere and fall through to the
            # geometric choice below, so this only ever REPLACES an arbitrary tie-break.
            cand = np.where(claimed, neighbour_label, -1)
            flat_rows = np.repeat(unclaimed, cand.shape[1])
            flat_cols = cand.ravel()
            ok = flat_cols >= 0
            probs = np.zeros(flat_cols.shape)
            if ok.any():
                probs[ok] = np.asarray(soft[flat_rows[ok], flat_cols[ok]]).ravel()
            probs = probs.reshape(cand.shape)
            has_opinion = probs.max(axis=1) > 0
            soft_pick = neighbour_label[rows, probs.argmax(axis=1)]
            geom_pick = neighbour_label[rows, best]
            chosen = np.where(has_opinion, soft_pick, geom_pick)
        else:
            chosen = neighbour_label[rows, best]

        chosen = np.where(claimed.any(axis=1), chosen, -1)
        grew = chosen >= 0
        if not grew.any():
            break

        # Assign the whole round at once. Doing it simultaneously rather than cell by cell keeps
        # the result independent of cell ordering, which matters because the store's cell order is
        # a lexsort on (subsystem, layer, phi) and would otherwise bias growth in phi.
        label[unclaimed[grew]] = chosen[grew]

    if min_cluster_hits > 1:
        counts = np.bincount(label[label >= 0], minlength=n_base)
        too_small = np.flatnonzero(counts < min_cluster_hits)
        if too_small.size:
            label[np.isin(label, too_small)] = -1

    return label, n_base
