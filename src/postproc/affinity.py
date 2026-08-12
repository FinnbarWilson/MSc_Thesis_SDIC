"""A learned cell-pair affinity over encoder embeddings, for use as chaining's tie-break.

WHY THIS BEATS WHAT CHAINING DOES NOW
-------------------------------------
`chain.py` decides which cluster an unclaimed cell joins by taking its nearest already-claimed
neighbour. That is the crudest decision in the pipeline and it governs most of the misassigned
energy: 48.6% of assigned cells end up in the wrong cluster, and 82.1% of those cells have a single
unambiguous contributing particle, so they are wrong rather than genuinely contested.

Measured on cell pairs within 0.06 m, fitted on the tune window and scored on the eval window:

    plain 3D distance, no model      AUC 0.670
    raw embedding cosine, no model       0.671
    learned, geometry only               0.742
    learned, embedding only              0.789
    learned, geometry + embedding        0.817

So the encoder's embeddings carry co-membership information worth **+0.075 AUC over geometry**, and
raw cosine -- which is what the encoder-affinity probe measured (deleted with dias/; see git history), and what led to the earlier
and wrong conclusion that the encoder knew nothing -- is no better than a ruler. Cosine over 256
dimensions is dominated by variance unrelated to co-membership; a learned readout of the same
vectors is not.

WHERE THE EMBEDDINGS COME FROM
------------------------------
A sidecar directory of per-event `.npz` files written by the embedding extractor (deleted with dias/; see git history), NOT the event
store. The store has a format version and a contract checked on load, and adding a 5.5 GB array to
it to test a hypothesis would mean bumping that format and updating `src/io/event_store.py` before
knowing whether the hypothesis holds. If this earns its place, moving it into the store is the right
follow-up; until then a sidecar keeps every existing store valid.

THE CAVEAT THAT KILLED THE LAST MODEL LIKE THIS
-----------------------------------------------
`src/postproc/attribute.py` reached AUC 0.765 and changed nothing end to end, because its dominant
feature was distance and it simply reproduced the rule chaining already applies. A better pair AUC
is not automatically a better clustering. What is different here is that this signal BEATS geometry
rather than matching it, so it can flip decisions -- but that has to be measured end to end, not
assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURES = (
    "d3d",         # metres between the two cells
    "d_eta",
    "d_phi",
    "d_r",         # depth separation, kept apart from the angular terms: showers are long and narrow
    "log_e_i",
    "log_e_j",
    "e_ratio",
    "cos_emb",     # cosine of the encoder embeddings
    "l2_emb",
    "norm_i",      # embedding norms carry how confident the encoder is about each cell
    "norm_j",
)


class EmbeddingCache:
    """Per-event encoder embeddings, loaded on demand from the sidecar directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._event: int | None = None
        self._data: dict | None = None

    def get(self, sample_id: int) -> dict | None:
        if self._event != sample_id:
            path = self.root / f"event_{sample_id:06d}.npz"
            if not path.exists():
                return None
            d = np.load(path)
            self._data = {"embed": d["embed"].astype(np.float32), "xyz": d["xyz"], "energy": d["energy"]}
            self._event = sample_id
        return self._data


def pair_features(record, cache: EmbeddingCache, i: np.ndarray, j: np.ndarray) -> np.ndarray | None:
    """Features for parallel arrays of cell indices, in the order of :data:`FEATURES`."""
    data = cache.get(int(record.sample_id))
    if data is None or i.size == 0:
        return None
    embed = data["embed"]
    if embed.shape[0] != record.n_hits:
        return None

    xyz = np.column_stack([record.x, record.y, record.z])
    energy = np.asarray(record.energy_calib, dtype=np.float64)
    eta, phi, r = record.eta(), record.phi(), record.r()

    d3d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    dphi = np.arctan2(np.sin(phi[i] - phi[j]), np.cos(phi[i] - phi[j]))
    ei, ej = embed[i], embed[j]
    ni, nj = np.linalg.norm(ei, axis=1), np.linalg.norm(ej, axis=1)
    cos = (ei * ej).sum(1) / np.maximum(ni * nj, 1e-9)

    hi = np.maximum(energy[i], energy[j])
    lo = np.minimum(energy[i], energy[j])
    return np.column_stack([
        d3d, np.abs(eta[i] - eta[j]), np.abs(dphi), np.abs(r[i] - r[j]),
        np.log10(np.maximum(energy[i], 1e-12)), np.log10(np.maximum(energy[j], 1e-12)),
        lo / np.maximum(hi, 1e-12),
        cos, np.linalg.norm(ei - ej, axis=1), ni, nj,
    ])


def pair_truth(record, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """1 where both cells have the same exclusive truth owner."""
    tl = np.asarray(record.truth_label)
    return ((tl[i] >= 0) & (tl[i] == tl[j])).astype(int)


def sample_training_pairs(record, cache: EmbeddingCache, radius: float = 0.06,
                          anchors: int = 1500, k: int = 8, rng=None):
    """Local (cell, neighbour) pairs for fitting, biased the way the CHAINER will see them.

    k-nearest rather than all-pairs-in-radius on purpose. The decision this model is being fitted
    for is "given an unclaimed cell, which of its nearest claimed neighbours does it join", so the
    nearest few neighbours ARE the operational population. Fitting on all pairs within the radius
    would train it on a question nobody asks.
    """
    from scipy.spatial import cKDTree

    rng = rng or np.random.default_rng(0)
    xyz = np.column_stack([record.x, record.y, record.z])
    n = len(xyz)
    if n < 50:
        return None
    tree = cKDTree(xyz)
    a = rng.choice(n, min(anchors, n), replace=False)
    dist, idx = tree.query(xyz[a], k=min(k + 1, n), distance_upper_bound=radius)
    ok = np.isfinite(dist) & (idx < n)
    i = np.repeat(a, idx.shape[1])[ok.ravel()]
    j = idx.ravel()[ok.ravel()]
    keep = i != j
    i, j = i[keep], j[keep]
    if i.size == 0:
        return None
    x = pair_features(record, cache, i, j)
    if x is None:
        return None
    return x, pair_truth(record, i, j)


def make_affinity_fn(model, record, cache: EmbeddingCache):
    """Bind a fitted model to one event, giving the callable ``chain_labels(affinity=...)`` wants.

    Returns None when the event has no embeddings, so a caller can fall back to plain geometric
    chaining rather than silently scoring everything the same.
    """
    if cache.get(int(record.sample_id)) is None:
        return None

    def score(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        x = pair_features(record, cache, i, j)
        if x is None:
            return np.zeros(len(i))
        return model.predict_proba(x)[:, 1]

    return score
