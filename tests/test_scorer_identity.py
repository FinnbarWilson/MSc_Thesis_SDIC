"""Self-consistency of the scorer: the truth fed back in as a prediction.

Efficiency must come out at exactly 1, every particle must match exactly one cluster, and
nothing may be reported as split, merged or fake. Anything else is an indexing bug in the
matcher or the overlap matrix, and would be invisible in a real run where every number is
legitimately below 1.

Energy purity is asserted to be *below* 1 on purpose: a cell owned by one particle still holds
energy from others, including particles below the pT cut, and that contamination counts against
purity by design.
"""


import numpy as np
import pytest

from src.evaluation.matching import contamination, fragmentation, hungarian_match, overlap_matrix
from src.evaluation.metrics import local_density, score_event
from tests.conftest import open_smoke_store

_store = open_smoke_store


def test_overlap_matrix_counts_shared_cells():
    truth = np.array([0, 0, 1, 1, 1, -1, 2])
    pred = np.array([0, 1, 1, 1, -1, 0, -1])
    counts = overlap_matrix(truth, pred, None, 3, 2)
    assert counts.tolist() == [[1, 1], [0, 2], [0, 0]]

    weights = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    energy = overlap_matrix(truth, pred, weights, 3, 2)
    assert energy.tolist() == [[1.0, 2.0], [0.0, 12.0], [0.0, 0.0]]


def test_hungarian_never_invents_a_disjoint_match():
    # Two truth particles, two clusters, but only one pair shares anything. A global
    # optimum would happily pair the other two to complete the assignment.
    overlap = np.array([[5.0, 0.0], [0.0, 0.0]])
    match = hungarian_match(overlap)
    assert match.n_matched == 1
    assert match.truth_index.tolist() == [0]
    assert match.pred_index.tolist() == [0]
    assert match.unmatched_truth.tolist() == [1]
    assert match.unmatched_pred.tolist() == [1]


def test_hungarian_prefers_the_globally_best_assignment():
    # Greedy-by-largest would give cluster 0 to particle 0 (9 > 8) and leave particle 1
    # with 1, totalling 10. The optimum is 8 + 7 = 15.
    overlap = np.array([[9.0, 8.0], [7.0, 0.0]])
    match = hungarian_match(overlap)
    pairs = dict(zip(match.truth_index.tolist(), match.pred_index.tolist()))
    assert pairs == {0: 1, 1: 0}


def test_split_and_merge_counting():
    # particle 0 spread over both clusters; particle 1 entirely in cluster 1.
    overlap = np.array([[5.0, 5.0], [0.0, 10.0]])
    totals = np.array([10.0, 10.0])
    assert fragmentation(overlap, totals, 0.10).tolist() == [2, 1]
    assert contamination(overlap, totals, 0.10).tolist() == [1, 2]


def test_local_density_ignores_self_and_wraps_phi():
    eta = np.array([0.0, 0.0, 3.0])
    phi = np.array([np.pi - 0.05, -np.pi + 0.05, 0.0])
    dr_min, n_within = local_density(eta, phi, radius=0.2)
    # The first two are 0.1 apart across the +/-pi boundary, not 2*pi - 0.1.
    assert dr_min[0] == pytest.approx(0.1, abs=1e-6)
    assert n_within.tolist() == [1, 1, 0]


def test_truth_scored_against_itself_is_perfect():
    store = _store()
    for record in store:
        n = record.n_particles
        particles, clusters, event = score_event(record, record.truth_label, n, algo="truth")

        # A handful of target particles (~0.4%) own no cell exclusively: every cell they
        # touched was dominated by someone else. They pass the >=3-calo-hit cut on the
        # multi-owner truth but have nothing of their own to find, so no exclusive-partition
        # algorithm can ever recover them. They are a ceiling on efficiency, and the scorer
        # is right to report them as unmatched rather than to quietly drop them.
        empty = particles["n_hits"].to_numpy() == 0
        assert empty.sum() < 0.02 * n, "unexpectedly many particles own no cell exclusively"
        assert event["n_matched"] == n - int(empty.sum())
        assert event["n_unmatched_truth"] == int(empty.sum())
        assert event["n_unmatched_pred"] == int(empty.sum())

        real = particles[~empty]
        assert real["matched"].all()
        assert (real["cluster_row"].to_numpy() == np.flatnonzero(~empty)).all()

        np.testing.assert_allclose(real["eff_n"], 1.0, atol=1e-12)
        np.testing.assert_allclose(real["eff_e"], 1.0, atol=1e-9)
        np.testing.assert_allclose(real["pur_n"], 1.0, atol=1e-12)

        assert not particles["is_split"].any(), "perfect clustering cannot split anything"
        assert not clusters["is_merge"].any(), "perfect clustering cannot merge anything"
        np.testing.assert_allclose(real["frag_frac"], 0.0, atol=1e-9)
        np.testing.assert_allclose(particles["e_lost_noise"], 0.0, atol=1e-9)
        np.testing.assert_allclose(particles["e_lost_other"], 0.0, atol=1e-9)

        # Energy purity is below 1 by construction: owned cells still hold energy from
        # other particles, mostly ones below the pT cut. The tolerance is 1e-6 rather than
        # exact because incidence is stored as float32, so a sole owner's share normalises
        # to 1 +/- 1e-7 and the ratio can land just above one.
        pur_e = particles["pur_e"].to_numpy()
        assert (pur_e <= 1.0 + 1e-6).all()
        assert pur_e.mean() < 1.0


def test_scoring_is_invariant_under_relabelling():
    """Cluster ids are arbitrary; permuting them must not change any metric."""
    store = _store()
    record = store[0]
    n = record.n_particles

    rng = np.random.default_rng(0)
    permutation = rng.permutation(n)
    relabelled = np.where(record.truth_label >= 0, permutation[record.truth_label], -1)

    base, _, _ = score_event(record, record.truth_label, n, algo="truth")
    moved, _, _ = score_event(record, relabelled, n, algo="truth")

    for column in ("eff_n", "eff_e", "pur_n", "pur_e", "n_frag"):
        np.testing.assert_allclose(base[column], moved[column], atol=1e-9, err_msg=column)


def test_single_giant_cluster_is_pure_merge():
    """Everything in one cluster: efficiency high for one particle, merging everywhere."""
    store = _store()
    record = store[0]
    everything = np.zeros(record.n_hits, dtype=np.int32)
    particles, clusters, event = score_event(record, everything, 1, algo="blob")

    assert event["n_matched"] == 1, "one cluster can only match one particle"
    assert event["n_unmatched_truth"] == record.n_particles - 1
    assert clusters["is_merge"].all(), "a single cluster over the whole event has merged everything"
    assert clusters["n_owners"].iloc[0] > 1
    # The one matched particle keeps all its cells, so its efficiency is 1 ...
    matched = particles[particles["matched"]]
    np.testing.assert_allclose(matched["eff_e"], 1.0, atol=1e-9)
    # ... but its purity is tiny, since the cluster is the whole calorimeter.
    assert float(matched["pur_e"].iloc[0]) < 0.5


def test_every_cell_its_own_cluster_is_pure_split():
    """The opposite degenerate case: maximal purity, minimal efficiency, maximal splitting."""
    store = _store()
    record = store[0]
    singletons = np.arange(record.n_hits, dtype=np.int32)
    particles, clusters, _ = score_event(record, singletons, record.n_hits, algo="atomise")

    multi_cell = particles[particles["n_hits"] > 1]
    # frag_frac sees every one of them: nothing keeps more than one cell of itself.
    assert multi_cell["frag_frac"].min() > 0.0
    assert multi_cell["frag_frac"].mean() > 0.5  # energy-weighted: the hottest single cell still holds a sizeable share
    assert (multi_cell["n_touched"] == multi_cell["n_hits"]).all()

    # ... whereas the brief's n_frag-based split flag misses most of them, because a
    # particle spread over more than ten clusters gives none of them the required 10%.
    # This documents the blind spot rather than asserting the metric is wrong.
    big = multi_cell[multi_cell["n_hits"] > 10]
    assert (big["n_frag"] == 0).all()
    assert not big["is_split"].any()

    # A single-cell cluster can only hold one particle's worth of >=10%.
    assert not clusters["is_merge"].any()
