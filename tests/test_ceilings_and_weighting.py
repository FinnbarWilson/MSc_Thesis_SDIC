"""The resolution reference, the match floor and the split/merge weighting.

Each of the three alters a published number, so each gets a test that would fail if it were
silently reverted: the match floor is relative and symmetric, the hit-counted and
energy-weighted split/merge definitions must be able to disagree, and the resolution reference
must behave like a ceiling, with efficiency near 1 and purity below it.
"""


import numpy as np
import pytest

from src.evaluation.differential import binned_fraction
from src.evaluation.matching import contamination, fragmentation, hungarian_match
from src.evaluation.metrics import score_event
from src.evaluation.oracle import particle_geometry, resolution_labels, unresolvable_groups
from tests.conftest import open_smoke_store

_store = open_smoke_store


# --- the match floor --------------------------------------------------------


def test_absolute_floor_admits_a_negligible_graze():
    """The behaviour being fixed: without a relative floor, one shared cell is a match."""
    overlap = np.array([[100.0, 0.001]])
    match = hungarian_match(overlap, min_overlap=0.0)
    assert match.n_matched == 1


def test_relative_floor_rejects_a_graze_but_keeps_containment():
    # A cluster holding 0.1% of a particle it barely touches is not that particle's cluster.
    truth_total = np.array([100.0])
    pred_total = np.array([50.0])
    grazed = hungarian_match(
        np.array([[0.05]]), truth_total=truth_total, pred_total=pred_total, min_overlap_frac=0.05
    )
    assert grazed.n_matched == 0
    assert grazed.unmatched_truth.tolist() == [0]
    assert grazed.unmatched_pred.tolist() == [0]

    # A small cluster sitting wholly inside a large particle must still match: it is 100% of
    # the cluster even though it is only 5% of the particle. This is why the floor is taken
    # against the smaller of the two totals rather than against the truth total.
    contained = hungarian_match(
        np.array([[5.0]]), truth_total=np.array([100.0]), pred_total=np.array([5.0]), min_overlap_frac=0.05
    )
    assert contained.n_matched == 1


def test_relative_floor_requires_both_totals():
    with pytest.raises(ValueError, match="min_overlap_frac requires"):
        hungarian_match(np.array([[1.0]]), min_overlap_frac=0.1)


# --- split and merge weighting ---------------------------------------------


def test_hit_and_energy_weighting_disagree_on_a_tail_graze():
    """The case the change exists for: many cells, almost no energy.

    Particle 0 has 100 cells. Cluster 1 takes 20 of them, comfortably over the 10% hit
    threshold, but they are tail cells holding 1% of its energy. By hits that is a split
    and a merge; by energy it is neither, and energy is the weighting reported.
    """
    overlap_n = np.array([[80.0, 20.0]])
    total_n = np.array([100.0])
    overlap_e = np.array([[99.0, 1.0]])
    total_e = np.array([100.0])

    assert fragmentation(overlap_n, total_n, 0.10).tolist() == [2]
    assert fragmentation(overlap_e, total_e, 0.10).tolist() == [1]
    assert contamination(overlap_n, total_n, 0.10).tolist() == [1, 1]
    assert contamination(overlap_e, total_e, 0.10).tolist() == [1, 0]


def test_both_weightings_are_written_to_the_tables():
    store = _store()
    record = store[0]
    particles, clusters, _ = score_event(record, record.truth_label, record.n_particles, algo="truth")
    for column in ("n_frag", "n_frag_e", "is_split", "is_split_e"):
        assert column in particles
    for column in ("n_owners", "n_owners_e", "is_merge", "is_merge_e"):
        assert column in clusters
    # Perfect clustering splits and merges nothing under either definition.
    assert not particles["is_split_e"].any()
    assert not clusters["is_merge_e"].any()


def test_response_matched_is_bounded_but_response_is_not():
    """A merged cluster reports response > 1; the matched-only form cannot."""
    store = _store()
    record = store[0]
    everything = np.zeros(record.n_hits, dtype=np.int32)
    particles, _, _ = score_event(record, everything, 1, algo="blob")

    matched = particles[particles["matched"]]
    assert float(matched["response"].iloc[0]) > 1.0
    assert float(matched["response_matched"].iloc[0]) == pytest.approx(1.0, abs=1e-9)


# --- the reference clusterings ---------------------------------------------


def test_resolution_reference_recovers_almost_everything():
    """Efficiency is ~1 by construction, which is why its PURITY is the number to read."""
    store = _store()
    for record in store:
        label, n = resolution_labels(record, fraction=0.5)
        particles, clusters, _ = score_event(record, label, n, algo="oracle_resolution")
        assert particles["eff_e"].mean() > 0.95
        # And its purity is well below 1, because sub-threshold deposits still land in it.
        assert clusters["pur_e"].mean() < 1.0


def test_resolution_grouping_is_monotonic_in_the_threshold():
    """A stricter overlap requirement can only merge fewer particles."""
    store = _store()
    record = store[0]
    loose = np.unique(unresolvable_groups(record, fraction=0.2)).size
    strict = np.unique(unresolvable_groups(record, fraction=0.8)).size
    assert loose <= strict <= record.n_particles


def test_resolution_grouping_never_drops_a_particle():
    store = _store()
    for record in store:
        groups = unresolvable_groups(record, fraction=0.5)
        assert groups.size == record.n_particles
        assert groups.min() >= 0


def test_particle_geometry_directions_are_unit_vectors():
    store = _store()
    geometry = particle_geometry(store[0])
    norms = np.linalg.norm(geometry.direction, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_resolution_labels_assign_every_cell():
    store = _store()
    record = store[0]
    label, n = resolution_labels(record, fraction=0.5)
    assert (label >= 0).all()
    assert label.max() < n


# --- error bars -------------------------------------------------------------


def test_event_clustered_interval_is_wider_than_the_binomial():
    """Correlated particles do not give binomial errors.

    Every particle in an event is given the same outcome here, which is the extreme of the
    correlation that really exists. The binomial interval sees 400 independent trials and
    shrinks to nothing; the clustered one sees 4 events and does not.
    """
    rng = np.random.default_rng(0)
    events = np.repeat(np.arange(4), 100)
    outcome = np.repeat(np.array([True, True, True, False]), 100)
    values = rng.uniform(1.1, 9.9, size=events.size)
    bins = np.array([1.0, 10.0])

    binomial = binned_fraction(values, outcome, bins)
    clustered = binned_fraction(values, outcome, bins, cluster=events)

    assert binomial.value[0] == pytest.approx(0.75)
    assert clustered.value[0] == pytest.approx(0.75)
    binomial_width = binomial.high[0] - binomial.low[0]
    clustered_width = clustered.high[0] - clustered.low[0]
    assert clustered_width > 3 * binomial_width


def test_event_clustered_interval_falls_back_when_events_are_too_few():
    """Two events cannot be bootstrapped; a binomial bar beats a zero-width one."""
    events = np.repeat(np.arange(2), 50)
    outcome = np.repeat(np.array([True, False]), 50)
    values = np.full(events.size, 5.0)
    bins = np.array([1.0, 10.0])

    result = binned_fraction(values, outcome, bins, cluster=events)
    assert result.high[0] > result.low[0]
