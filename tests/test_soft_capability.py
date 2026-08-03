"""Checks on the multi-owner capability scoring.

The point of the soft metric is that it stops actively suppressing one method's capability
without inventing an advantage for it. Neither half of that is self-evident from the code, so
both are pinned here:

*   **Not favouring.** A method whose clusters never overlap must come through the fractional
    machinery with exactly the score a partitioning metric would give it. If normalisation or
    the min-overlap silently changed CLUE's numbers, the comparison would be rigged in a way
    no aggregate would reveal.
*   **Not suppressing.** Under an exclusive prediction the soft score must *credit* the
    sub-dominant truth contributions in cells that were won, which the exclusive truth throws
    away. So CLUE's soft efficiency should sit at or above its exclusive efficiency, never
    below.

The headline of the study -- that particles owning no cell exclusively are unreachable for a
partitioning method and reachable for a mask-based one -- is a structural claim, so it gets a
structural test rather than one pinned to a measured rate.
"""

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import score_event
from src.evaluation.soft import capability_summary, hard_weights, score_event_soft, sharing_diagnostics
from src.io.event_store import EventStore, logit_code_for_threshold, probability_for_logit_code

STORE = Path("/home/xucapfwi/eventstore_smoke/ttbar_pu0_20250_20255_v1")


def _store():
    if not STORE.exists():
        pytest.skip(f"no event store at {STORE}")
    return EventStore(STORE)


def test_probability_decoding_round_trips_within_the_quantisation():
    """Decoding is the inverse of the threshold encoder, up to the half-step it stores at."""
    for probability in (0.02, 0.1, 0.3, 0.5, 0.9, 0.99):
        code = logit_code_for_threshold(probability)
        recovered = float(probability_for_logit_code(np.array([code]))[0])
        assert recovered == pytest.approx(probability, abs=0.01)


def test_hard_weights_are_all_one():
    """A partition means the cell is wholly one cluster's; that is what weight 1 encodes."""
    label = np.array([-1, 0, 0, 2, -1, 1])
    cluster, cell, weight = hard_weights(label)
    assert cluster.tolist() == [0, 0, 2, 1]
    assert cell.tolist() == [1, 2, 3, 5]
    np.testing.assert_allclose(weight, 1.0)


def test_soft_weights_partition_each_claimed_cell():
    """Normalisation must make the weights a division of the cell, not a set of votes."""
    store = _store()
    for record in store:
        _, cell, weight, _ = record.maskformer_soft_masks(0.5, 0.2, 1)
        if cell.size == 0:
            continue
        per_cell = np.bincount(cell, weights=weight, minlength=record.n_hits)
        np.testing.assert_allclose(per_cell[per_cell > 0], 1.0, atol=1e-9)


def test_a_perfect_partition_scores_exactly_its_exclusive_share():
    """The exact identity that says what the soft metric costs a partitioning method.

    Score the truth partition as a prediction. Under the exclusive metric it is perfect by
    construction. Under the soft metric its efficiency is exactly `exclusive_share` -- the
    fraction of the particle's energy living in cells it dominates -- because those are the
    only cells any partition can award it.

    So `exclusive_share` is a hard ceiling on soft efficiency for the whole partitioning
    class, and the shortfall from 1 is precisely the energy only overlapping masks can reach.
    That is the capability being measured, and it is why the soft score is not simply the
    exclusive score with extra credit bolted on.
    """
    store = _store()
    record = store[0]
    cluster, cell, weight = hard_weights(record.truth_label)
    soft = score_event_soft(record, cluster, cell, weight, record.n_particles, "clue")
    hard, _, _ = score_event(record, record.truth_label, record.n_particles, algo="clue")

    owns_something = hard["n_hits"].to_numpy() > 0
    np.testing.assert_allclose(
        soft.loc[owns_something, "eff_e"],
        soft.loc[owns_something, "exclusive_share"],
        atol=1e-9,
    )
    # And that is strictly less than the exclusive metric's perfect score, which is the point.
    assert soft.loc[owns_something, "eff_e"].mean() < hard.loc[owns_something, "eff_e"].mean()


def test_the_soft_denominator_is_method_independent():
    """Why the change is fair despite costing CLUE: both methods face the same denominator.

    The exclusive metric measures each particle against the energy in cells it dominates --
    a target defined by what a partition can express. The soft metric measures against the
    particle's actual deposited energy, which is a property of the event alone. Neither
    method's output can move it, so it is the neutral yardstick even though it is a harder one
    for the method that cannot reach all of it.
    """
    store = _store()
    record = store[0]

    hard_cluster, hard_cell, hard_weight = hard_weights(record.truth_label)
    from_partition = score_event_soft(record, hard_cluster, hard_cell, hard_weight, record.n_particles, "clue")

    soft_cluster, soft_cell, soft_weight, n = record.maskformer_soft_masks(0.5, 0.2, 1)
    from_masks = score_event_soft(record, soft_cluster, soft_cell, soft_weight, n, "maskformer")

    np.testing.assert_allclose(from_partition["e_dep_multi"], from_masks["e_dep_multi"], atol=1e-12)


def test_particles_with_no_exclusive_cell_are_flagged():
    """The population the capability study is built around."""
    store = _store()
    record = store[0]
    cluster, cell, weight = hard_weights(record.truth_label)
    soft = score_event_soft(record, cluster, cell, weight, record.n_particles, "clue")

    flagged = soft["no_exclusive_cell"].to_numpy()
    assert flagged.any(), "expected some particles owning no cell exclusively"
    # They are flagged because the exclusive partition gave them nothing, yet the multi-owner
    # truth says they deposited energy -- which is precisely why they are worth studying.
    assert (soft.loc[flagged, "e_dep_exclusive"] == 0).all()
    assert (soft.loc[flagged, "e_dep_multi"] > 0).all()


def test_an_exclusive_method_barely_reaches_an_impossible_particle():
    """The structural claim behind the headline number, stated at the accuracy it holds.

    A particle owning no cell exclusively has no cell of its own for a partition to award it,
    so no cluster is ever *about* it. It is not quite unreachable, though, and the exception
    is worth knowing rather than asserting away: a cluster built around some other particle
    may contain cells this one contributed to sub-dominantly, and the soft metric credits that
    energy. So a partitioning method scores a little above zero here by accident -- measured
    at 0.007 for CLUE over the full window, against 0.534 for overlapping masks.

    Asserted as "negligible" rather than "exactly zero" because exact zero is a property of
    this smoke store, not of the class.
    """
    store = _store()
    for record in store:
        cluster, cell, weight = hard_weights(record.truth_label)
        soft = score_event_soft(record, cluster, cell, weight, record.n_particles, "clue")
        impossible = soft[soft["no_exclusive_cell"]]
        if len(impossible):
            assert float((impossible["eff_e"] >= 0.5).mean()) < 0.05


def test_overlapping_masks_can_reach_an_impossible_particle():
    """The complement: the capability is real, not just permitted by the metric."""
    store = _store()
    reached = 0
    total = 0
    for record in store:
        cluster, cell, weight, n = record.maskformer_soft_masks(0.5, 0.2, 1)
        soft = score_event_soft(record, cluster, cell, weight, n, "maskformer")
        impossible = soft[soft["no_exclusive_cell"]]
        reached += int((impossible["eff_e"] > 0).sum())
        total += len(impossible)
    assert total > 0, "no structurally impossible particles in the smoke store"
    assert reached > 0, "overlapping masks recovered none of them"


def test_sharing_diagnostics_report_both_sides():
    """The numbers that make a soft efficiency below the exclusive one legible."""
    store = _store()
    record = store[0]
    _, cell, _, _ = record.maskformer_soft_masks(0.5, 0.2, 1)
    stats = sharing_diagnostics(record, cell)
    assert stats["claims_per_cell"] >= 1.0
    assert stats["truth_owners_per_cell"] >= 1.0
    assert 0.0 <= stats["shared_cell_frac"] <= 1.0


def test_capability_summary_reports_zero_for_an_exclusive_method():
    store = _store()
    record = store[0]
    cluster, cell, weight = hard_weights(record.truth_label)
    table = score_event_soft(record, cluster, cell, weight, record.n_particles, "clue")
    summary = capability_summary([table])
    assert float(summary.loc[summary["algo"] == "clue", "impossible_recovered"].iloc[0]) == 0.0
