"""The incidence-head labellings, and the invariant that makes them a controlled comparison.

Reading the incidence head instead of the mask head is a change of *rule*, not of scope, and
the tests here are built around that. The one that matters most is
``test_detection_is_identical_to_the_mask_head``: both labellings must claim exactly the same
cells, because detection stays with the mask head in both. If that holds, any difference in
efficiency, purity, splitting or merging between ``maskformer`` and ``maskformer_incidence``
is the assignment rule alone, and the comparison means something. If it ever fails, the two
methods are also capturing different amounts of energy and the difference is uninterpretable.

The records here are built by hand rather than read from a store. The behaviour under test is
a handful of index manipulations whose failure modes are off-by-ones, and a six-cell event
with a known answer pins those far more tightly than 24k cells of real data can.
"""

import numpy as np
import pytest

from src.io.event_store import EventRecord, EventStoreMismatchError

CALIBRATION = np.array([37.5, 38.7, 45.0, 46.9], dtype=np.float32)


def make_record(
    n_hits: int,
    mask_indptr: list[int],
    mask_indices: list[int],
    valid_prob: list[float],
    incidence_query: list[list[int]],
    incidence_share: list[list[float]],
    mask_codes: list[int] | None = None,
) -> EventRecord:
    """A minimal EventRecord carrying only what the labelling methods read.

    Every field the methods do not touch is filled with a zero of the right length, so a
    method that starts reading one fails loudly here rather than quietly using a plausible
    number.
    """
    n_q = len(valid_prob)
    nnz = len(mask_indices)
    zeros_h = np.zeros(n_hits, dtype=np.float32)
    return EventRecord(
        sample_id=0,
        x=zeros_h,
        y=zeros_h,
        z=zeros_h,
        energy=np.ones(n_hits, dtype=np.float32),
        detector=np.zeros(n_hits, dtype=np.uint8),
        subsystem=np.zeros(n_hits, dtype=np.uint8),
        layer=np.zeros(n_hits, dtype=np.uint8),
        truth_label=np.zeros(n_hits, dtype=np.int32),
        truth_indptr=np.zeros(1, dtype=np.int32),
        truth_indices=np.zeros(0, dtype=np.int32),
        truth_incidence=np.zeros(0, dtype=np.float32),
        particle_id=np.zeros(0, dtype=np.uint64),
        particle_px=np.zeros(0, dtype=np.float32),
        particle_py=np.zeros(0, dtype=np.float32),
        particle_pz=np.zeros(0, dtype=np.float32),
        particle_energy=np.zeros(0, dtype=np.float32),
        particle_pt=np.zeros(0, dtype=np.float32),
        particle_eta=np.zeros(0, dtype=np.float32),
        particle_phi=np.zeros(0, dtype=np.float32),
        particle_pdg_id=np.zeros(0, dtype=np.int32),
        particle_class=np.zeros(0, dtype=np.uint8),
        particle_num_calohits=np.zeros(0, dtype=np.int32),
        particle_energy_calo_sum=np.zeros(0, dtype=np.float32),
        mf_query_index=np.arange(n_q, dtype=np.int16),
        mf_valid_prob=np.array(valid_prob, dtype=np.float32),
        mf_indptr=np.array(mask_indptr, dtype=np.int32),
        mf_indices=np.array(mask_indices, dtype=np.int32),
        # 255 is the top logit code, i.e. mask probability ~1: these tests are about which
        # query wins a cell, not about where the mask threshold sits.
        mf_logit_u8=np.array(mask_codes if mask_codes is not None else [255] * nnz, dtype=np.uint8),
        mf_incidence_query=np.array(incidence_query, dtype=np.int16),
        mf_incidence_share=np.array(incidence_share, dtype=np.float16),
        n_hits=n_hits,
        n_particles=0,
        n_particles_untruncated=0,
        truncated=False,
        event_energy_raw=0.0,
        event_energy_calib=0.0,
        event_energy_on_target_calib=0.0,
        calibration=CALIBRATION,
    )


def fragmented_record() -> EventRecord:
    """One particle's six cells, split across two queries by the mask head.

    This is the failure the incidence head is being asked to fix, in miniature: the measured
    ``frag_frac`` of 0.55 says that on average 55% of a particle's energy sits outside its
    largest predicted piece, and it exceeds the geometric ceiling in every energy bin. Here
    the mask head gives cells 0-2 to query 0 and cells 3-5 to query 1, while the incidence
    head puts 0.8 of every cell on query 0.
    """
    return make_record(
        n_hits=6,
        mask_indptr=[0, 3, 6],
        mask_indices=[0, 1, 2, 3, 4, 5],
        valid_prob=[0.9, 0.9],
        incidence_query=[[0, 1]] * 6,
        incidence_share=[[0.8, 0.2]] * 6,
    )


def test_mask_head_fragments_and_incidence_head_does_not():
    record = fragmented_record()

    mask_label, mask_n = record.maskformer_labels(0.5, 0.2)
    assert mask_n == 2, "the mask head should split this particle, that is the premise"
    assert sorted(set(mask_label.tolist())) == [0, 1]

    inc_label, inc_n = record.maskformer_incidence_labels(0.5, 0.2)
    assert inc_n == 1
    assert inc_label.tolist() == [0] * 6


def test_detection_is_identical_to_the_mask_head():
    """The invariant the whole comparison rests on: same cells claimed, different owners.

    Checked on an event where the two rules genuinely disagree about ownership, so this
    cannot pass by the labellings being accidentally equal.
    """
    record = fragmented_record()
    mask_label, _ = record.maskformer_labels(0.5, 0.2)
    inc_label, _ = record.maskformer_incidence_labels(0.5, 0.2)

    assert np.array_equal(mask_label >= 0, inc_label >= 0)
    assert not np.array_equal(mask_label, inc_label)


def test_undetected_cells_stay_unclaimed():
    """A cell no query's mask reaches is not rescued by the incidence head.

    The incidence softmax sums to one over queries for every cell, including the ~63% of
    hits belonging to no target particle, so on its own it can never decline anything.
    Detection has to keep coming from the mask head or purity collapses.
    """
    record = make_record(
        n_hits=3,
        mask_indptr=[0, 1],
        mask_indices=[0],
        valid_prob=[0.9],
        incidence_query=[[0, -1]] * 3,
        incidence_share=[[1.0, 0.0]] * 3,
    )
    label, n = record.maskformer_incidence_labels(0.5, 0.2)
    assert n == 1
    assert label.tolist() == [0, -1, -1]


def test_object_threshold_excludes_a_rejected_query_from_winning():
    """A query the object head rejects cannot own a cell even with the largest share."""
    record = make_record(
        n_hits=2,
        mask_indptr=[0, 2, 4],
        mask_indices=[0, 1, 0, 1],
        valid_prob=[0.01, 0.9],
        incidence_query=[[0, 1]] * 2,
        incidence_share=[[0.9, 0.1]] * 2,
    )
    label, n = record.maskformer_incidence_labels(0.5, 0.2)
    assert n == 1
    # Query 0 holds the larger share but sits below the object threshold, so query 1 wins.
    assert label.tolist() == [0, 0]


def test_incidence_floor_leaves_contested_cells_unclaimed():
    """The floor is a working point: a cell nobody clearly owns can be declined."""
    record = make_record(
        n_hits=2,
        mask_indptr=[0, 2, 4],
        mask_indices=[0, 1, 0, 1],
        valid_prob=[0.9, 0.9],
        incidence_query=[[0, 1], [0, 1]],
        incidence_share=[[0.9, 0.1], [0.34, 0.33]],
    )
    assert record.maskformer_incidence_labels(0.5, 0.2, incidence_floor=0.0)[0].tolist() == [0, 0]
    kept, _ = record.maskformer_incidence_labels(0.5, 0.2, incidence_floor=0.5)
    assert kept.tolist() == [0, -1]


def test_padding_never_wins_a_cell():
    """-1 padding, written where an event keeps fewer queries than k, is not a cluster id."""
    record = make_record(
        n_hits=2,
        mask_indptr=[0, 2],
        mask_indices=[0, 1],
        valid_prob=[0.9],
        incidence_query=[[0, -1], [0, -1]],
        incidence_share=[[0.7, 0.0], [0.7, 0.0]],
    )
    label, n = record.maskformer_incidence_labels(0.5, 0.2)
    assert n == 1
    assert label.tolist() == [0, 0]


def test_soft_weights_form_a_partition_of_each_cell():
    """Fractional claims must sum to 1 per cell, or energy is created or destroyed."""
    record = fragmented_record()
    cluster, cell, weight, n = record.maskformer_incidence_soft_masks(0.5, 0.2)
    assert n == 2
    totals = np.bincount(cell, weights=weight, minlength=record.n_hits)
    assert np.allclose(totals, 1.0)
    # 0.8 / (0.8 + 0.2) on every cell, which the store's raw shares only give after the
    # per-cell renormalisation this method applies.
    assert np.allclose(weight[cluster == 0], 0.8, atol=1e-3)


def test_soft_masks_only_divide_detected_cells():
    record = make_record(
        n_hits=3,
        mask_indptr=[0, 1, 2],
        mask_indices=[0, 0],
        valid_prob=[0.9, 0.9],
        incidence_query=[[0, 1]] * 3,
        incidence_share=[[0.6, 0.4]] * 3,
    )
    _, cell, weight, _ = record.maskformer_incidence_soft_masks(0.5, 0.2)
    assert set(cell.tolist()) == {0}
    assert np.isclose(weight.sum(), 1.0)


def test_a_store_without_the_head_raises_rather_than_falling_back():
    """Silently using the mask head here would report the two methods as identical."""
    record = make_record(
        n_hits=2,
        mask_indptr=[0, 2],
        mask_indices=[0, 1],
        valid_prob=[0.9],
        incidence_query=np.zeros((2, 0), dtype=np.int16).tolist(),
        incidence_share=np.zeros((2, 0), dtype=np.float16).tolist(),
    )
    assert not record.has_incidence
    with pytest.raises(EventStoreMismatchError, match="no incidence head"):
        record.maskformer_incidence_labels(0.5, 0.2)
    with pytest.raises(EventStoreMismatchError, match="no incidence head"):
        record.maskformer_incidence_soft_masks(0.5, 0.2)


def test_restrict_to_mask_control_keeps_the_winner_inside_the_mask():
    """The control path: the winner must also be a query whose mask claims the cell.

    On the fragmented record this reproduces the mask head's own split, because each query's
    mask covers only its own three cells -- which is exactly what makes it a control for
    "how much of the gain is reaching cells the mask head gave to nobody".
    """
    record = fragmented_record()
    label, n = record.maskformer_incidence_labels(0.5, 0.2, restrict_to_mask=True)
    assert n == 2
    assert label.tolist() == [0, 0, 0, 1, 1, 1]
