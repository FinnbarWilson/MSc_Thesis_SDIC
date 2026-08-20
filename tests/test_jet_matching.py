"""Jet matching and the four-vector sum behind the jet figure.

A matching error moves a published ratio without moving anything in the per-object tables. The
parts tested here need neither `fastjet` nor an event store: `delta_r`, the greedy match and the
cluster four-vector sum. The phi wrap is the reason this file exists; `run_antikt` is a thin
call into fastjet and is not re-tested.
"""

import numpy as np
import pytest

from src.evaluation.jets import Jets, cluster_four_vectors, delta_r, match

# --- delta_r ----------------------------------------------------------------


def test_delta_r_wraps_across_pi():
    """Jets at 3.1 and -3.1 are 0.083 apart, not 6.2, so inside the match cone."""
    d = delta_r(0.0, 3.1, 0.0, -3.1)
    assert d == pytest.approx(2 * np.pi - 6.2, abs=1e-9)
    assert d < 0.3


def test_delta_r_is_plain_euclidean_away_from_the_wrap():
    assert delta_r(0.0, 0.0, 0.3, 0.4) == pytest.approx(0.5)


def test_delta_r_is_symmetric_and_zero_on_itself():
    assert delta_r(1.0, 2.0, 1.0, 2.0) == pytest.approx(0.0)
    assert delta_r(0.5, 1.0, -0.5, 2.0) == pytest.approx(delta_r(-0.5, 2.0, 0.5, 1.0))


def test_delta_r_broadcasts_over_an_array_of_candidates():
    """`match` relies on this: one reference against every reco jet at once."""
    d = delta_r(0.0, 0.0, np.array([0.0, 1.0]), np.array([0.1, 0.0]))
    assert d.shape == (2,)
    assert d[0] == pytest.approx(0.1)
    assert d[1] == pytest.approx(1.0)


# --- the greedy match -------------------------------------------------------


def _jets(eta, phi):
    n = len(eta)
    return Jets(pt=np.linspace(100.0, 50.0, n), eta=np.array(eta, dtype=float),
                phi=np.array(phi, dtype=float), energy=np.full(n, 100.0), mass=np.zeros(n))


def test_the_nearest_jet_inside_the_cone_is_taken():
    out = match(_jets([0.0], [0.0]), _jets([0.5, 0.05], [0.0, 0.0]), max_dr=0.3)
    assert out.tolist() == [1]


def test_a_jet_outside_the_cone_is_unmatched_rather_than_matched_to_the_least_bad():
    """-1 is a real outcome: an unmatched reference jet is a miss the figure has to show."""
    out = match(_jets([0.0], [0.0]), _jets([5.0], [0.0]), max_dr=0.3)
    assert out.tolist() == [-1]


def test_a_reco_jet_is_consumed_once():
    """Two reference jets cannot both claim one reco jet, or the response is double counted."""
    reference = _jets([0.0, 0.05], [0.0, 0.0])       # pt-ordered, both near the same reco jet
    out = match(reference, _jets([0.02], [0.0]), max_dr=0.3)
    assert out.tolist() == [0, -1]


def test_matching_wraps_in_phi():
    """The wrap has to survive into the matcher, not just live in delta_r."""
    out = match(_jets([0.0], [3.13]), _jets([0.0], [-3.13]), max_dr=0.3)
    assert out.tolist() == [0]


def test_empty_sides_return_all_unmatched_without_raising():
    empty = Jets(*(np.zeros(0) for _ in range(5)))
    assert match(_jets([0.0], [0.0]), empty).tolist() == [-1]
    assert match(empty, _jets([0.0], [0.0])).size == 0


# --- the cluster four-vector sum -------------------------------------------


class _Record:
    """The four attributes cluster_four_vectors reads, on a unit sphere for easy arithmetic."""

    def __init__(self, x, y, z, energy_calib):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.energy_calib = np.asarray(energy_calib, dtype=float)


def test_cells_are_summed_into_massless_four_vectors_per_cluster():
    """E is the calibrated sum and |p| = E, since a cell has position but no mass."""
    record = _Record(x=[1.0, 1.0, 0.0], y=[0.0, 0.0, 1.0], z=[0.0, 0.0, 0.0],
                     energy_calib=[3.0, 2.0, 7.0])
    px, py, pz, en = cluster_four_vectors(record, np.array([0, 0, 1]), 2)

    assert en.tolist() == [5.0, 7.0]
    assert px.tolist() == [5.0, 0.0]
    assert py.tolist() == [0.0, 7.0]
    assert np.allclose(np.hypot(px, py) + pz * 0, en)   # massless: |p| == E


def test_unclustered_cells_carry_no_energy_into_any_jet():
    """Cells at label -1 are the energy a method left behind; a jet must not inherit it."""
    record = _Record(x=[1.0, 1.0], y=[0.0, 0.0], z=[0.0, 0.0], energy_calib=[4.0, 96.0])
    _, _, _, en = cluster_four_vectors(record, np.array([0, -1]), 1)
    assert en.tolist() == [4.0]


def test_an_empty_cluster_is_dropped_rather_than_becoming_a_zero_energy_jet():
    record = _Record(x=[1.0], y=[0.0], z=[0.0], energy_calib=[5.0])
    _, _, _, en = cluster_four_vectors(record, np.array([2]), 3)
    assert en.tolist() == [5.0]


def test_no_clustered_cells_returns_empty_arrays_rather_than_raising():
    record = _Record(x=[1.0], y=[0.0], z=[0.0], energy_calib=[5.0])
    px, py, pz, en = cluster_four_vectors(record, np.array([-1]), 1)
    assert px.size == py.size == pz.size == en.size == 0
