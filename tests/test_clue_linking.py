"""The cross-subsystem linking stage.

Built on a synthetic record rather than a store, so these need no event store. They do need
CLUEstering, because src.clue.pipeline imports it at module scope -- skipped without it, the
same way tests/test_clue_periodic.py does, so `pytest` still runs on a bare clone.
"""

import numpy as np
import pytest

pytest.importorskip("CLUEstering", reason="CLUEstering is not installed; see environment.yml")

from src.clue.pipeline import link_across_subsystems  # noqa: E402


class FakeRecord:
    """The four attributes link_across_subsystems reads, and nothing else."""

    def __init__(self, eta, phi, subsystem, energy=None):
        self._eta = np.asarray(eta, dtype=float)
        self._phi = np.asarray(phi, dtype=float)
        self.subsystem = np.asarray(subsystem, dtype=np.int64)
        self.energy_calib = np.ones_like(self._eta) if energy is None else np.asarray(energy, dtype=float)
        self.n_hits = self._eta.size

    def eta(self):
        return self._eta

    def phi(self):
        return self._phi


def test_disabled_at_zero_radius_returns_input_unchanged():
    rec = FakeRecord([0.0, 0.0], [0.0, 0.0], [0, 2])
    label = np.array([0, 1])
    out = link_across_subsystems(rec, label, 0.0)
    assert np.array_equal(out, label)


def test_links_ecal_to_hcal_at_the_same_angle():
    # one cell in ecb (code 0), one in hcb (code 2), same direction: one shower, split by the
    # per-subsystem pass. This is the 42% case the stage exists for.
    rec = FakeRecord([0.30, 0.30], [1.0, 1.0], [0, 2])
    out = link_across_subsystems(rec, np.array([0, 1]), radius=0.05)
    assert out[0] == out[1]


def test_leaves_distant_clusters_alone():
    rec = FakeRecord([0.30, 1.50], [1.0, 1.0], [0, 2])
    out = link_across_subsystems(rec, np.array([0, 1]), radius=0.05)
    assert out[0] != out[1]


def test_never_merges_within_one_subsystem():
    """Two adjacent ecb clusters stay apart: CLUE already ruled on those."""
    rec = FakeRecord([0.30, 0.301], [1.0, 1.0], [0, 0])
    out = link_across_subsystems(rec, np.array([0, 1]), radius=0.05)
    assert out[0] != out[1]


def test_linking_is_transitive_across_three_subsystems():
    # ecb -> hcb -> hce, each hop inside the radius but the ends 0.08 apart, i.e. further than
    # the radius. Union-find should still make them one cluster.
    rec = FakeRecord([0.30, 0.34, 0.38], [1.0, 1.0, 1.0], [0, 2, 3])
    out = link_across_subsystems(rec, np.array([0, 1, 2]), radius=0.05)
    assert out[0] == out[1] == out[2]


def test_phi_wrap_does_not_hide_a_link():
    """Centroids either side of +-pi are adjacent, not 2pi apart."""
    rec = FakeRecord([0.0, 0.0], [np.pi - 0.01, -np.pi + 0.01], [0, 2])
    out = link_across_subsystems(rec, np.array([0, 1]), radius=0.05)
    assert out[0] == out[1]


def test_phi_wrap_in_the_centroid_itself():
    """A cluster straddling +-pi must not get a centroid at 0.

    Averaging the angle would put cluster 0 at phi=0, half a detector away from where its
    cells are, and it would then fail to link to a genuine neighbour at pi.
    """
    rec = FakeRecord(
        [0.0, 0.0, 0.0],
        [np.pi - 0.02, -np.pi + 0.02, np.pi],
        [0, 0, 2],
    )
    out = link_across_subsystems(rec, np.array([0, 0, 1]), radius=0.05)
    assert out[0] == out[2]


def test_unclustered_cells_stay_unclustered():
    rec = FakeRecord([0.30, 0.30, 0.30], [1.0, 1.0, 1.0], [0, 2, 2])
    out = link_across_subsystems(rec, np.array([0, 1, -1]), radius=0.05)
    assert out[2] == -1
    assert out[0] == out[1]


def test_all_noise_is_a_no_op():
    rec = FakeRecord([0.0, 0.0], [0.0, 0.0], [0, 2])
    label = np.array([-1, -1])
    assert np.array_equal(link_across_subsystems(rec, label, 0.05), label)


@pytest.mark.parametrize("radius", [0.01, 0.05, 0.2])
def test_never_increases_the_cluster_count(radius):
    rng = np.random.default_rng(0)
    n = 200
    rec = FakeRecord(
        rng.uniform(-2, 2, n),
        rng.uniform(-np.pi, np.pi, n),
        rng.integers(0, 4, n),
        rng.uniform(0.1, 5.0, n),
    )
    label = rng.integers(0, 25, n)
    out = link_across_subsystems(rec, label, radius)
    assert np.unique(out[out >= 0]).size <= np.unique(label[label >= 0]).size
