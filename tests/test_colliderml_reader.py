"""The raw-parquet reader behind the dataset-features figures.

This reader is the one place in the repo that does not go through the event store, so nothing
downstream would catch it getting the truth links wrong -- the figures would simply be wrong
and look plausible. Everything here runs on hand-built events, so it runs anywhere, with no
dataset and no hepattn.

Three properties are worth pinning, and they are the three the reader could plausibly break:

*   Contributions are matched to particles by id, and a particle contributing twice to one
    cell is one hit, not two. Geant records a particle per step, so this is not hypothetical
    and a hit count that counts steps is not a hit count.
*   The shower collapse merges an in-calorimeter secondary onto whatever entered, and the
    secondary's energy goes with it. A collapse that moved the label but not the deposit
    would leave the energy coverage unchanged, which is exactly the number the target
    definition was chosen on.
*   Isolation wraps in phi. Two particles either side of the seam are neighbours, and a
    k-d tree that does not know that reports them as the loneliest pair in the event.
"""

import numpy as np
import pytest

from src.io import colliderml as cml

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# A particle entering the calorimeter, and a secondary it makes once inside. The vertices are
# what decides which is which: 800 mm is inside the tracker, 1600 mm is past the ECAL face.
ENTERING_ID, SECONDARY_ID, MUON_ID = 101, 102, 103


def make_event():
    particles = {
        "event_id": 7,
        "particle_id": np.array([ENTERING_ID, SECONDARY_ID, MUON_ID], dtype=np.int64),
        "parent_id": np.array([0, ENTERING_ID, 0], dtype=np.int64),
        "pdg_id": np.array([211, 22, 13], dtype=np.int64),
        "charge": np.array([1.0, 0.0, -1.0], dtype=np.float32),
        "energy": np.array([10.0, 3.0, 5.0], dtype=np.float32),
        "px": np.array([10.0, 3.0, 0.0], dtype=np.float32),
        "py": np.array([0.0, 0.0, 5.0], dtype=np.float32),
        "pz": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "vx": np.array([800.0, 1600.0, 800.0], dtype=np.float32),
        "vy": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "vz": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "primary": np.array([True, False, True], dtype=bool),
    }
    # Three cells: an ECAL barrel cell the entering particle hits twice (two Geant steps), an
    # ECAL barrel cell only the secondary reaches, and an HCAL barrel cell below threshold.
    calohits = {
        "event_id": 7,
        "detector": np.array([10, 10, 13], dtype=np.int64),
        "total_energy": np.array([1.0e-2, 1.0e-2, 1.0e-9], dtype=np.float64),
        "contrib_counts": np.array([2, 1, 1], dtype=np.int64),
        "contrib_particle_id": np.array([ENTERING_ID, ENTERING_ID, SECONDARY_ID, MUON_ID], dtype=np.int64),
        "contrib_energy": np.array([6.0e-3, 4.0e-3, 1.0e-2, 1.0e-9], dtype=np.float64),
    }
    return particles, calohits


def test_repeated_contribution_is_one_hit():
    particles, calohits = make_event()
    table = cml.event_particle_table(particles, calohits, min_hit_energy=2.0e-4, collapse_shower_secondaries=False)

    entering = table.set_index("particle_id").loc[ENTERING_ID]
    assert entering["n_calohits"] == 1
    assert entering["n_hits_ecal"] == 1
    # Both steps' energy is still counted, unlike the cell.
    assert entering["energy_ecal_calib"] == pytest.approx(1.0e-2 * cml.SUBSYSTEM_CALIBRATION["ecb"])


def test_hit_below_threshold_removes_its_particle():
    particles, calohits = make_event()
    table = cml.event_particle_table(particles, calohits, min_hit_energy=2.0e-4, collapse_shower_secondaries=False)

    # The muon's only cell is 1e-9 GeV, well under zero-suppression, so it owns nothing and
    # drops out of the plotted set entirely rather than appearing with zero hits.
    assert MUON_ID not in set(table["particle_id"])


def test_collapse_moves_both_the_label_and_the_energy():
    particles, calohits = make_event()
    loose = cml.event_particle_table(particles, calohits, min_hit_energy=2.0e-4, collapse_shower_secondaries=False)
    collapsed = cml.event_particle_table(particles, calohits, min_hit_energy=2.0e-4, collapse_shower_secondaries=True)

    assert set(loose["particle_id"]) == {ENTERING_ID, SECONDARY_ID}
    assert set(collapsed["particle_id"]) == {ENTERING_ID}

    # The secondary's cell is now the entering particle's, and so is its energy.
    row = collapsed.set_index("particle_id").loc[ENTERING_ID]
    assert row["n_calohits"] == 2
    assert row["energy_ecal_calib"] == pytest.approx(2.0e-2 * cml.SUBSYSTEM_CALIBRATION["ecb"])
    assert loose["energy_calo_calib"].sum() == pytest.approx(collapsed["energy_calo_calib"].sum())


def test_classification_falls_back_to_hadrons():
    pdg_id = np.array([22, 11, 13, 15, 12, 211, 2112, 1000020040], dtype=np.int64)
    charge = np.array([0, -1, -1, -1, 0, 1, 0, 2], dtype=np.float64)
    assert list(cml.classify(pdg_id, charge)) == [
        "photon",
        "electron",
        "muon",
        "tau",
        "neutrino",
        "charged_hadron",
        "neutral_hadron",
        # A helium nucleus is in no whitelist and must still land somewhere.
        "charged_hadron",
    ]


def test_isolation_wraps_in_phi():
    eta = np.array([0.0, 0.0, 2.0])
    phi = np.array([np.pi - 0.05, -np.pi + 0.05, 0.0])
    dr = cml.min_delta_r(eta, phi)

    # The two either side of the seam are 0.1 apart, not the 2pi - 0.1 a flat metric gives.
    assert dr[0] == pytest.approx(0.1)
    assert dr[1] == pytest.approx(0.1)
    assert dr[2] > 1.0


def test_isolation_of_a_lone_particle_is_infinite():
    assert np.isinf(cml.min_delta_r(np.array([0.5]), np.array([0.1])))[0]


def test_ancestors_stop_outside_the_calorimeter():
    # A chain of three: entering particle, its secondary, and that secondary's secondary.
    particle_id = np.array([1, 2, 3], dtype=np.int64)
    parent_id = np.array([0, 1, 2], dtype=np.int64)
    vx = np.array([500.0, 1600.0, 1800.0])
    vy = np.zeros(3)
    vz = np.zeros(3)

    ancestors = cml.calo_entry_ancestors(particle_id, parent_id, vx, vy, vz)
    assert list(ancestors) == [0, 0, 0]

    # Move the first secondary out to the tracker and it becomes its own target, taking the
    # deeper one with it -- a conversion gives one object per outgoing leg, not one per shower.
    vx_tracker = np.array([500.0, 900.0, 1800.0])
    assert list(cml.calo_entry_ancestors(particle_id, parent_id, vx_tracker, vy, vz)) == [0, 1, 1]
