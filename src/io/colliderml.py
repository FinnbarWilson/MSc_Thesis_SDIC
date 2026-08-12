"""Reading the raw ColliderML parquet, for the figures that describe the dataset itself.

Every other reader in this repo goes through :mod:`src.io.event_store`, and for good reason:
the store is the guarantee that CLUE and the MaskFormer saw the same cells. This module is
the one place that deliberately does not, because the store cannot answer the question these
figures ask.

The store holds the TARGET set -- particles that already passed ``particle_selection`` and,
under ``particle_collapse_shower_secondaries``, were already merged onto their shower's
calorimeter-entering ancestor. A figure drawn from it can only show that the cuts did what
they say. The point of a dataset-features figure is to show the distributions the cuts were
chosen FROM, so it has to start one step earlier, at the parquet the dump itself reads.

WHAT IS AND IS NOT HERE

ColliderML also ships silicon tracker hits and their truth links. This repo never downloaded
them -- the comparison is a calorimeter clustering problem -- so there is no tracker panel to
draw and no muon chamber either. What survives is the calorimeter (four subsystems, two
groups) plus whatever the particle table itself carries: kinematics, vertex, provenance.

The physics constants below are duplicated from ``ColliderMLDataset`` in hepattn rather than
imported, for the same reason :mod:`src.io.event_store` duplicates the store format: this
side of the analysis has to be readable and runnable without a GPU stack installed. They are
checked against the store's own metadata by :func:`check_against_store_metadata`, so the
duplication is verifiable rather than a promise.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

#: Where the downloaded shards live on this machine. Only ever a default: every entry point
#: takes ``--raw-root``, because this path is a property of the machine and not of the
#: experiment, and so has no business in ``config/experiment.yaml``.
DEFAULT_RAW_ROOT = Path("/mnt/ai-datastore/finnbar/ColliderML_data/data")

#: Detector ids grouped into the four readout subsystems, in the order the store uses.
SUBSYSTEM_DETECTOR_IDS: Mapping[str, tuple[int, ...]] = {
    "ecb": (10,),
    "ece": (9, 11),
    "hcb": (13,),
    "hce": (12, 14),
}

#: Sampling calibration per subsystem. ECAL and HCAL differ, so a particle's calibrated
#: energy depends on how its shower split between them and the factor does not cancel.
SUBSYSTEM_CALIBRATION: Mapping[str, float] = {
    "ecb": 37.5,
    "ece": 38.7,
    "hcb": 45.0,
    "hce": 46.9,
}

#: The two groups the figures are binned by. ECAL and HCAL, as unions of barrel and endcap.
GROUP_SUBSYSTEMS: Mapping[str, tuple[str, ...]] = {
    "ecal": ("ecb", "ece"),
    "hcal": ("hcb", "hce"),
}

#: Calorimeter front face, in millimetres, measured from the ColliderML hit cloud. A particle
#: born outside it entered the calorimeter; one born inside is a shower product of something
#: that did. Same numbers as hepattn's ``calo_entry_radius`` / ``calo_entry_abs_z``.
CALO_ENTRY_RADIUS_MM = 1252.0
CALO_ENTRY_ABS_Z_MM = 3202.0

#: Plotted in this order, and this is also the priority: a particle is charged- or
#: neutral-hadron only if it matched none of the explicit pdg tests above it.
CLASS_ORDER: tuple[str, ...] = (
    "charged_hadron",
    "neutral_hadron",
    "electron",
    "photon",
    "muon",
    "tau",
    "neutrino",
    "other",
)

_PARTICLE_COLUMNS = (
    "event_id",
    "particle_id",
    "parent_id",
    "pdg_id",
    "charge",
    "energy",
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
    "primary",
)

# contrib_times is deliberately absent: it is a third of the calo-hit file by volume and
# nothing here reads it. Dropping it is the difference between a pu200 shard fitting in
# memory and not.
_CALOHIT_COLUMNS = (
    "event_id",
    "detector",
    "total_energy",
    "contrib_particle_ids",
    "contrib_energies",
)


class ColliderMLError(Exception):
    """A problem with the raw dataset on disk."""


def shard_paths(root: Path | str, dataset: str, kind: str) -> list[Path]:
    """The parquet shards for one collection, in file order.

    Args:
        root: directory holding the ``ttbar_<dataset>_<kind>`` subdirectories.
        dataset: ``pu0`` or ``pu200``.
        kind: ``particles`` or ``calo_hits``.
    """
    directory = Path(root) / f"ttbar_{dataset}_{kind}"
    if not directory.is_dir():
        msg = f"no such collection: {directory}. Pass --raw-root if the data lives elsewhere."
        raise ColliderMLError(msg)
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        msg = f"{directory} holds no parquet shards"
        raise ColliderMLError(msg)
    return paths


def _lists(batch, name: str, row: int) -> np.ndarray:
    """One row of a list column, as a flat numpy array."""
    return batch.column(name)[row].values.to_numpy(zero_copy_only=False)


def _flat_lists(batch, name: str, row: int) -> np.ndarray:
    """One row of a list-of-list column, flattened across both levels.

    ``.flatten()`` rather than ``.values``: the latter hands back the whole column's child
    array, ignoring the row's offsets, which silently gives every event the same 1.3 million
    contributions instead of its own sixteen thousand.
    """
    return batch.column(name)[row].values.flatten().to_numpy(zero_copy_only=False)


def _list_lengths(batch, name: str, row: int) -> np.ndarray:
    """Per inner list, its length -- i.e. contributions per hit."""
    import pyarrow.compute as pc

    lengths = pc.list_value_length(batch.column(name)[row].values)
    return lengths.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def iter_events(
    root: Path | str,
    dataset: str,
    n_events: int | None = None,
    batch_size: int = 25,
) -> Iterator[tuple[dict, dict]]:
    """Yield ``(particles, calohits)`` per event, as dicts of flat numpy arrays.

    The two collections are separate files that agree row for row, which is asserted rather
    than assumed: a mismatch here would silently score one event's hits against another's
    particles.

    Args:
        root: the raw data root.
        dataset: ``pu0`` or ``pu200``.
        n_events: stop after this many. ``None`` reads every shard, which for pu200 is
            hundreds of gigabytes.
        batch_size: events per pyarrow batch. Only affects peak memory.
    """
    particle_shards = shard_paths(root, dataset, "particles")
    calohit_shards = shard_paths(root, dataset, "calo_hits")
    if len(particle_shards) != len(calohit_shards):
        msg = f"{len(particle_shards)} particle shards against {len(calohit_shards)} calo-hit shards"
        raise ColliderMLError(msg)

    seen = 0
    for particle_path, calohit_path in zip(particle_shards, calohit_shards, strict=True):
        particle_batches = pq.ParquetFile(particle_path).iter_batches(batch_size=batch_size, columns=list(_PARTICLE_COLUMNS))
        calohit_batches = pq.ParquetFile(calohit_path).iter_batches(batch_size=batch_size, columns=list(_CALOHIT_COLUMNS))
        for particle_batch, calohit_batch in zip(particle_batches, calohit_batches, strict=True):
            for row in range(particle_batch.num_rows):
                event_id = int(particle_batch.column("event_id")[row].as_py())
                hit_event_id = int(calohit_batch.column("event_id")[row].as_py())
                if event_id != hit_event_id:
                    msg = f"{particle_path.name} row {row} is event {event_id}, {calohit_path.name} row {row} is event {hit_event_id}"
                    raise ColliderMLError(msg)

                particles = {
                    "event_id": event_id,
                    **{name: _lists(particle_batch, name, row) for name in _PARTICLE_COLUMNS if name != "event_id"},
                }
                calohits = {
                    "event_id": event_id,
                    "detector": _lists(calohit_batch, "detector", row),
                    "total_energy": _lists(calohit_batch, "total_energy", row),
                    "contrib_counts": _list_lengths(calohit_batch, "contrib_particle_ids", row),
                    "contrib_particle_id": _flat_lists(calohit_batch, "contrib_particle_ids", row),
                    "contrib_energy": _flat_lists(calohit_batch, "contrib_energies", row),
                }
                yield particles, calohits

                seen += 1
                if n_events is not None and seen >= n_events:
                    return


def classify(pdg_id: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Particle class per row, as strings from :data:`CLASS_ORDER`.

    Mirrors ``ColliderMLDataset._add_particle_classes_inplace``: the explicit pdg tests come
    first, and anything left over is a hadron, charged or neutral by its charge. "Left over"
    rather than a hadron whitelist is the point -- it means nuclear fragments and exotics land
    in a bin instead of vanishing from the plot.
    """
    abs_pdg = np.abs(np.asarray(pdg_id, dtype=np.int64))
    label = np.full(abs_pdg.size, "other", dtype=object)

    explicit = {
        "photon": abs_pdg == 22,
        "electron": abs_pdg == 11,
        "muon": abs_pdg == 13,
        "tau": abs_pdg == 15,
        "neutrino": np.isin(abs_pdg, (12, 14, 16, 18)),
    }
    known = np.zeros(abs_pdg.size, dtype=bool)
    for name, mask in explicit.items():
        label[mask] = name
        known |= mask

    charged = np.asarray(charge) != 0
    label[~known & charged] = "charged_hadron"
    label[~known & ~charged] = "neutral_hadron"
    return label


def calo_entry_ancestors(particle_id: np.ndarray, parent_id: np.ndarray, vx, vy, vz) -> np.ndarray:
    """Row index of each particle's calorimeter-entering ancestor.

    Climbs ``parent_id`` while the current particle was born inside the calorimeter, so the
    chain stops at whatever entered it. A particle born in the tracker is its own ancestor,
    which keeps a conversion's two legs as two objects rather than fusing them at the photon.

    Pointer doubling rather than a per-particle loop: a pu200 event holds 150,000 particles.
    The iteration cap also makes a cyclic parent chain terminate instead of hanging.

    Returns:
        int64 row indices, ``arange`` where the particle is its own ancestor.
    """
    particle_id = np.asarray(particle_id, dtype=np.int64)
    parent_id = np.asarray(parent_id, dtype=np.int64)
    n = particle_id.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    born_in_calo = (np.hypot(vx, vy) >= CALO_ENTRY_RADIUS_MM) | (np.abs(vz) >= CALO_ENTRY_ABS_Z_MM)

    parent_row, has_parent = _lookup_rows(particle_id, parent_id)
    parent_row_of = np.arange(n, dtype=np.int64)
    parent_row_of[has_parent] = parent_row

    ancestors = np.arange(n, dtype=np.int64)
    for _ in range(64):
        climbing = born_in_calo[ancestors] & has_parent[ancestors]
        if not climbing.any():
            break
        nxt = ancestors.copy()
        nxt[climbing] = parent_row_of[ancestors[climbing]]
        if np.array_equal(nxt, ancestors):
            break
        ancestors = nxt
    return ancestors


def _lookup_rows(particle_id: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row of each query id in the particle table.

    Returns:
        ``(rows, matched)`` where ``rows`` holds a row index per MATCHED query, in query
        order, and ``matched`` is the boolean mask selecting them. Unmatched queries are
        generator-level roots and hits from particles the table does not carry.
    """
    order = np.argsort(particle_id, kind="stable")
    sorted_ids = particle_id[order]
    pos = np.searchsorted(sorted_ids, query)
    pos_clipped = np.clip(pos, 0, sorted_ids.size - 1)
    matched = sorted_ids[pos_clipped] == query
    return order[pos_clipped[matched]], matched


def min_delta_r(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Angular distance from each particle to the nearest OTHER particle in the set.

    The same variable :func:`src.evaluation.metrics.local_density` reports, computed the same
    way from generator momenta -- but that function forms the full n-by-n distance matrix,
    which is fine for a few hundred targets and impossible for a pu200 particle table. A
    k-d tree gives the identical answer at 150,000 particles.

    phi wraps, so the tree is built over three copies of the points, shifted by -2pi, 0 and
    +2pi. Querying the unshifted copies for two neighbours then finds a partner across the
    wrap; the first neighbour is always the point itself.

    Returns:
        float64, ``inf`` for a particle that is alone in its event.
    """
    from scipy.spatial import cKDTree

    n = int(np.size(eta))
    if n < 2:
        return np.full(n, np.inf)

    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    tiled = np.column_stack(
        [
            np.tile(eta, 3),
            np.concatenate([phi - 2 * np.pi, phi, phi + 2 * np.pi]),
        ]
    )
    distance, _ = cKDTree(tiled).query(np.column_stack([eta, phi]), k=2)
    return distance[:, 1]


def event_particle_table(
    particles: Mapping,
    calohits: Mapping,
    min_hit_energy: float,
    collapse_shower_secondaries: bool,
) -> pd.DataFrame:
    """One event's particles, with their calorimeter deposits summarised per particle.

    Only particles that own at least one surviving cell are kept. That is the "post event
    cleaning" set the reconstructability cuts are then chosen from: a particle with no cell
    is invisible to a calorimeter clustering algorithm whatever its momentum, and at pu200 it
    is 150,000 rows of nothing.

    Args:
        particles: one event from :func:`iter_events`.
        calohits: the same event's cells.
        min_hit_energy: zero-suppression threshold in GeV, applied before anything is counted,
            so the hit counts here are the hit counts the network is given.
        collapse_shower_secondaries: merge each in-calorimeter secondary onto the particle
            whose shower it belongs to, matching the thesis's target definition. With this
            off, every Geant fragment is its own row.

    Returns:
        One row per surviving particle.
    """
    px, py, pz = (np.asarray(particles[k], dtype=np.float64) for k in ("px", "py", "pz"))
    vx, vy, vz = (np.asarray(particles[k], dtype=np.float64) for k in ("vx", "vy", "vz"))
    particle_id = np.asarray(particles["particle_id"], dtype=np.int64)
    n_particles = particle_id.size

    keep_hit = np.asarray(calohits["total_energy"], dtype=np.float64) >= min_hit_energy
    detector = np.asarray(calohits["detector"], dtype=np.int64)
    counts = np.asarray(calohits["contrib_counts"], dtype=np.int64)

    hit_of_contrib = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    contrib_id = np.asarray(calohits["contrib_particle_id"], dtype=np.int64)
    contrib_energy = np.asarray(calohits["contrib_energy"], dtype=np.float64)

    alive = keep_hit[hit_of_contrib]
    hit_of_contrib, contrib_id, contrib_energy = hit_of_contrib[alive], contrib_id[alive], contrib_energy[alive]

    if collapse_shower_secondaries and n_particles:
        ancestors = calo_entry_ancestors(particle_id, particles["parent_id"], vx, vy, vz)
        remap_from = particle_id
        remap_to = particle_id[ancestors]
        rows, matched = _lookup_rows(remap_from, contrib_id)
        contrib_id = contrib_id.copy()
        contrib_id[matched] = remap_to[rows]
        is_target = ancestors == np.arange(n_particles)
    else:
        is_target = np.ones(n_particles, dtype=bool)

    rows, matched = _lookup_rows(particle_id, contrib_id)
    hit_of_contrib, contrib_energy = hit_of_contrib[matched], contrib_energy[matched]

    columns: dict[str, np.ndarray] = {}
    n_calohits = np.zeros(n_particles, dtype=np.int64)
    if rows.size:
        # Distinct (particle, cell) pairs, not contributions: Geant can record a particle
        # twice in one cell, and a hit count that grows with step multiplicity is not a hit
        # count. This is what the store's CSR does with sum_duplicates.
        pair = np.unique(np.column_stack([rows, hit_of_contrib]), axis=0)
        n_calohits = np.bincount(pair[:, 0], minlength=n_particles)

        detector_of_contrib = detector[hit_of_contrib]
        for subsystem, ids in SUBSYSTEM_DETECTOR_IDS.items():
            in_subsystem = np.isin(detector_of_contrib, ids)
            columns[f"energy_{subsystem}"] = np.bincount(
                rows[in_subsystem], weights=contrib_energy[in_subsystem], minlength=n_particles
            )
            pair_in = np.unique(np.column_stack([rows[in_subsystem], hit_of_contrib[in_subsystem]]), axis=0)
            columns[f"n_hits_{subsystem}"] = (
                np.bincount(pair_in[:, 0], minlength=n_particles) if pair_in.size else np.zeros(n_particles, dtype=np.int64)
            )
    else:
        for subsystem in SUBSYSTEM_DETECTOR_IDS:
            columns[f"energy_{subsystem}"] = np.zeros(n_particles)
            columns[f"n_hits_{subsystem}"] = np.zeros(n_particles, dtype=np.int64)

    keep = (n_calohits > 0) & is_target
    pt = np.hypot(px, py)
    p = np.sqrt(px**2 + py**2 + pz**2)
    eta = np.arctanh(np.clip(np.divide(pz, np.maximum(p, 1e-12)), -1 + 1e-12, 1 - 1e-12))
    phi = np.arctan2(py, px)

    table = {
        "event_id": np.full(int(keep.sum()), particles["event_id"], dtype=np.int32),
        "particle_id": particle_id[keep],
        "pdg_id": np.asarray(particles["pdg_id"], dtype=np.int64)[keep],
        "particle_class": classify(particles["pdg_id"], particles["charge"])[keep],
        "charge": np.asarray(particles["charge"], dtype=np.float32)[keep],
        "primary": np.asarray(particles["primary"], dtype=bool)[keep],
        "pt": pt[keep],
        "eta": eta[keep],
        "phi": phi[keep],
        "energy": np.asarray(particles["energy"], dtype=np.float64)[keep],
        "vertex_r": np.hypot(vx, vy)[keep],
        "vertex_z": vz[keep],
        "n_calohits": n_calohits[keep],
    }

    calibrated_total = np.zeros(n_particles)
    for group, subsystems in GROUP_SUBSYSTEMS.items():
        raw = sum(columns[f"energy_{s}"] for s in subsystems)
        calibrated = sum(columns[f"energy_{s}"] * SUBSYSTEM_CALIBRATION[s] for s in subsystems)
        calibrated_total += calibrated
        table[f"n_hits_{group}"] = sum(columns[f"n_hits_{s}"] for s in subsystems)[keep]
        table[f"energy_{group}_raw"] = raw[keep]
        table[f"energy_{group}_calib"] = calibrated[keep]
    for subsystem, scale in SUBSYSTEM_CALIBRATION.items():
        table[f"n_hits_{subsystem}"] = columns[f"n_hits_{subsystem}"][keep]
        table[f"energy_{subsystem}_calib"] = columns[f"energy_{subsystem}"][keep] * scale
    table["energy_calo_calib"] = calibrated_total[keep]

    frame = pd.DataFrame(table)
    # Isolation is a property of the SET being plotted, so it is computed after the cut on
    # deposits: the nearest neighbour of a particle in this figure is another particle in
    # this figure, not a neutrino two millimetres away that no calorimeter ever saw.
    frame["dr_min"] = min_delta_r(frame["eta"].to_numpy(), frame["phi"].to_numpy())
    return frame


def build_particle_table(
    root: Path | str,
    dataset: str,
    n_events: int,
    min_hit_energy: float,
    collapse_shower_secondaries: bool,
    progress_every: int = 25,
) -> pd.DataFrame:
    """Run :func:`event_particle_table` over the first ``n_events`` events and concatenate."""
    frames: list[pd.DataFrame] = []
    for i, (particles, calohits) in enumerate(iter_events(root, dataset, n_events), start=1):
        frames.append(event_particle_table(particles, calohits, min_hit_energy, collapse_shower_secondaries))
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{n_events} events, {sum(len(f) for f in frames):,} particles", flush=True)
    if not frames:
        msg = "no events read"
        raise ColliderMLError(msg)
    table = pd.concat(frames, ignore_index=True)
    table["particle_class"] = table["particle_class"].astype("category")
    return table


def check_against_store_metadata(metadata: Mapping, subsystem_order: Sequence[str]) -> list[str]:
    """Compare this module's physics constants against an event store's own metadata.

    The constants here are a hand copy of hepattn's, which is only defensible if the copy is
    checkable. A store dumped from the real loader carries the detector-id groups and the
    sampling calibrations it actually used; this reports every place the two disagree.

    Returns:
        Human-readable disagreements, empty when the copy is faithful.
    """
    problems: list[str] = []
    detector = metadata.get("detector", {})
    groups = detector.get("subsystem_detector_ids", {})
    calibration = detector.get("subsystem_calibration", {})

    for subsystem in subsystem_order:
        if subsystem in groups and tuple(groups[subsystem]) != SUBSYSTEM_DETECTOR_IDS.get(subsystem, ()):
            problems.append(f"{subsystem}: store detector ids {tuple(groups[subsystem])} against {SUBSYSTEM_DETECTOR_IDS.get(subsystem)}")
        if subsystem in calibration and not np.isclose(calibration[subsystem], SUBSYSTEM_CALIBRATION.get(subsystem, np.nan)):
            problems.append(f"{subsystem}: store calibration {calibration[subsystem]} against {SUBSYSTEM_CALIBRATION.get(subsystem)}")
    return problems
