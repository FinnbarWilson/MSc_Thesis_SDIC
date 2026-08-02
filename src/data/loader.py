"""Reading ColliderML calorimeter events and applying the shared event selection.

One event is a row of the calorimeter parquet file, holding per-hit arrays: the
cell position ``x, y, z`` in mm, the deposited ``total_energy`` in GeV, the
detector collection each cell belongs to, and the truth record for that cell as
``contrib_particle_ids`` and ``contrib_energies``.

The selection applied here is the one defined in ``config/experiment.yaml``, so
every consumer of an event sees the same cells and the same events.
"""

import numpy as np
import pandas as pd

from src.config import dataset_paths, settings, split_bounds

ENERGY_COLUMN = "total_energy"


def load_calo_events():
    """Read the calorimeter file for the active dataset into memory.

    Returns:
        pd.DataFrame: one row per event, before any selection.
    """
    calo_path, _ = dataset_paths()
    return pd.read_parquet(calo_path)


def hit_count(event):
    """Number of recorded cells in an event, before the hit energy threshold."""
    return len(event["x"])


def passes_event_selection(event, min_calo_hits=None):
    """Whether an event carries enough calorimeter signal to be usable.

    Part of the pu0 release records a near-empty calorimeter despite a complete
    generator record. Such events hold no reconstructable jets and are excluded.

    Args:
        event: one event row.
        min_calo_hits: override for ``event_selection.min_calo_hits``.

    Returns:
        bool: True if the event should be used.
    """
    if min_calo_hits is None:
        min_calo_hits = settings()["event_selection"]["min_calo_hits"]
    return hit_count(event) >= min_calo_hits


def hit_energy_mask(event, min_energy_gev=None):
    """Boolean mask over an event's cells selecting those above the readout threshold.

    Applied before either clustering method sees the event, so both consume the
    same cells.

    Args:
        event: one event row.
        min_energy_gev: override for ``hit_selection.min_energy_gev``.

    Returns:
        np.ndarray: boolean mask, length equal to the number of cells.
    """
    if min_energy_gev is None:
        min_energy_gev = settings()["hit_selection"]["min_energy_gev"]
    return np.asarray(event[ENERGY_COLUMN], dtype=float) > min_energy_gev


def detector_mask(event, detector):
    """Boolean mask over an event's cells selecting one detector collection."""
    return np.asarray(event["detector"]) == detector


def select_events(events, split):
    """Return the events of one split, after event selection.

    Selection is applied first and the window is then taken over the surviving
    events, so a split always contains exactly the number of events configured
    rather than that number minus however many were rejected.

    Args:
        events: the full DataFrame from :func:`load_calo_events`.
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        pd.DataFrame: the selected events, with a fresh index.
    """
    keep = [passes_event_selection(events.iloc[i]) for i in range(len(events))]
    usable = events[np.asarray(keep)].reset_index(drop=True)

    start, count = split_bounds(split)
    if start + count > len(usable):
        raise ValueError(
            f"Split '{split}' needs events [{start}, {start + count}) but only "
            f"{len(usable)} events survive selection. Reduce the split sizes or "
            f"point at a larger file."
        )
    return usable.iloc[start:start + count].reset_index(drop=True)


def selection_summary(events):
    """Count how many events pass selection, for reporting the cost of the cut.

    Returns:
        dict: ``total``, ``kept`` and ``rejected`` event counts.
    """
    keep = [passes_event_selection(events.iloc[i]) for i in range(len(events))]
    kept = int(np.sum(keep))
    return {"total": len(events), "kept": kept, "rejected": len(events) - kept}
