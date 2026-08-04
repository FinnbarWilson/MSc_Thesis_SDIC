"""The truth record: which particle deposited how much energy in which cell.

ColliderML stores, for every cell, the list of particles that contributed to it
and how much each contributed. That makes the truth a matrix of energy fractions
rather than one label per cell, which is what allows a cell shared between two
showers to be represented honestly.

This module turns that record into the two things the rest of the code needs:
the set of reconstructable particles that both methods are scored against, and a
description of how much each particle's footprint is shared with its neighbours.
"""

import numpy as np
import pandas as pd

from src.config import settings
from src.data.loader import ENERGY_COLUMN


def explode_event(event, hit_mask=None):
    """Expand an event into one row per (cell, contributing particle).

    Args:
        event: one event row.
        hit_mask: optional boolean mask over cells, applied before exploding.

    Returns:
        pd.DataFrame with columns ``hit`` (cell index within the masked event),
        ``pid``, ``pe`` (that particle's energy in the cell, GeV), ``x``, ``y``,
        ``z``, ``detector``, ``total_energy`` and ``n_contrib``.
    """
    frame = pd.DataFrame({
        "x": event["x"],
        "y": event["y"],
        "z": event["z"],
        "detector": event["detector"],
        "total_energy": event[ENERGY_COLUMN],
    })
    frame["pids"] = list(event["contrib_particle_ids"])
    frame["pes"] = list(event["contrib_energies"])
    frame["n_contrib"] = frame["pids"].map(len)

    if hit_mask is not None:
        frame = frame[np.asarray(hit_mask)]
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return frame.iloc[0:0]
    frame["hit"] = frame.index

    exploded = frame.explode(["pids", "pes"]).reset_index(drop=True)
    exploded = exploded.rename(columns={"pids": "pid", "pes": "pe"})
    exploded["pe"] = exploded["pe"].astype(float)
    exploded["pid"] = exploded["pid"].astype(np.int64)
    return exploded


def reconstructable_particles(event, hit_mask=None, min_energy_gev=None,
                              min_calo_hits=None):
    """The particles both methods are required to reconstruct.

    A particle qualifies when it is the leading contributor in at least one cell,
    deposits more than ``min_energy_gev`` in total, and touches at least
    ``min_calo_hits`` cells. Leading in a cell matters because a particle that
    only ever contributes a minority share has no cell it could plausibly claim.

    This set is the denominator of efficiency for CLUE and the target set for the
    MaskFormer. The two must not diverge, or the methods stop measuring the same
    quantity.

    Args:
        event: one event row.
        hit_mask: optional boolean mask over cells, restricting the count and the
            energy total to one detector collection.
        min_energy_gev: override for ``particle_selection.min_deposited_energy_gev``.
        min_calo_hits: override for ``particle_selection.min_calo_hits``.

    Returns:
        pd.DataFrame with columns ``pid``, ``deposited_energy`` and ``n_cells``.
    """
    selection = settings()["particle_selection"]
    if min_energy_gev is None:
        min_energy_gev = selection["min_deposited_energy_gev"]
    if min_calo_hits is None:
        min_calo_hits = selection["min_calo_hits"]

    exploded = explode_event(event, hit_mask)
    columns = ["pid", "deposited_energy", "n_cells"]
    if exploded.empty:
        return pd.DataFrame(columns=columns)

    leading = exploded.loc[exploded.groupby("hit")["pe"].idxmax(), "pid"]
    energy = exploded.groupby("pid")["pe"].sum()
    cells = exploded.groupby("pid")["hit"].nunique()

    out = pd.DataFrame({"pid": pd.unique(leading)})
    out["deposited_energy"] = out["pid"].map(energy)
    out["n_cells"] = out["pid"].map(cells)
    keep = (out["deposited_energy"] > min_energy_gev) & (out["n_cells"] >= min_calo_hits)
    return out[keep].reset_index(drop=True)[columns]


def footprint_contamination(event, hit_mask=None):
    """How much of each particle's footprint energy belongs to other particles.

    A particle's footprint is the set of cells it leads. The contamination is the
    fraction of the energy in those cells deposited by anyone else, so 0 means
    the particle owns its cells outright and larger values mean its shower sits
    on top of a neighbour's.

    This is measured from the truth alone, before any algorithm runs, which is
    what makes it a ceiling rather than a result.

    Args:
        event: one event row.
        hit_mask: optional boolean mask over cells.

    Returns:
        pd.DataFrame with columns ``pid``, ``footprint_energy``, ``own_energy``
        and ``contamination``.
    """
    exploded = explode_event(event, hit_mask)
    columns = ["pid", "footprint_energy", "own_energy", "contamination"]
    if exploded.empty:
        return pd.DataFrame(columns=columns)

    leading = exploded.loc[exploded.groupby("hit")["pe"].idxmax()][["hit", "pid"]]
    leading = leading.rename(columns={"pid": "owner"})
    joined = exploded.merge(leading, on="hit")

    grouped = joined.groupby("owner")
    footprint = grouped["pe"].sum()
    own = joined[joined["pid"] == joined["owner"]].groupby("owner")["pe"].sum()

    out = pd.DataFrame({"pid": footprint.index.astype(np.int64)})
    out["footprint_energy"] = footprint.to_numpy()
    out["own_energy"] = out["pid"].map(own).fillna(0.0)
    out["contamination"] = 1.0 - out["own_energy"] / out["footprint_energy"]
    return out.reset_index(drop=True)[columns]


def isolation_flags(contamination, thresholds=None):
    """Label each particle isolated or merged at several contamination thresholds.

    Args:
        contamination: output of :func:`footprint_contamination`.
        thresholds: contamination fractions; defaults to
            ``metrics.contamination_thresholds``.

    Returns:
        pd.DataFrame: ``contamination`` with one extra boolean column per
        threshold, named ``isolated_0pct``, ``isolated_1pct`` and so on.
    """
    if thresholds is None:
        thresholds = settings()["metrics"]["contamination_thresholds"]
    out = contamination.copy()
    for threshold in thresholds:
        out[isolation_column(threshold)] = out["contamination"] <= threshold
    return out


def isolation_column(threshold):
    """Column name for the isolated flag at a given contamination threshold."""
    return f"isolated_{int(round(threshold * 100))}pct"
