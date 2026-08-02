"""Matching predicted clusters to truth particles.

Nothing in this module knows which method produced a clustering. It takes a
label per cell and the event's truth record, and returns the correspondence
between clusters and particles that the metrics are computed from. Both the CLUE
baseline and the MaskFormer are scored through it, which is what guarantees the
two are measured the same way.

A cluster is matched to the particle that contributed the most energy to it.
"""

import numpy as np
import pandas as pd

from src.data.loader import ENERGY_COLUMN

CLUSTER_COLUMNS = [
    "cluster_id", "matched_pid", "matched_energy", "cluster_energy",
    "particle_energy", "purity", "efficiency",
]


def cluster_matches(event, cluster_ids, hit_mask=None):
    """Match every predicted cluster to its dominant truth particle.

    Args:
        event: one event row.
        cluster_ids: label per cell, -1 for unclustered.
        hit_mask: optional boolean mask over cells. Both the purity numerator and
            the efficiency denominator are restricted to the masked cells, so the
            result is local to one detector collection.

    Returns:
        pd.DataFrame, one row per cluster, with ``purity`` (fraction of the
        cluster's energy belonging to its matched particle) and ``efficiency``
        (fraction of that particle's energy captured by this cluster).
    """
    frame = pd.DataFrame({
        "energy": event[ENERGY_COLUMN],
        "pids": list(event["contrib_particle_ids"]),
        "pes": list(event["contrib_energies"]),
        "cluster_id": np.asarray(cluster_ids),
    })
    if hit_mask is not None:
        frame = frame[np.asarray(hit_mask)].reset_index(drop=True)

    if frame.empty or not (frame["cluster_id"] != -1).any():
        return pd.DataFrame(columns=CLUSTER_COLUMNS)

    exploded = frame.explode(["pids", "pes"])
    exploded["pes"] = exploded["pes"].astype(float)
    exploded["pids"] = exploded["pids"].astype(np.int64)
    clustered = exploded[exploded["cluster_id"] != -1]

    per_pair = clustered.groupby(["cluster_id", "pids"], as_index=False)["pes"].sum()
    per_pair = per_pair.sort_values(["cluster_id", "pes"], ascending=[True, False])
    dominant = per_pair.drop_duplicates("cluster_id").rename(
        columns={"pids": "matched_pid", "pes": "matched_energy"}
    )

    cluster_energy = frame[frame["cluster_id"] != -1].groupby(
        "cluster_id", as_index=False
    )["energy"].sum().rename(columns={"energy": "cluster_energy"})

    particle_energy = exploded.groupby("pids", as_index=False)["pes"].sum().rename(
        columns={"pids": "matched_pid", "pes": "particle_energy"}
    )

    rows = dominant.merge(cluster_energy, on="cluster_id")
    rows = rows.merge(particle_energy, on="matched_pid")
    rows["purity"] = rows["matched_energy"] / rows["cluster_energy"]
    rows["efficiency"] = rows["matched_energy"] / rows["particle_energy"]
    return rows[CLUSTER_COLUMNS]


def particle_matches(matches, targets):
    """Score every reconstructable particle, including those no cluster claimed.

    Working from the cluster side alone hides failures: a particle merged into a
    neighbour's cluster dominates nothing and simply disappears from the
    denominator instead of counting as a miss. Starting from the target list
    instead asks, of every particle that should have been reconstructed, how much
    of it ended up in one cluster of its own.

    A particle split across several clusters is credited only with its largest
    piece, so fragmentation costs efficiency.

    Args:
        matches: output of :func:`cluster_matches`.
        targets: reconstructable particles, needing ``pid`` and
            ``deposited_energy``.

    Returns:
        pd.DataFrame, one row per target particle, with ``n_clusters`` (how many
        clusters it dominates), ``recovered_energy`` and ``efficiency``.
    """
    out = targets[["pid", "deposited_energy"]].copy()
    out["pid"] = out["pid"].astype(np.int64)

    if matches.empty or out.empty:
        out["n_clusters"] = 0
        out["recovered_energy"] = 0.0
        out["efficiency"] = 0.0
        return out

    grouped = matches.groupby(matches["matched_pid"].astype(np.int64)).agg(
        n_clusters=("cluster_id", "size"),
        recovered_energy=("matched_energy", "max"),
    )
    out = out.merge(grouped, left_on="pid", right_index=True, how="left")
    out["n_clusters"] = out["n_clusters"].fillna(0).astype(int)
    out["recovered_energy"] = out["recovered_energy"].fillna(0.0)
    out["efficiency"] = out["recovered_energy"] / out["deposited_energy"].replace(0, np.nan)
    return out


def energy_breakdown(event, cluster_ids, matches, hit_mask=None):
    """Split each matched particle's energy into recovered, discarded and misplaced.

    Separates the two ways a particle loses energy: cells the algorithm called
    noise, and cells it gave to a different cluster. The first is governed by the
    density threshold and the second by genuine mis-clustering, so keeping them
    apart says which one is limiting performance.

    Returns:
        pd.DataFrame: ``matches`` with ``energy_recovered``, ``energy_discarded``
        and ``energy_misplaced`` columns added.
    """
    out = matches.copy()
    if out.empty:
        for column in ("energy_recovered", "energy_discarded", "energy_misplaced"):
            out[column] = pd.Series(dtype=float)
        return out

    frame = pd.DataFrame({
        "pids": list(event["contrib_particle_ids"]),
        "pes": list(event["contrib_energies"]),
        "cluster_id": np.asarray(cluster_ids),
    })
    if hit_mask is not None:
        frame = frame[np.asarray(hit_mask)].reset_index(drop=True)

    noise = frame[frame["cluster_id"] == -1]
    if noise.empty:
        discarded = pd.Series(dtype=float)
    else:
        exploded = noise.explode(["pids", "pes"])
        exploded["pes"] = exploded["pes"].astype(float)
        discarded = exploded.groupby("pids")["pes"].sum()

    out["energy_recovered"] = out["matched_energy"]
    out["energy_discarded"] = out["matched_pid"].map(discarded).fillna(0.0)
    out["energy_misplaced"] = (
        out["particle_energy"] - out["energy_recovered"] - out["energy_discarded"]
    ).clip(lower=0.0)
    return out
