"""Where each method's energy goes, resolved inside the shower rather than per particle.

:mod:`src.evaluation.metrics` sums every spatial fact away before it writes a row, so two
methods can reach the same efficiency by failing in opposite ways and no per-particle scalar
tells them apart. Everything here keeps the cell and its position relative to its shower.

Each truth cell gets two shower-frame coordinates: ``dr``, the angular distance from the
particle's energy-weighted axis, and ``depth``, the calibrated layer index minus the first layer
the particle lit. Each is then given one fate: ``recovered`` (in the cluster matched to this
particle), ``stolen`` (in another cluster) or ``dropped`` (in none). The three are exhaustive,
so the profiles sum to the truth profile. The matching is the scorer's own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.matching import hungarian_match, overlap_matrix
from src.evaluation.metrics import delta_phi

#: Transverse bins in (eta, phi). The first few resolve a Moliere-radius core, ~0.02 in angle
#: at a barrel radius of ~1.5 m; the tail runs out to where showers overlap.
DR_EDGES = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35])

#: Layers from the shower's own first lit layer. ECAL is 48 layers of 5.05 mm and HCAL 36 of
#: 51 mm, so the subsystem is kept alongside.
DEPTH_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48])

FATES = ("recovered", "stolen", "dropped")


@dataclass
class ShowerCells:
    """Every (particle, cell) pair of one event, in shower coordinates, with its fate."""

    particle: np.ndarray  # index into the event's particle table
    dr: np.ndarray  # angular distance from the particle's axis
    depth: np.ndarray  # layers past the particle's first lit layer
    subsystem: np.ndarray
    deposit: np.ndarray  # the particle's own energy in this cell, E_ia
    fate: np.ndarray  # 0 recovered, 1 stolen, 2 dropped
    p_energy: np.ndarray  # the particle's energy, repeated per cell
    p_pt: np.ndarray  # the particle's transverse momentum, repeated per cell
    # Which event each pair came from. Filled in by the caller that concatenates events;
    # without it the profiles can be drawn but carry no uncertainty band.
    event: np.ndarray | None = None


def shower_axes(record) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Energy-weighted (eta, phi) axis and first lit layer of every truth particle.

    Weighted by the particle's own deposit rather than the cell's total energy, so a cell the
    particle barely touched cannot drag the axis toward a neighbour's core.
    """
    eta, phi = record.eta(), record.phi()
    deposit = record.truth_deposit
    n = record.n_particles
    axis_eta = np.zeros(n)
    axis_phi = np.zeros(n)
    first_layer = np.zeros(n, dtype=np.int64)
    for p in range(n):
        cells = np.flatnonzero(record.truth_label == p)
        if cells.size == 0:
            continue
        w = deposit[cells]
        if w.sum() <= 0:
            w = np.ones_like(w)
        axis_eta[p] = np.average(eta[cells], weights=w)
        # Circular mean, so a shower at the +/-pi seam is not averaged to zero.
        axis_phi[p] = np.arctan2(np.average(np.sin(phi[cells]), weights=w), np.average(np.cos(phi[cells]), weights=w))
        first_layer[p] = record.layer[cells].min()
    return axis_eta, axis_phi, first_layer


def shower_cells(record, pred_label: np.ndarray, n_pred: int) -> ShowerCells:
    """Every truth cell of the event, in shower coordinates, with its fate under `pred_label`."""
    eta, phi = record.eta(), record.phi()
    deposit = record.truth_deposit
    axis_eta, axis_phi, first_layer = shower_axes(record)

    overlap = overlap_matrix(record.truth_label, pred_label, deposit, record.n_particles, n_pred)
    match = hungarian_match(overlap, min_overlap=0.0)
    # Cluster matched to each particle, -1 where the particle went unmatched.
    matched_cluster = np.full(record.n_particles, -1, dtype=np.int64)
    matched_cluster[match.truth_index] = match.pred_index

    owned = np.flatnonzero(record.truth_label >= 0)
    p = record.truth_label[owned]
    dr = np.hypot(eta[owned] - axis_eta[p], delta_phi(phi[owned], axis_phi[p]))
    depth = record.layer[owned].astype(np.int64) - first_layer[p]

    assigned = pred_label[owned]
    fate = np.where(assigned < 0, 2, np.where(assigned == matched_cluster[p], 0, 1)).astype(np.int8)

    return ShowerCells(
        particle=p,
        dr=dr,
        depth=np.maximum(depth, 0),
        subsystem=record.subsystem[owned],
        deposit=deposit[owned],
        fate=fate,
        p_energy=record.particle_energy[p],
        p_pt=record.particle_pt[p],
    )


def profile(cells: ShowerCells, coord: str, edges: np.ndarray, energy_range: tuple[float, float] | None = None):
    """Energy in each bin of `coord`, split by fate. Returns (centres, truth, {fate: energy})."""
    values = getattr(cells, coord)
    keep = np.ones(values.shape, dtype=bool)
    if energy_range is not None:
        # Selects on pT, matching the differential figures' binning variable.
        keep = (cells.p_pt >= energy_range[0]) & (cells.p_pt < energy_range[1])
    idx = np.digitize(values[keep], edges) - 1
    inside = (idx >= 0) & (idx < len(edges) - 1)
    idx, w, f = idx[inside], cells.deposit[keep][inside], cells.fate[keep][inside]
    truth = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    out = {}
    for code, name in enumerate(FATES):
        out[name] = np.bincount(idx[f == code], weights=w[f == code], minlength=len(edges) - 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, truth, out


def cluster_shape(record, label: np.ndarray, n_clusters: int) -> dict[str, np.ndarray]:
    """Energy-weighted RMS extent of every predicted cluster.

    Transverse extent in (eta, phi), longitudinal in layer index. Their ratio distinguishes a
    cluster that follows a shower from a fixed-radius blob, without needing the truth.
    """
    eta, phi, e = record.eta(), record.phi(), record.energy_calib
    t_ext = np.full(n_clusters, np.nan)
    l_ext = np.full(n_clusters, np.nan)
    n_cells = np.zeros(n_clusters, dtype=np.int64)
    energy = np.zeros(n_clusters)
    for c in range(n_clusters):
        cells = np.flatnonzero(label == c)
        if cells.size == 0:
            continue
        w = e[cells]
        tot = w.sum()
        if tot <= 0:
            continue
        me = np.average(eta[cells], weights=w)
        mp = np.arctan2(np.average(np.sin(phi[cells]), weights=w), np.average(np.cos(phi[cells]), weights=w))
        d = np.hypot(eta[cells] - me, delta_phi(phi[cells], mp))
        t_ext[c] = np.sqrt(np.average(d**2, weights=w))
        lay = record.layer[cells].astype(float)
        l_ext[c] = np.sqrt(np.average((lay - np.average(lay, weights=w)) ** 2, weights=w))
        n_cells[c] = cells.size
        energy[c] = tot
    return {"transverse": t_ext, "longitudinal": l_ext, "n_cells": n_cells, "energy": energy}


def truth_shape(record) -> dict[str, np.ndarray]:
    """The same extents for the truth partition, as the target the methods are read against."""
    return cluster_shape(record, record.truth_label, record.n_particles)
