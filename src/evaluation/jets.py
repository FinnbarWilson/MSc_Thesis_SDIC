"""Anti-k_t jets built from clusters, and the same jets built from the truth partition.

The reference is the truth partition of the same cells, not the generator particles, so the
reference jets are what a perfect clusterer would produce from this calorimeter under these
cuts. Comparing against generator jets would fold in zero-suppression, the target selection and
the sampling resolution, none of which belong to the algorithm under test.

Each cell is treated as a massless particle from the interaction point through the cell centre,
``p = E_calib * (x, y, z) / |(x, y, z)|``, and a cluster's four-vector is the sum over its
cells. Both sides use the same rule, so any difference in the jets comes from which cells were
grouped together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: anti-k_t radius: the LHC default for a ttbar final state.
JET_R = 0.4
#: Analysis threshold, applied to both the reference and the reconstructed jets.
JET_MIN_PT = 25.0
#: Match cone, deliberately smaller than JET_R.
MATCH_DR = 0.3


@dataclass
class Jets:
    pt: np.ndarray
    eta: np.ndarray
    phi: np.ndarray
    energy: np.ndarray
    mass: np.ndarray

    def __len__(self) -> int:
        return int(self.pt.size)


def cluster_four_vectors(record, label: np.ndarray, n_clusters: int) -> tuple[np.ndarray, ...]:
    """Sum each cluster's cells into a massless four-vector. Returns (px, py, pz, E)."""
    e = record.energy_calib
    r = np.sqrt(record.x**2 + record.y**2 + record.z**2)
    r = np.where(r > 0, r, 1.0)
    ux, uy, uz = record.x / r, record.y / r, record.z / r
    keep = label >= 0
    if not keep.any() or n_clusters == 0:
        z = np.zeros(0)
        return z, z, z, z
    idx = label[keep]
    w = e[keep]
    px = np.bincount(idx, weights=w * ux[keep], minlength=n_clusters)
    py = np.bincount(idx, weights=w * uy[keep], minlength=n_clusters)
    pz = np.bincount(idx, weights=w * uz[keep], minlength=n_clusters)
    en = np.bincount(idx, weights=w, minlength=n_clusters)
    alive = en > 0
    return px[alive], py[alive], pz[alive], en[alive]


def run_antikt(px, py, pz, energy, radius: float = JET_R, min_pt: float = JET_MIN_PT) -> Jets:
    """Cluster four-vectors into anti-k_t jets above `min_pt`."""
    import fastjet

    if px.size == 0:
        z = np.zeros(0)
        return Jets(z, z, z, z, z)
    pjs = [fastjet.PseudoJet(float(a), float(b), float(c), float(d)) for a, b, c, d in zip(px, py, pz, energy, strict=True)]
    sequence = fastjet.ClusterSequence(pjs, fastjet.JetDefinition(fastjet.antikt_algorithm, radius))
    jets = fastjet.sorted_by_pt(sequence.inclusive_jets(min_pt))
    if not jets:
        z = np.zeros(0)
        return Jets(z, z, z, z, z)
    return Jets(
        pt=np.array([j.pt() for j in jets]),
        eta=np.array([j.eta() for j in jets]),
        phi=np.array([j.phi() for j in jets]),
        energy=np.array([j.e() for j in jets]),
        mass=np.array([j.m() for j in jets]),
    )


def delta_r(eta1, phi1, eta2, phi2):
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    return np.hypot(eta1 - eta2, dphi)


def match(reference: Jets, reco: Jets, max_dr: float = MATCH_DR) -> np.ndarray:
    """For each reference jet, the index of the closest reco jet within `max_dr`, else -1.

    Greedy by reference pt, the order `run_antikt` returns. A reco jet is consumed once taken.
    """
    out = np.full(len(reference), -1, dtype=np.int64)
    if len(reference) == 0 or len(reco) == 0:
        return out
    taken = np.zeros(len(reco), dtype=bool)
    for i in range(len(reference)):
        d = delta_r(reference.eta[i], reference.phi[i], reco.eta, reco.phi)
        d = np.where(taken, np.inf, d)
        j = int(np.argmin(d))
        if d[j] <= max_dr:
            out[i] = j
            taken[j] = True
    return out


def event_rows(record, labels_by_method: dict[str, tuple[np.ndarray, int]], dataset: str) -> list[dict]:
    """One row per (reference jet, method), plus the fake count for that method."""
    tpx, tpy, tpz, te = cluster_four_vectors(record, record.truth_label, record.n_particles)
    reference = run_antikt(tpx, tpy, tpz, te)
    rows: list[dict] = []
    for method, (label, n) in labels_by_method.items():
        px, py, pz, en = cluster_four_vectors(record, label, n)
        reco = run_antikt(px, py, pz, en)
        idx = match(reference, reco)
        for i in range(len(reference)):
            j = idx[i]
            rows.append({
                "dataset": dataset, "algo": method, "sample_id": int(record.sample_id),
                "ref_pt": float(reference.pt[i]), "ref_eta": float(reference.eta[i]),
                "ref_energy": float(reference.energy[i]), "ref_mass": float(reference.mass[i]),
                "matched": bool(j >= 0),
                "reco_pt": float(reco.pt[j]) if j >= 0 else np.nan,
                "reco_energy": float(reco.energy[j]) if j >= 0 else np.nan,
                "dr": float(delta_r(reference.eta[i], reference.phi[i], reco.eta[j], reco.phi[j])) if j >= 0 else np.nan,
                "n_ref_jets": len(reference),
                "n_reco_jets": len(reco),
                "n_fake": int(len(reco) - (idx >= 0).sum()),
            })
    return rows
