"""Anti-k_t jets built from clusters, and the same jets built from the truth partition.

WHY JETS, AND WHAT THE REFERENCE IS

Per-cluster efficiency and purity say how well a method groups cells. They do not say whether the
error survives into an observable, and jets are where it either does or does not: a cluster that
loses a shower's halo makes a slightly soft particle candidate, and a jet is a sum of many such
candidates, so the per-cluster errors can cancel, accumulate, or push a jet across a threshold.

The reference is the TRUTH PARTITION, not the generator particles. For every target particle the
cells it actually owns are summed into a four-vector and those are clustered -- so the reference
jets are exactly what a PERFECT clusterer would produce from this calorimeter, given these cells
and these cuts. Comparing against generator jets instead would fold in three other effects that
have nothing to do with clustering: zero-suppression, the pt/eta target selection, and the
calorimeter's own sampling resolution. Those belong to the detector, not to the algorithm under
test, and including them would flatter both methods equally while hiding the thing being measured.

FOUR-VECTORS FROM CELLS

A calorimeter cell measures energy and position, not momentum. Each cell is therefore treated as a
massless particle travelling from the interaction point through the cell centre:

    p = E_calib * (x, y, z) / |(x, y, z)|,    E = E_calib

and a cluster's four-vector is the sum over its cells. This is the standard massless-cell
assumption used to feed calorimeter clusters into jet finding, and it makes the reference and the
methods commensurate: both are sums over the same cells with the same rule, so any difference in
the jets comes from WHICH cells were grouped together, which is the question.

CHOICES, AND WHY

anti-k_t at R = 0.4, the LHC default for a ttbar final state: it is infrared and collinear safe,
and it yields the regular, close-to-circular boundaries that make a jet's area well defined
(Cacciari, Salam & Soyez). Jets are kept above 25 GeV, which is the usual analysis threshold for
ttbar and the value this study reports.

Reco jets are matched to reference jets within dR < 0.3, chosen smaller than R so a reco jet
cannot be equidistant between two adjacent reference jets, and the closest is taken. A reference
jet with no match inside that cone is a LOST jet; a reco jet matching no reference jet is a FAKE.
Both are reported, because a method can buy response by splitting one jet into two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: LHC default for ttbar. See the module docstring.
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

    Greedy by reference pt, which is the order `run_antikt` already returns, and a reco jet is
    consumed once taken. A global assignment would be defensible too, but jets inside a cone this
    small are rarely contested and greedy-by-pt matches how an analysis would do it.
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
