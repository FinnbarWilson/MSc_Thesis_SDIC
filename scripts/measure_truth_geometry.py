"""Measure properties of the truth partition, for the methodology chapter.

    python -m scripts.measure_truth_geometry [--events 100]

None of these depend on a clustering method; each is a statement about the target definition and
the detector, and each is quoted in the report:

  1. what the exclusive truth partition discards, in associations and in energy;
  2. what fraction of targets span a subsystem boundary, which is what justifies CLUE's
     cross-subsystem linking stage;
  3. what fraction of calorimeter energy belongs to any target, whose complement is a ceiling on
     every energy-weighted metric.

Isolation is summarised separately from the scored particle tables, where ``dr_min`` already
exists. Writes ``truth_geometry.csv`` and ``truth_isolation.csv``.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import describe, results_dir, store_expectations, store_path
from src.io.event_store import EventStore
from src.plotting import thesis as th

SUBSYSTEMS = ("ecb", "ece", "hcb", "hce")


def measure_store(store: EventStore, n_events: int) -> dict[str, float]:
    """Walk the multi-owner CSR and the exclusive partition together."""
    assoc_total = assoc_kept = 0
    energy_total = energy_kept = 0.0
    n_targets = n_spanning = 0
    energy_targets = energy_spanning = energy_minority = 0.0
    calo_total = calo_on_target = 0.0
    cells_owned_total = cells_shared_total = 0
    subsystem_hist: dict[int, int] = {}

    for i, record in enumerate(store):
        if i >= n_events:
            break

        calo_total += float(record.event_energy_calib)
        calo_on_target += float(record.event_energy_on_target_calib)

        if record.n_particles == 0 or record.truth_indices.size == 0:
            continue

        # --- 1. exclusive-partition cost ---
        # The CSR is particle-major and `truth_incidence` is the particle's share of the cell, so
        # its contribution is share x E_calib. An association survives iff the cell's winner is
        # this same particle.
        rows = np.repeat(np.arange(record.n_particles), np.diff(record.truth_indptr))
        cols = record.truth_indices
        contrib = record.truth_incidence * record.energy_calib[cols]
        winner = record.truth_label[cols] == rows

        assoc_total += int(cols.size)
        assoc_kept += int(winner.sum())
        energy_total += float(contrib.sum())
        energy_kept += float(contrib[winner].sum())

        # How often the rule has to choose at all; counted over cells some target owns, since a
        # cell owned by nobody is not part of the partition either way.
        contributors = np.bincount(cols, minlength=record.n_hits)
        owned_cells = contributors > 0
        cells_owned_total += int(owned_cells.sum())
        cells_shared_total += int((contributors > 1).sum())

        # --- 2. subsystem spanning, on the exclusive partition ---
        # That partition is what both methods are scored against, so it is those targets whose
        # straddling matters.
        owned = record.truth_label >= 0
        if not owned.any():
            continue
        labels = record.truth_label[owned]
        subs = record.subsystem[owned]
        deposit = record.truth_deposit[owned]

        # distinct subsystems per particle, via a particle x subsystem occupancy table
        occupancy = np.zeros((record.n_particles, len(SUBSYSTEMS)), dtype=bool)
        occupancy[labels, subs] = True
        n_sub = occupancy.sum(axis=1)

        per_particle_e = np.bincount(labels, weights=deposit, minlength=record.n_particles)
        present = n_sub > 0

        # Energy outside each target's dominant subsystem: what a per-subsystem clusterer puts
        # at risk, and much smaller than the energy of spanning targets.
        by_sub = np.zeros((record.n_particles, len(SUBSYSTEMS)))
        np.add.at(by_sub, (labels, subs), deposit)
        energy_minority += float((by_sub.sum(axis=1) - by_sub.max(axis=1))[present].sum())

        n_targets += int(present.sum())
        n_spanning += int((n_sub > 1).sum())
        energy_targets += float(per_particle_e[present].sum())
        energy_spanning += float(per_particle_e[n_sub > 1].sum())
        for k in np.unique(n_sub[present]):
            subsystem_hist[int(k)] = subsystem_hist.get(int(k), 0) + int((n_sub == k).sum())

    return {
        "events": min(n_events, len(store)),
        "assoc_total": assoc_total,
        "cells_owned": cells_owned_total,
        "cells_shared_frac": cells_shared_total / cells_owned_total if cells_owned_total else np.nan,
        "assoc_kept_frac": assoc_kept / assoc_total if assoc_total else np.nan,
        "energy_kept_frac": energy_kept / energy_total if energy_total else np.nan,
        "targets": n_targets,
        "spanning_frac": n_spanning / n_targets if n_targets else np.nan,
        "spanning_energy_frac": energy_spanning / energy_targets if energy_targets else np.nan,
        "minority_energy_frac": energy_minority / energy_targets if energy_targets else np.nan,
        "coverage_frac": calo_on_target / calo_total if calo_total else np.nan,
        **{f"targets_in_{k}_subsystems": v for k, v in sorted(subsystem_hist.items())},
    }


def summarise_isolation() -> pd.DataFrame | None:
    """Median angular separation to the nearest other target, per pT bin, or None if unscored.

    Read from a scored particle table rather than recomputed: `dr_min` is already there, and any
    single method's rows give the truth geometry.
    """
    path = results_dir(create=False) / "particles_maskformer.parquet"
    if not path.exists():
        return None
    p = pd.read_parquet(path, columns=["p_pt", "dr_min", "eff_e", "sample_id"])
    p = p[np.isfinite(p.dr_min)]
    if p.empty:
        return None
    idx = np.digitize(p.p_pt.to_numpy(), th.E_BINS) - 1
    rows = []
    for k in range(len(th.E_BINS) - 1):
        sel = p[idx == k]
        if sel.empty:
            continue
        isolated = sel[sel.dr_min > 0.2]
        rows.append({
            "pt_lo": th.E_BINS[k],
            "pt_hi": th.E_BINS[k + 1],
            "n": len(sel),
            "median_dr_min": float(sel.dr_min.median()),
            "frac_isolated_dr_gt_0.2": float((sel.dr_min > 0.2).mean()),
            "eff_all": float((sel.eff_e >= 0.5).mean()),
            "eff_isolated": float((isolated.eff_e >= 0.5).mean()) if len(isolated) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=0, help="cap the number of events (0 = all)")
    ap.add_argument("--store", default="store", choices=["store", "tune_store"])
    args = ap.parse_args()

    print(describe())
    store = EventStore(store_path(args.store), expect=store_expectations())
    n = args.events or len(store)
    print(f"\nmeasuring over {min(n, len(store))} events from {store.root}\n")

    m = measure_store(store, n)

    print("1. Exclusive truth partition: what the winner-takes-the-cell rule discards")
    print(f"   associations retained   {m['assoc_kept_frac']:.4f}   ({m['assoc_total']:,} total)")
    print(f"   target energy retained  {m['energy_kept_frac']:.4f}")
    print(f"   owned cells with >1 contributing target  {m['cells_shared_frac']:.4f}"
          f"   ({m['cells_owned']:,} owned cells)")
    print()
    print("2. TARGETS SPANNING A SUBSYSTEM BOUNDARY")
    print(f"   targets                 {m['targets']:,}")
    print(f"   spanning >1 subsystem   {m['spanning_frac']:.4f}")
    print(f"   their share of energy   {m['spanning_energy_frac']:.4f}")
    print(f"   energy outside each target's dominant subsystem  {m['minority_energy_frac']:.4f}")
    for k in (1, 2, 3, 4):
        key = f"targets_in_{k}_subsystems"
        if key in m:
            print(f"     in {k} subsystem(s)     {m[key]:,} ({m[key] / m['targets']:.4f})")
    print()
    print("3. TARGET ENERGY COVERAGE")
    print(f"   calo energy on target   {m['coverage_frac']:.4f}   (floor {1 - m['coverage_frac']:.4f})")

    out = results_dir() / "truth_geometry.csv"
    pd.DataFrame([m]).to_csv(out, index=False, float_format="%.6g")
    print(f"\nwrote {out}")

    iso = summarise_isolation()
    if iso is not None:
        print("\n4. TARGET ISOLATION, per truth pT bin")
        print(iso.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        iso_out = results_dir() / "truth_isolation.csv"
        iso.to_csv(iso_out, index=False, float_format="%.6g")
        print(f"\nwrote {iso_out}")
    else:
        print("\n4. Target isolation: skipped, no scored particle table")


if __name__ == "__main__":
    main()
