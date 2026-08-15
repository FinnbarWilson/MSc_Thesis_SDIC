"""Does lowering the mask threshold recover MaskFormer's unclaimed energy?

    python -m scripts.scan_mask_recovery
    python -m scripts.scan_mask_recovery --events 100

THE QUESTION THIS ANSWERS. The jet-level comparison found MaskFormer carrying a flat ~2.5% energy
deficit that CLUE does not: it clusters 0.970 of the on-target energy against CLUE's 0.987, and its
jets come out light by almost exactly that amount. The deficit is cells whose highest mask
probability falls below the working point of 0.05, so nothing about the architecture requires them
to be lost -- the store keeps mask probabilities down to 0.02 and any working point above that is
re-derivable offline, with no GPU and no retraining.

So: sweep the mask threshold, and measure how much of the on-target energy ends up in some cluster.
If the deficit closes, the jet result is a working-point artefact and the fix is free. If it does
not, the energy is genuinely unreachable and the deficit is a property of the model.

WHAT IS DELIBERATELY NOT MEASURED HERE. Recovering energy is only half the trade -- a lower
threshold also admits cells the model did not want, which costs purity. This script reports the
energy recovered and the cluster sizes it comes with; it does not re-score. Read it as "is the
energy reachable at all", and re-run scoring at the chosen threshold before quoting any efficiency
or purity.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import describe, results_dir, settings, store_expectations, store_path
from src.io.event_store import EventStore

#: The store keeps masks down to 0.02, so that is the floor; 0.05 is the working point in use.
THRESHOLDS = (0.02, 0.03, 0.05, 0.10, 0.20, 0.50)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=0, help="cap the number of events (0 = all)")
    args = ap.parse_args()

    cfg = settings()
    obj = cfg["maskformer"]["object_threshold"]
    nominal = cfg["maskformer"]["mask_threshold"]
    print(describe())
    store = EventStore(store_path(), expect=store_expectations())
    n_events = args.events or len(store)
    print(f"\nsweeping the mask threshold over {min(n_events, len(store))} events "
          f"at object threshold {obj}\n")

    totals = {t: 0.0 for t in THRESHOLDS}
    cells = {t: 0 for t in THRESHOLDS}
    clusters = {t: 0 for t in THRESHOLDS}
    on_target = 0.0
    calo = 0.0

    for i, record in enumerate(store):
        if i >= n_events:
            break
        on_target += float(record.event_energy_on_target_calib)
        calo += float(record.event_energy_calib)
        e = record.energy_calib
        for t in THRESHOLDS:
            label, n = record.maskformer_labels(
                mask_threshold=t, object_threshold=obj,
                min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
            )
            claimed = label >= 0
            totals[t] += float(e[claimed].sum())
            cells[t] += int(claimed.sum())
            clusters[t] += int(n)

    rows = []
    for t in THRESHOLDS:
        rows.append({
            "mask_threshold": t,
            "nominal": t == nominal,
            "e_clustered_over_on_target": totals[t] / on_target,
            "e_clustered_over_calo": totals[t] / calo,
            "cells_claimed_per_event": cells[t] / min(n_events, len(store)),
            "clusters_per_event": clusters[t] / min(n_events, len(store)),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = results_dir() / "mask_recovery_scan.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    print(f"\nwrote {out}")
    print("\nCLUE clusters 0.9871 of the on-target energy; the row that matches or exceeds that is "
          "the threshold at which the jet-level deficit would close.")


if __name__ == "__main__":
    main()
