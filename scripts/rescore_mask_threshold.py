"""Re-score the model at several mask thresholds, with jets rebuilt.

    python -m scripts.rescore_mask_threshold [--thresholds 0.02 0.05] [--events 200]

The energy MaskFormer leaves unclustered is reachable at a lower mask threshold, but what that
costs cannot be inferred: efficiency and purity are counts of threshold crossings, so no bound
on the contaminating energy bounds the purity loss. Each row re-derives the labels, re-runs the
scorer and rebuilds the anti-k_t jets.

Not a re-tuning: the object threshold is held at its tuned value and only the mask axis moves.
CLUE is scored once and repeated on every row, so each row is a complete comparison. Writes
``mask_threshold_rescore.csv``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.clue.pipeline import cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation import jets as jt
from src.evaluation.metrics import score_event
from src.io.event_store import EventStore

#: 0.02 is the store's probability floor, so it is the lowest threshold recoverable offline.
THRESHOLDS = (0.02, 0.03, 0.05, 0.10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    ap.add_argument("--events", type=int, default=0, help="cap the number of events (0 = all)")
    args = ap.parse_args()

    cfg = settings()
    mf = cfg["maskformer"]
    metrics = cfg["metrics"]
    obj = mf["object_threshold"]
    nominal = mf["mask_threshold"]
    wp = metrics["working_points"][0]

    print(describe())
    store = EventStore(store_path(), expect=store_expectations())
    n_events = args.events or len(store)
    n_events = min(n_events, len(store))

    clue_params = {k: v["parameters"] for k, v in
                   json.loads((results_dir() / "clue_parameters.json").read_text())["subsystems"].items()}

    print(f"\nre-scoring {len(args.thresholds)} mask thresholds over {n_events} events "
          f"at object threshold {obj}, working point {wp}\n", flush=True)

    per_t = {t: {"particles": [], "clusters": [], "jets": [], "clustered": 0.0} for t in args.thresholds}
    clue = {"particles": [], "clusters": [], "jets": []}
    on_target = 0.0

    for i, record in enumerate(store):
        if i >= n_events:
            break
        on_target += float(record.event_energy_on_target_calib)
        energy = record.energy_calib

        labels = {}
        for t in args.thresholds:
            label, n = record.maskformer_labels(
                mask_threshold=t, object_threshold=obj,
                min_cluster_hits=metrics["min_cluster_hits"],
            )
            labels[t] = (label, n)
            per_t[t]["clustered"] += float(energy[label >= 0].sum())
            p, c, _ = score_event(record, label, n, algo="maskformer",
                                  split_fraction=metrics["split_fraction"],
                                  min_overlap=metrics["min_overlap"],
                                  min_overlap_frac=metrics["min_overlap_frac"])
            per_t[t]["particles"].append(p)
            per_t[t]["clusters"].append(c)

        cl_label, cl_n = cluster_event(record, clue_params, coords=cfg["clue"]["coords"],
                                       backend=cfg["clue"]["backend"],
                                       min_cluster_hits=metrics["min_cluster_hits"],
                                       link_radius=cfg["clue"].get("link_radius", 0.0))
        p, c, _ = score_event(record, cl_label, cl_n, algo="clue",
                              split_fraction=metrics["split_fraction"],
                              min_overlap=metrics["min_overlap"],
                              min_overlap_frac=metrics["min_overlap_frac"])
        clue["particles"].append(p)
        clue["clusters"].append(c)

        # One call per event for every threshold and for CLUE, so all of them see the same
        # reference jets.
        by_method = {f"t{t}": labels[t] for t in args.thresholds}
        by_method["clue"] = (cl_label, cl_n)
        rows = jt.event_rows(record, by_method, cfg["dataset"]["active"])
        for r in rows:
            (clue if r["algo"] == "clue" else per_t[float(r["algo"][1:])])["jets"].append(r)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_events} events", flush=True)

    def summarise(particles, clusters, jets, clustered=None):
        p = pd.concat(particles, ignore_index=True)
        c = pd.concat(clusters, ignore_index=True)
        j = pd.DataFrame(jets)
        m = j[j.matched]
        return {
            "efficiency": float((p.eff_e >= wp).mean()),
            "purity_all": float((c.pur_e >= wp).mean()),
            "purity_matched": float((c[c.matched].pur_e >= wp).mean()),
            "fake_rate": float((~c.matched).mean()),
            "frag_frac": float(p.frag_frac.mean()),
            "clusters_per_event": len(c) / n_events,
            "e_clustered_frac": (clustered / on_target) if clustered is not None else np.nan,
            "jet_sum_ratio": float(m.reco_pt.sum() / m.ref_pt.sum()),
            "jet_median_ratio": float(np.median(m.reco_pt / m.ref_pt)),
            "jets_matched": int(len(m)),
        }

    rows = []
    for t in args.thresholds:
        d = per_t[t]
        rows.append({"algo": "maskformer", "mask_threshold": t, "nominal": t == nominal,
                     **summarise(d["particles"], d["clusters"], d["jets"], d["clustered"])})
    rows.append({"algo": "clue", "mask_threshold": np.nan, "nominal": True,
                 **summarise(clue["particles"], clue["clusters"], clue["jets"])})

    table = pd.DataFrame(rows)
    out = results_dir() / "mask_threshold_rescore.csv"
    table.to_csv(out, index=False, float_format="%.6g")

    show = ["algo", "mask_threshold", "efficiency", "purity_all", "purity_matched", "fake_rate",
            "clusters_per_event", "e_clustered_frac", "jet_sum_ratio", "jet_median_ratio"]
    print("\n" + table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out}")

    mf_rows = table[table.algo == "maskformer"].set_index("mask_threshold")
    cl = table[table.algo == "clue"].iloc[0]
    if nominal in mf_rows.index and len(mf_rows) > 1:
        best = mf_rows.jet_sum_ratio.idxmax()
        print(f"\nAt the working point ({nominal:g}) the jet ratio is "
              f"{mf_rows.loc[nominal, 'jet_sum_ratio']:.4f} against CLUE's {cl.jet_sum_ratio:.4f}.")
        print(f"The best mask threshold for jets is {best:g}, at {mf_rows.loc[best, 'jet_sum_ratio']:.4f}, "
              f"and it costs {mf_rows.loc[nominal, 'purity_all'] - mf_rows.loc[best, 'purity_all']:+.4f} "
              f"in all-cluster purity and "
              f"{mf_rows.loc[best, 'efficiency'] - mf_rows.loc[nominal, 'efficiency']:+.4f} in efficiency.")


if __name__ == "__main__":
    main()
