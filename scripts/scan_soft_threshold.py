"""Trace the soft metric against the mask threshold, which is what actually sets cell sharing.

The mask threshold does a different job in each metric, and that is the reason this scan
exists. In the exclusive head-to-head, contested cells are resolved by highest logit whatever
the threshold is, so it mostly controls cluster size. In the soft metric nothing resolves the
contest: every query above threshold keeps its claim and the claims are normalised per cell,
so the threshold directly controls **how many ways each cell is divided**.

At the nominal 0.5 the model claims each cell 2.08 times against the truth's 1.22. Because the
overlap is ``E_i * min(t_ai, w_ci)``, a spurious second claimant halves the true owner's weight
and caps its credit at 0.5 of a cell it may own 0.9 of -- which is why the model's soft
efficiency lands below its exclusive one. The scan was built to test whether that is a working
point chosen badly or a property of the model, since those are very different claims to make.

**It is the model.** The measured answer is flat and one-directional, with no sweet spot:

    threshold      0.02    0.5 (nominal)    0.95
    mean soft eff  0.350   0.322            0.300
    soft purity    0.273   0.337            0.412
    claims/cell    2.50    2.08             1.85     (truth: 1.22)

Efficiency falls monotonically as the threshold rises, so the best value sits at the store's
0.02 floor and is bounded by this scan rather than located by it. More importantly, the third
row never approaches the truth: even at 0.95, where masks are shedding genuine cells and
efficiency is at its worst, the model still divides each cell 1.85 ways against the truth's
1.22. The overlap is not made of marginal claims that a threshold can trim -- it is made of
confident ones, and no threshold reaches the truth's sharing rate.

That is consistent with how the mask head is trained. The architecture applies an element-wise
sigmoid per (query, cell) rather than a softmax over queries, precisely so a cell *can* belong
to several objects. Nothing in the loss constrains the claims on one cell to sum to anything,
so the model is never taught what a cell's division should add up to, and there is no reason
its probabilities would be calibrated as energy fractions. What is left is a genuine
efficiency-purity trade with no optimum in efficiency.

The scan is free. The store keeps masks down to a probability of 0.02, so every point is
re-derived offline with no GPU and no re-inference. CLUE does not appear here: its clusters
never overlap, so nothing in this scan can move it.

    python -m scripts.scan_soft_threshold --events 200
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR, settings, store_expectations, store_path
from src.evaluation.soft import score_event_soft, sharing_diagnostics
from src.io.event_store import EventStore
from src.plotting import style

#: Runs from the store's own 0.02 floor up to a value tight enough to start losing real cells.
#: The lower end is a hard limit rather than a choice -- masks below 0.02 were never written --
#: so an optimum sitting there is bounded by this scan rather than located by it, and must be
#: reported that way.
THRESHOLD_SCAN = (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "soft_threshold_scan.parquet")
    parser.add_argument("--figure", type=Path, default=FIGURES_DIR / "soft_threshold_scan")
    args = parser.parse_args()

    cfg = settings()
    wp = cfg["metrics"]["working_points"][0]
    nominal = cfg["maskformer"]["mask_threshold"]
    object_threshold = cfg["maskformer"]["object_threshold"]

    store = EventStore(store_path(), expect=store_expectations())
    records = [store[i] for i in range(min(args.events, len(store)))]
    print(f"scanning {len(THRESHOLD_SCAN)} mask thresholds on {len(records)} events")

    rows = []
    for threshold in THRESHOLD_SCAN:
        tables, diagnostics, cluster_counts = [], [], []
        for record in records:
            cluster, cell, weight, n = record.maskformer_soft_masks(
                threshold, object_threshold, cfg["metrics"]["min_cluster_hits"]
            )
            tables.append(
                score_event_soft(record, cluster, cell, weight, n, "maskformer",
                                 min_overlap_frac=cfg["metrics"]["min_overlap_frac"])
            )
            diagnostics.append(sharing_diagnostics(record, cell))
            cluster_counts.append(n)

        table = pd.concat(tables, ignore_index=True)
        shared = pd.DataFrame(diagnostics).mean(numeric_only=True)
        impossible = table[table["no_exclusive_cell"]]
        rows.append({
            "mask_threshold": threshold,
            "nominal": threshold == nominal,
            "eff_soft": float((table["eff_e"] >= wp).mean()),
            "eff_soft_mean": float(table["eff_e"].mean()),
            "pur_soft": float((table["pur_e"] >= wp).mean()),
            "pur_soft_mean": float(table["pur_e"].mean()),
            "match_rate": float(table["matched"].mean()),
            "claims_per_cell": float(shared["claims_per_cell"]),
            "truth_owners_per_cell": float(shared["truth_owners_per_cell"]),
            "impossible_recovered": float((impossible["eff_e"] >= wp).mean()) if len(impossible) else 0.0,
            "clusters_per_event": float(np.mean(cluster_counts)),
        })
        print(
            f"  mask {threshold:<5} eff {rows[-1]['eff_soft']:.3f} (mean {rows[-1]['eff_soft_mean']:.3f})"
            f"  pur {rows[-1]['pur_soft']:.3f}  claims/cell {rows[-1]['claims_per_cell']:.2f}"
            f"  impossible {rows[-1]['impossible_recovered']:.3f}",
            flush=True,
        )

    scan = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scan.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")

    best = scan.loc[scan["eff_soft_mean"].idxmax()]
    at_nominal = scan[scan["nominal"]]
    print(f"\nbest mean soft efficiency {best['eff_soft_mean']:.3f} at mask threshold {best['mask_threshold']}")
    if len(at_nominal):
        print(f"nominal ({nominal}) gives {float(at_nominal['eff_soft_mean'].iloc[0]):.3f}")
    print(f"truth shares {scan['truth_owners_per_cell'].iloc[0]:.2f} owners per cell; the "
          f"threshold matching that is where over-claiming stops")

    style.apply()
    _draw(scan, wp, args.figure)


def _draw(scan: pd.DataFrame, working_point: float, stem: Path) -> None:
    """Three panels: what the threshold costs, what it buys, and why."""
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))
    colour = style.colour("maskformer")
    nominal = scan[scan["nominal"]]

    axes[0].plot(scan["mask_threshold"], scan["eff_soft_mean"], color=colour, marker="o", label="mean")
    axes[0].plot(scan["mask_threshold"], scan["eff_soft"], color=colour, marker="s",
                 linestyle="--", alpha=0.7, label=f"fraction $\\geq$ {working_point}")
    axes[0].set_ylabel("soft energy efficiency")
    axes[0].set_title("Efficiency against mask threshold")

    axes[1].plot(scan["mask_threshold"], scan["pur_soft_mean"], color=colour, marker="o", label="mean")
    axes[1].plot(scan["mask_threshold"], scan["pur_soft"], color=colour, marker="s",
                 linestyle="--", alpha=0.7, label=f"fraction $\\geq$ {working_point}")
    axes[1].set_ylabel("soft energy purity")
    axes[1].set_title("Purity against mask threshold")

    # The diagnostic panel: the threshold's real job here is setting how often a cell is
    # divided, and the truth line is the value it would have to reach to stop over-claiming.
    axes[2].plot(scan["mask_threshold"], scan["claims_per_cell"], color=colour, marker="o", label="MaskFormer")
    axes[2].axhline(
        scan["truth_owners_per_cell"].iloc[0], color="#555555", linestyle=":",
        label=f"truth ({scan['truth_owners_per_cell'].iloc[0]:.2f})",
    )
    axes[2].set_ylabel("claims per claimed cell")
    axes[2].set_title("How often a cell is divided")

    for ax in axes:
        ax.set_xlabel("mask threshold")
        if len(nominal):
            ax.axvline(float(nominal["mask_threshold"].iloc[0]), color="#999999", linewidth=0.9, alpha=0.6)
        ax.legend()
    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(0.0, 1.05)

    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"))
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    main()
