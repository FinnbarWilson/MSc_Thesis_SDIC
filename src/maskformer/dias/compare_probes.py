"""Compare the one-epoch probe arms against the one-epoch baseline, and say which one moved.

    python src/maskformer/dias/compare_probes.py

Run automatically by the last job in the overnight chain (see dias/README.md); safe to re-run by
hand at any time, since it only reads stores that already exist and skips the ones that do not.

WHAT IT MEASURES, AND WHY NOT EFFICIENCY

The headline metric (fraction of particles with >= 0.5 of their energy recovered) is far too
insensitive at one epoch: on run 48247 it moved 0.122 -> 0.136 at E > 20 GeV across seven epochs.
The quantity that actually distinguishes the hypotheses is the SHAPE of

    cells in the matched cluster    versus    the particle's true cell count

which on the unmodified config is flat-to-declining -- 6.4 cells for a 13-cell particle and 5.5 for
a 38-cell one. A fix should make it RISE. `slope` below is the ratio of cells recovered for
> 20 GeV particles to cells recovered for < 2 GeV ones, and > 1 means clusters grow with the shower.

READ THE BASELINE ROW, NOT A REMEMBERED NUMBER. This ratio is 0.86 on the FULLY TRAINED model
(seven epochs, 500 events) but 0.69 on the ONE-EPOCH baseline these arms are compared against, and
an earlier version of this docstring quoted the 0.86 as if it applied here. They are different
models measured over different windows. The comparison that means anything is arm-vs-baseline
within one run of this script, which is why the baseline is scored from its own store every time
rather than carried as a constant.

`merged` is the second number to read. It merges every fragment a particle dominates and asks how
often that clears 0.5 -- an oracle, so a ceiling rather than a method, but it separates "the model
does not know" from "the model knows and the assignment throws it away". On the 20 h baseline it
was 0.294 at E > 20 against CLUE's 0.224, which is what makes these arms worth running at all.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/xucapfwi/MSc_Thesis_SDIC")
sys.path.insert(0, str(REPO))

from src.config import settings, store_expectations  # noqa: E402
from src.evaluation.metrics import score_event  # noqa: E402
from src.io.event_store import EventStore  # noqa: E402

BINS = [0, 2, 5, 10, 20, 1e9]
LABELS = ["<2", "2-5", "5-10", "10-20", ">20"]

#: The baseline is epoch 0 of run 48247 -- the same one-epoch budget under the unmodified config,
#: dumped over the same 100 events. Not a purpose-built one-epoch run: its OneCycle schedule was
#: sized for seven epochs, so it sits mid-schedule at a high learning rate where the arms complete
#: their decay. That bias FAVOURS THE ARMS, bounded by roughly the whole seven-epoch improvement
#: (+1.6 cells at E > 20). It cannot manufacture a change of SHAPE, which is what is being read.
ARMS = {
    "baseline (ep0 of 48247)": REPO / "external/eventstore_ep000/ttbar_pu0_20250_20350_v2",
    "1: mask_attention off": REPO / "external/probes/overlay_probe_maskattn/store",
    "2: incidence head": REPO / "external/probes/overlay_probe_incidence/store",
    "3: exclusive target": REPO / "external/probes/overlay_probe_exclusive/store",
    "4a: posenc scale 0.5": REPO / "external/probes/overlay_probe_posenc05/store",
    "4b: posenc scale 0.2": REPO / "external/probes/overlay_probe_posenc02/store",
    "5: affinity head": REPO / "external/probes/overlay_probe_affinity/store",
}


def resolve(path: Path) -> Path | None:
    """A dumped store is one directory deeper, named for its event window."""
    if (path / "meta.json").exists():
        return path
    subs = sorted(p for p in path.glob("ttbar_pu0_*") if (p / "meta.json").exists()) if path.is_dir() else []
    return subs[0] if subs else None


def analyse(store_path: Path, cfg: dict, n_events: int) -> tuple[pd.DataFrame, dict] | None:
    store = EventStore(store_path, expect=store_expectations())
    m = cfg["metrics"]
    mt = cfg["maskformer"]["mask_threshold"]
    ot = cfg["maskformer"]["object_threshold"]

    parts, merged_rows = [], []
    for i in range(min(n_events, len(store))):
        rec = store[i]
        label, n_pred = rec.maskformer_labels(mt, ot, m["min_cluster_hits"])
        p, _c, _e = score_event(
            rec, label, n_pred, algo="mf",
            split_fraction=m["split_fraction"],
            min_overlap=m["min_overlap"],
            min_overlap_frac=m["min_overlap_frac"],
        )
        parts.append(p)

        # Oracle merge: every fragment in which this particle is the dominant owner.
        tl = np.asarray(rec.truth_label)
        n_part = int(rec.n_particles)
        if n_part == 0 or n_pred == 0:
            continue
        fac = np.divide(rec.energy_calib, np.maximum(rec.energy, 1e-12))
        dep = rec.truth_deposit * fac
        total = np.bincount(tl[tl >= 0], weights=dep[tl >= 0], minlength=n_part)
        keep = (label >= 0) & (tl >= 0)
        pair = np.bincount(
            label[keep].astype(np.int64) * n_part + tl[keep],
            weights=dep[keep], minlength=n_pred * n_part,
        ).reshape(n_pred, n_part)
        cluster_total = pair.sum(1, keepdims=True)
        mine = np.divide(pair, np.maximum(cluster_total, 1e-12)) > 0.5
        merged = (pair * mine).sum(0)
        good = total > 0
        merged_rows.append(pd.DataFrame({
            "sample_id": rec.sample_id,
            "particle_row": np.nonzero(good)[0],
            "merged": merged[good] / total[good],
        }))

    if not parts:
        return None
    particles = pd.concat(parts, ignore_index=True)
    particles["bin"] = pd.cut(particles.p_energy, BINS, labels=LABELS)
    if merged_rows:
        particles = particles.merge(pd.concat(merged_rows, ignore_index=True),
                                    on=["sample_id", "particle_row"], how="left")
    else:
        particles["merged"] = np.nan

    rows = []
    for b in LABELS:
        g = particles[(particles.bin == b) & particles.matched]
        a = particles[particles.bin == b]
        if not len(a):
            continue
        rows.append({
            "bin": b,
            "true_cells": g.n_hits.mean() if len(g) else np.nan,
            "recovered": (g.n_hits * g.eff_n).mean() if len(g) else np.nan,
            "eff": (a.eff_e >= 0.5).mean(),
            "merged": (a.merged >= 0.5).mean(),
        })
    table = pd.DataFrame(rows).set_index("bin")
    lo, hi = table.loc["<2", "recovered"], table.loc[">20", "recovered"]
    # THE COST COLUMNS MATTER AS MUCH AS THE GAIN, and the posenc arms are why they exist. Widening
    # the encoder's correlation length binds distant cells of one shower, but it also makes two
    # neighbours 0.1 apart look alike -- and 69% of target particles sit in jet cores (dr_min < 0.1)
    # against 2.9% above 20 GeV. An arm that lifts `eff_hi` while dropping `eff_core` has moved the
    # failure rather than fixed it, and reporting only the high-energy number would hide that.
    summary = {
        "slope": hi / lo if lo else np.nan,
        "eff_hi": table.loc[">20", "eff"],
        "merged_hi": table.loc[">20", "merged"],
        "eff_lo": table.loc["<2", "eff"],
        "eff_core": (particles.loc[particles.dr_min < 0.1, "eff_e"] >= 0.5).mean(),
        "eff_iso": (particles.loc[particles.dr_min > 0.2, "eff_e"] >= 0.5).mean(),
        "eff_all": (particles.eff_e >= 0.5).mean(),
        "n": len(particles),
    }
    return table, summary


def main() -> None:
    cfg = settings()
    n_events = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    results = {}

    for name, raw in ARMS.items():
        path = resolve(raw)
        if path is None:
            print(f"  ! no store for {name!r} at {raw} -- arm skipped")
            continue
        try:
            out = analyse(path, cfg, n_events)
        except Exception as exc:  # a failed arm must not take the report down
            print(f"  ! {name}: {type(exc).__name__}: {exc}")
            continue
        if out is None:
            print(f"  ! {name}: store had no scoreable events")
            continue
        results[name] = out

    if not results:
        raise SystemExit("No arms produced a store. Check external/slurm_logs/calo_probe_*.err")

    print("\n" + "=" * 78)
    print("CELLS IN THE MATCHED CLUSTER vs THE PARTICLE'S TRUE SIZE")
    print("A fix makes this column RISE with energy. The baseline falls: ~6.4 -> ~5.5.")
    print("=" * 78)
    for name, (table, _s) in results.items():
        print(f"\n{name}")
        print(f"  {'E bin':>7s} {'true cells':>11s} {'recovered':>10s} {'eff@0.5':>8s} {'merged':>7s}")
        for b, r in table.iterrows():
            print(f"  {b:>7s} {r.true_cells:11.1f} {r.recovered:10.1f} {r.eff:8.3f} {r.merged:7.3f}")

    print("\n" + "=" * 78)
    print("VERDICT   slope = cells recovered at >20 GeV / at <2 GeV   (baseline ~0.86)")
    print("          higher is better; > 1.0 means clusters finally grow with the shower")
    print("=" * 78)
    print(f"\n{'arm':>26s} {'slope':>7s} {'eff>20':>8s} {'merged>20':>10s} | {'eff<2':>7s} {'jetcore':>8s} {'isolated':>9s} {'all':>7s}")
    base = next(iter(results.values()))[1]
    for name, (_t, s) in results.items():
        flag = ""
        if name.startswith("baseline"):
            flag = "  <- reference"
        elif s["eff_hi"] > base["eff_hi"] * 1.10 and s["eff_core"] >= base["eff_core"] * 0.97:
            flag = "  <- GAIN WITHOUT COST"
        elif s["eff_hi"] > base["eff_hi"] * 1.10:
            flag = "  <- high-E up, jet core DOWN: the trade is real"
        elif s["slope"] > base["slope"] * 1.15:
            flag = "  <- shape moved"
        elif s["eff_all"] < base["eff_all"] * 0.9:
            flag = "  <- worse overall"
        print(f"{name:>26s} {s['slope']:7.2f} {s['eff_hi']:8.3f} {s['merged_hi']:10.3f} | "
              f"{s['eff_lo']:7.3f} {s['eff_core']:8.3f} {s['eff_iso']:9.3f} {s['eff_all']:7.3f}{flag}")

    print("\nCLUE, for reference, on the full 500-event window: eff@0.5 = 0.224 above 20 GeV.")
    print("An arm that raises `slope` is the mechanism confirmed and is worth a full 20 h run.")
    print("An arm that raises only `merged` fixed the fragmentation but not the coverage.")
    print("Read dias/README.md before deciding; do not promote an arm on eff>20 alone, which is")
    print("too noisy at one epoch to separate the hypotheses.")


if __name__ == "__main__":
    main()
