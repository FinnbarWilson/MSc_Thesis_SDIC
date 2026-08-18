"""What is in the ColliderML dataset, before any of it is clustered.

    python -m scripts.make_dataset_figures --dataset pu0 --events 200

Writes figures/<dataset>/dataset_features.png|pdf, a per-panel breakdown by particle class,
and results/<dataset>/dataset_features_selection.csv, the table of what each reconstructability
cut costs. The per-particle features are cached in results/<dataset>/dataset_features.parquet,
so re-styling the figure never re-reads the parquet shards.

WHY THIS DOES NOT COME FROM THE EVENT STORE

Every other figure in this repo is built from the store, because the store is the guarantee
that both methods saw the same cells. This one cannot be, and the reason is the whole point of
the figure. The store holds the TARGET set: particles that already passed particle_selection
and were already merged onto their shower's calorimeter-entering ancestor. Drawn from it,
these panels would show only that the cuts did what they say. The distributions the cuts were
CHOSEN from live one step earlier, in the raw parquet, which is what src/io/colliderml.py
reads. The cut values are drawn on top as vertical lines so the two are visible together.

WHAT IS PLOTTED

One row per particle that owns at least one cell surviving zero-suppression -- the "post event
cleaning" set. A particle with no surviving cell is invisible to any calorimeter clustering
algorithm whatever its momentum, and at pu200 there are 125,000 of them per event.

There is no tracker panel. ColliderML ships silicon hits and this repo never downloaded them:
the comparison is a calorimeter problem, so the analogues of the Si/VTXD/muon-chamber panels
are the four calorimeter subsystems instead.

THE PARTICLE DEFINITION

Each in-calorimeter secondary is merged onto the particle whose shower it belongs to, which is
the thesis's target definition. Without that collapse there is one row per Geant fragment, a
genuinely different picture of the same events -- 325 against 1,936 depositing particles per pu0
event. The collapse itself lives in src/io/colliderml.py and is tested both ways.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATASETS, FIGURES_ROOT, RESULTS_ROOT, active_dataset, settings_for
from src.io import colliderml as cml
from src.plotting import style

#: One panel each, in reading order: kinematics, then where the particle came from and how
#: crowded it is, then what it left in the calorimeter.
#:
#: `bins` is either an array or a callable given the column, for the counts, whose upper edge
#: has to follow the data rather than a guess -- a pu200 shower core reaches four figures.
PANELS: tuple[dict, ...] = (
    {"column": "pt", "label": r"Particle $p_\mathrm{T}$ [GeV]", "bins": np.geomspace(1e-2, 3e2, 45), "xscale": "log"},
    {"column": "eta", "label": r"Particle $\eta$", "bins": np.linspace(-4.5, 4.5, 45), "xscale": "linear"},
    {"column": "phi", "label": r"Particle $\phi$", "bins": np.linspace(-np.pi, np.pi, 45), "xscale": "linear"},
    {"column": "energy", "label": "Particle energy [GeV]", "bins": np.geomspace(1e-2, 1e3, 45), "xscale": "log"},
    # The upper edge follows the data rather than being fixed, because the two particle
    # definitions put it in different places: collapsed, every vertex is outside the
    # calorimeter face by construction and the axis stops just past 1252 mm; as fragments, the
    # secondaries are born throughout the calorimeter and a fixed 1400 mm edge would silently
    # crop most of them.
    {"column": "vertex_r", "label": "Particle vertex $r$ [mm]", "bins": "span", "xscale": "linear"},
    {"column": "dr_min", "label": r"Particle angular isolation $\Delta R_\mathrm{min}$", "bins": np.geomspace(1e-4, 3.0, 45), "xscale": "log"},
    {"column": "n_hits_ecal", "label": "Particle num. ECAL hits", "bins": "count", "xscale": "count"},
    {"column": "n_hits_hcal", "label": "Particle num. HCAL hits", "bins": "count", "xscale": "count"},
    {"column": "n_calohits", "label": "Particle num. calo hits", "bins": "count", "xscale": "count"},
    {"column": "energy_ecal_calib", "label": "Particle total calibrated ECAL energy [GeV]", "bins": np.geomspace(1e-3, 1e2, 45), "xscale": "log"},
    {"column": "energy_hcal_calib", "label": "Particle total calibrated HCAL energy [GeV]", "bins": np.geomspace(1e-3, 1e2, 45), "xscale": "log"},
    {"column": "energy_calo_calib", "label": "Particle total calibrated calo energy [GeV]", "bins": np.geomspace(1e-3, 1e2, 45), "xscale": "log"},
)

#: Per-subsystem companion figure: the same hit counts and energies, split barrel from endcap.
SUBSYSTEM_LABELS = {"ecb": "ECAL barrel", "ece": "ECAL endcap", "hcb": "HCAL barrel", "hce": "HCAL endcap"}


def count_bins(values: np.ndarray) -> np.ndarray:
    """Log-spaced integer bins covering a hit count, with a bin for zero.

    Hit counts span 0 to ~4,000 at pu200, so linear bins spend forty of their forty-five on an
    empty tail. Log bins need a home for the zeros, which is the leading [0, 1) bin here.
    """
    top = max(int(np.max(values)) if np.size(values) else 1, 2)
    edges = np.unique(np.round(np.geomspace(1.0, top + 1, 40)))
    return np.concatenate([[0.0], edges])


def span_bins(values: np.ndarray) -> np.ndarray:
    """Linear bins from zero to just past the data, rounded to a readable edge."""
    top = float(np.max(values)) if np.size(values) else 1.0
    step = 10.0 ** np.floor(np.log10(max(top, 1e-9)))
    return np.linspace(0.0, np.ceil(top / step) * step, 45)


_AUTO_BINS = {"count": count_bins, "span": span_bins}


def draw_panel(ax, table: pd.DataFrame, panel: dict, classes: list[str], cuts: dict) -> None:
    values_all = table[panel["column"]].to_numpy(dtype=np.float64)
    finite = values_all[np.isfinite(values_all)]
    bins = _AUTO_BINS[panel["bins"]](finite) if isinstance(panel["bins"], str) else panel["bins"]

    ax.hist(finite, bins=bins, histtype="step", color=style.particle_class_colour("all"), label=style.particle_class_label("all"), lw=1.1)
    for name in classes:
        values = table.loc[table["particle_class"] == name, panel["column"]].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, histtype="step", color=style.particle_class_colour(name), label=style.particle_class_label(name), lw=0.9)

    # "count" is symlog rather than log: a hit count runs 0 to a few thousand, and the zero
    # -- a particle that reached the HCAL and nothing else, or the ECAL and nothing else --
    # is a real and populated value that a log axis cannot draw at all.
    if panel["xscale"] == "count":
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.35)
    else:
        ax.set_xscale(panel["xscale"])
    ax.set_yscale("log")
    ax.set_xlabel(panel["label"])
    ax.set_ylabel("Particles")
    ax.set_xlim(bins[0], bins[-1])

    # The reconstructability cuts, drawn where they land. This is what makes the figure a
    # justification of the target definition rather than a description of the file.
    for position in cuts.get(panel["column"], ()):
        ax.axvline(position, color="#474747", ls="--", lw=0.8, alpha=0.8, zorder=1)


def selection_cuts(cfg: dict) -> dict[str, tuple[float, ...]]:
    """Where ``particle_selection`` lands on each panel's axis."""
    selection = cfg["particle_selection"]
    max_abs_eta = float(selection["particle_max_abs_eta"])
    return {
        "pt": (float(selection["particle_min_pt"]),),
        "eta": (-max_abs_eta, max_abs_eta),
        "n_calohits": (float(selection["particle_min_num_calohits"]),),
    }


def figure_features(table: pd.DataFrame, classes: list[str], cuts: dict, out: Path) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 8.4))
    for ax, panel in zip(axes.ravel(), PANELS, strict=True):
        draw_panel(ax, table, panel, classes, cuts)
    style.clear_panel_legends(axes.ravel())
    style.panel_labels(axes.ravel())
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    style.legend_below(fig, ncol=min(len(classes) + 1, 7), y=0.045)
    return save(fig, out)


def figure_subsystems(table: pd.DataFrame, classes: list[str], out: Path) -> Path:
    """Hit count and calibrated energy per readout subsystem, barrel against endcap.

    The main figure collapses ECAL to a group because that is what the cuts are written in.
    CLUE is tuned per subsystem, so the barrel/endcap split is the granularity its parameters
    actually see, and it is not obvious from the grouped panels that the two halves of one
    calorimeter carry different occupancies.
    """
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.6))
    for column, subsystem in enumerate(SUBSYSTEM_LABELS):
        panel_hits = {"column": f"n_hits_{subsystem}", "label": f"Num. {SUBSYSTEM_LABELS[subsystem]} hits", "bins": "count", "xscale": "count"}
        draw_panel(axes[0, column], table, panel_hits, classes, {})

        panel_energy = {
            "column": f"energy_{subsystem}_calib",
            "label": f"{SUBSYSTEM_LABELS[subsystem]} calibrated energy [GeV]",
            "bins": np.geomspace(1e-3, 1e2, 45),
            "xscale": "log",
        }
        draw_panel(axes[1, column], table, panel_energy, classes, {})

    style.clear_panel_legends(axes.ravel())
    style.panel_labels(axes.ravel())
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    style.legend_below(fig, ncol=min(len(classes) + 1, 7), y=0.07)
    return save(fig, out)


def selection_table(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """What each reconstructability cut costs, per particle class.

    The cuts are applied cumulatively and in the order they are written in the config, so each
    row is the survivors of everything above it. Reported as counts per event and as the
    fraction of calibrated calorimeter energy still owned by a surviving particle -- the second
    is the one that matters, because a cut that removes 60% of the particles and 2% of the
    energy is cheap and the raw count does not say so.
    """
    selection = cfg["particle_selection"]
    n_events = int(table["event_id"].nunique())
    total_energy = float(table["energy_calo_calib"].sum())

    stages = [
        ("deposits >= 1 cell", np.ones(len(table), dtype=bool)),
        (f"pT >= {selection['particle_min_pt']} GeV", table["pt"].to_numpy() >= selection["particle_min_pt"]),
        (f"|eta| <= {selection['particle_max_abs_eta']}", np.abs(table["eta"].to_numpy()) <= selection["particle_max_abs_eta"]),
        (f"num calo hits >= {selection['particle_min_num_calohits']}", table["n_calohits"].to_numpy() >= selection["particle_min_num_calohits"]),
    ]

    rows = []
    keep = np.ones(len(table), dtype=bool)
    for name, mask in stages:
        keep = keep & mask
        surviving = table[keep]
        row = {
            "stage": name,
            "particles_per_event": len(surviving) / max(n_events, 1),
            "energy_fraction": float(surviving["energy_calo_calib"].sum()) / total_energy if total_energy else np.nan,
        }
        counts = surviving["particle_class"].value_counts()
        for name_class in table["particle_class"].cat.categories:
            row[f"n_{name_class}_per_event"] = float(counts.get(name_class, 0)) / max(n_events, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, default=active_dataset(), help="which pileup condition to describe")
    ap.add_argument("--events", type=int, default=200, help="events to read; pu200 costs about 3 s each")
    ap.add_argument("--raw-root", type=Path, default=cml.DEFAULT_RAW_ROOT, help="directory of ttbar_<dataset>_<collection> parquet shards")
    ap.add_argument("--rebuild", action="store_true", help="re-read the shards even if the cache is present")
    args = ap.parse_args()

    # Resolved against --dataset, not against dataset.active: pu200 overrides the eta cut to
    # the barrel-only 0.88, so a pu0 figure drawn while pu200 is active would put the
    # selection lines an axis-width away from where they belong.
    cfg = settings_for(args.dataset)

    results = RESULTS_ROOT / args.dataset
    figures = FIGURES_ROOT / args.dataset
    cache = results / "dataset_features.parquet"

    if cache.exists() and not args.rebuild:
        table = pd.read_parquet(cache)
        print(f"cache {cache} ({len(table):,} particles over {table['event_id'].nunique()} events)", flush=True)
    else:
        print(f"reading {args.events} {args.dataset} events from {args.raw_root}", flush=True)
        table = cml.build_particle_table(
            args.raw_root,
            args.dataset,
            args.events,
            min_hit_energy=float(cfg["hit_selection"]["calohit_min_energy"]),
            collapse_shower_secondaries=True,
        )
        results.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache, index=False)
        print(f"cached {len(table):,} particles to {cache}", flush=True)

    table["particle_class"] = table["particle_class"].astype("category")
    present = table["particle_class"].value_counts()
    classes = [name for name in cml.CLASS_ORDER if present.get(name, 0) > 0]

    style.apply()
    print("  dataset_features   ", figure_features(table, classes, selection_cuts(cfg), figures / "dataset_features"), flush=True)
    print("  dataset_subsystems ", figure_subsystems(table, classes, figures / "dataset_subsystems"), flush=True)

    summary = selection_table(table, cfg)
    summary_path = results / "dataset_features_selection.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n{summary.to_string(index=False, float_format=lambda v: f'{v:.3f}')}\n\nwrote {summary_path}")


if __name__ == "__main__":
    main()
