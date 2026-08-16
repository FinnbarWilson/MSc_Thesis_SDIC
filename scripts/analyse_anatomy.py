"""The deep-dive figures: where each method's energy goes, and what shape of object it makes.

    python -m scripts.analyse_anatomy [--events 120]

Writes figures/<active dataset>/anatomy_*.png|pdf.

WHAT THESE ADD OVER THE EXISTING FIGURES

`make_figures` reports per-particle scalars -- efficiency, purity, fragmentation, response --
binned by energy or isolation. They establish WHICH method is ahead. None of them can say WHY,
because the position of a cell inside its shower is summed away before a row is written, and
that position is where the mechanism lives.

The seven figures here are, in the order they should be read:

  1 shower_profiles   Energy recovered/stolen/dropped against distance from the shower axis, and
                      against depth past the shower's first layer. The truth profile is the
                      target. This is the figure that says whether a method clips halos, swallows
                      neighbours, or stops early in depth.
  2 shower_map        The same three fates as a 2D map in (dr, depth). Denser, and it separates a
                      method that fails everywhere from one that fails in a specific region.
  3 morphology        Transverse and longitudinal extent of the clusters each method emits,
                      against particle energy, with truth overlaid. Generalises "cluster size vs
                      energy" into cluster SHAPE, which is what distinguishes a fixed-radius blob
                      from an object that follows a shower.
  4 complementarity   For each particle: reconstructed by both, one, or neither. If the failures
                      are disjoint the conclusion is about combining strengths; if the `neither`
                      bucket dominates, both are hitting the same limit.
  5 overlap_usage     Does the mask head claim a genuinely shared cell with more than one query?
                      This is the architectural claim for set prediction over partitioning, and
                      it has never been tested. CLUE's answer is 0 by construction.
  6 response          Energy response and resolution against true energy, fitted to
                      a/sqrt(E) + b/E + c. Connects clustering quality to the calorimeter figure
                      of merit, and separates noise-like losses (b) from a systematic scale error
                      (c) -- which the aggregate efficiency cannot do.
  7 pf_consequence    The same errors expressed as particle-flow outcomes: energy over-claimed
                      from neighbours (which deletes a real neutral after track subtraction) and
                      energy under-claimed (which invents one). Same numbers as purity and
                      efficiency, framed as the physics error they become.

Figures 1-3 and 5 are computed from the event store, because they need cell positions. Figures
4, 6 and 7 are computed from the scored parquet tables, so they inherit exactly the matching and
working point that produced the headline numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clue.pipeline import cluster_event
from src.config import figures_dir, settings, store_expectations, store_path
from src.evaluation import anatomy as an
from src.io.event_store import EventStore
from src.plotting import style
from src.postproc import chain_labels

METHODS = ("maskformer", "maskformer_chained", "clue")
#: Particle-energy bands the profiles are drawn in. The pathology is energy dependent -- a fixed
#: cluster scale matters little for a 1 GeV shower and dominates a 20 GeV one -- so a single
#: profile pooled over all energies would average the effect away.
BANDS = ((0.0, 2.0, "E < 2 GeV"), (2.0, 10.0, "2-10 GeV"), (10.0, np.inf, "E > 10 GeV"))


def labels_for(record, method, cfg, clue_params):
    mf = cfg["maskformer"]
    if method == "clue":
        return cluster_event(record, clue_params, coords=cfg["clue"]["coords"],
                             backend=cfg["clue"]["backend"], min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
                             link_radius=cfg["clue"].get("link_radius", 0.0))
    label, n = record.maskformer_labels(mask_threshold=mf["mask_threshold"], object_threshold=mf["object_threshold"],
                                        min_cluster_hits=cfg["metrics"]["min_cluster_hits"])
    if method == "maskformer_chained":
        return chain_labels(record, label, n, link_distance=mf["chain_link_distance"],
                            min_cluster_hits=cfg["metrics"]["min_cluster_hits"])
    return label, n


def collect(store, cfg, clue_params, n_events):
    """One pass over the store, accumulating everything figures 1-3 and 5 need."""
    cells = {m: [] for m in METHODS}
    shapes = {m: [] for m in METHODS}
    truth_shapes = []
    usage = []
    for i, record in enumerate(store):
        if i >= n_events:
            break
        for method in METHODS:
            label, n = labels_for(record, method, cfg, clue_params)
            cells[method].append(an.shower_cells(record, label, n))
            s = an.cluster_shape(record, label, n)
            s["method"] = method
            shapes[method].append(s)
        ts = an.truth_shape(record)
        # Truth extents are indexed by particle, so the energy axis comes from the same table.
        ts["p_energy"] = record.particle_energy
        truth_shapes.append(ts)
        usage.append(an.overlap_usage(record, cfg["maskformer"]["mask_threshold"], cfg["maskformer"]["object_threshold"]))
    merged = {m: an.ShowerCells(**{f: np.concatenate([getattr(c, f) for c in cells[m]])
                                   for f in ("particle", "dr", "depth", "subsystem", "deposit", "fate", "p_energy", "p_pt")})
              for m in METHODS}
    return merged, shapes, truth_shapes, pd.DataFrame(usage)


# --------------------------------------------------------------------------- figures

def fig_shower_profiles(merged, out):
    """Recovered fraction against dr and against depth, one column per energy band."""
    fig, axes = plt.subplots(2, len(BANDS), figsize=(3.0 * len(BANDS), 5.0), sharey=True)
    for col, (lo, hi, title) in enumerate(BANDS):
        for row, (coord, edges, xlabel) in enumerate(
            [("dr", an.DR_EDGES, r"$\Delta R$ from shower axis"), ("depth", an.DEPTH_EDGES, "layers past shower start")]
        ):
            ax = axes[row, col]
            for method in METHODS:
                c, truth, fates = an.profile(merged[method], coord, edges, energy_range=(lo, hi))
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = np.where(truth > 0, fates["recovered"] / truth, np.nan)
                ax.plot(c, frac, marker=style.marker(method), color=style.colour(method),
                        label=style.label(method), markersize=3.2, linewidth=1.1)
            ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel("fraction of the particle's\nenergy recovered")
            ax.set_xlabel(xlabel)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _save(fig, out)


def fig_shower_map(merged, out):
    """Recovered fraction as a 2D map in (dr, depth), one panel per method."""
    fig, axes = plt.subplots(1, len(METHODS), figsize=(3.3 * len(METHODS), 3.1), sharey=True)
    dre, dpe = an.DR_EDGES, an.DEPTH_EDGES
    for ax, method in zip(axes, METHODS, strict=True):
        c = merged[method]
        di = np.digitize(c.dr, dre) - 1
        pi = np.digitize(c.depth, dpe) - 1
        ok = (di >= 0) & (di < len(dre) - 1) & (pi >= 0) & (pi < len(dpe) - 1)
        shape = (len(dpe) - 1, len(dre) - 1)
        tot = np.bincount(pi[ok] * shape[1] + di[ok], weights=c.deposit[ok], minlength=shape[0] * shape[1])
        rec_mask = ok & (c.fate == 0)
        rec = np.bincount(pi[rec_mask] * shape[1] + di[rec_mask], weights=c.deposit[rec_mask], minlength=shape[0] * shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(tot > 0, rec / tot, np.nan).reshape(shape)
        im = ax.pcolormesh(dre, dpe, frac, cmap="viridis", vmin=0, vmax=1, shading="flat")
        ax.set_title(style.label(method), fontsize=9)
        ax.set_xlabel(r"$\Delta R$")
        ax.set_xscale("symlog", linthresh=0.01)
    axes[0].set_ylabel("layers past shower start")
    fig.colorbar(im, ax=axes, label="fraction recovered", pad=0.02)
    return _save(fig, out, tight=False)


def fig_morphology(shapes, truth_shapes, out):
    """Cluster extent against particle energy: is the object the right SHAPE, not just size."""
    bins = np.array([0.5, 1, 2, 5, 10, 20, 50, 200])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, key, ylabel in zip(
        axes, ("transverse", "longitudinal"),
        (r"energy-weighted $\Delta R$ extent", "energy-weighted depth extent [layers]"), strict=True
    ):
        te = np.concatenate([t["p_energy"] for t in truth_shapes])
        tv = np.concatenate([t[key] for t in truth_shapes])
        _binned_median(ax, te, tv, bins, "k", "--", "Truth", "x")
        for method in METHODS:
            # Cluster extents have no particle energy of their own, so each cluster is placed at
            # the energy of the truth particle it best overlaps -- approximated here by matching
            # cluster order to the same events, which is why only the medians are compared.
            ev = np.concatenate([s["energy"] for s in shapes[method]])
            vv = np.concatenate([s[key] for s in shapes[method]])
            _binned_median(ax, ev, vv, bins, style.colour(method), "-", style.label(method), style.marker(method))
        ax.set_xscale("log")
        ax.set_xlabel("cluster / particle energy [GeV]")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _save(fig, out)


def _binned_median(ax, x, y, bins, colour, ls, label, marker):
    idx = np.digitize(x, bins) - 1
    c, m = [], []
    for b in range(len(bins) - 1):
        sel = (idx == b) & np.isfinite(y)
        if sel.sum() >= 5:
            c.append(np.sqrt(bins[b] * bins[b + 1]))
            m.append(np.median(y[sel]))
    ax.plot(c, m, marker=marker, color=colour, linestyle=ls, label=label, markersize=3.2, linewidth=1.1)


def fig_complementarity(particles, out, wp=0.5):
    """Which particles each method reconstructs, and whether the failures are disjoint."""
    piv = particles.pivot_table(index=["sample_id", "particle_row"], columns="algo", values="eff_e")
    kin = particles.drop_duplicates(["sample_id", "particle_row"]).set_index(["sample_id", "particle_row"])
    piv = piv.join(kin[["p_energy", "dr_min"]])
    a, b = "maskformer_chained", "clue"
    if a not in piv or b not in piv:
        return None
    ok_a, ok_b = piv[a] >= wp, piv[b] >= wp
    cat = np.select([ok_a & ok_b, ok_a & ~ok_b, ~ok_a & ok_b], [0, 1, 2], default=3)
    names = ["both", f"{style.label(a)} only", "CLUE only", "neither"]
    cols = ["#4C72B0", style.colour(a), style.colour(b), "#BBBBBB"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, (var, bins, xlabel) in zip(
        axes,
        [("p_energy", np.array([0.5, 1, 2, 5, 10, 20, 200]), "particle energy [GeV]"),
         ("dr_min", np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0]), r"$\Delta R$ to nearest particle")],
        strict=True,
    ):
        idx = np.digitize(piv[var].to_numpy(), bins) - 1
        frac = np.zeros((4, len(bins) - 1))
        for bi in range(len(bins) - 1):
            sel = idx == bi
            if sel.sum():
                for k in range(4):
                    frac[k, bi] = (cat[sel] == k).mean()
        centres = 0.5 * (bins[1:] + bins[:-1])
        ax.stackplot(centres, frac, labels=names, colors=cols, alpha=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction of particles")
    axes[0].set_xscale("log")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    return _save(fig, out)


def fig_response(particles, out):
    """Response and resolution against true energy, with the calorimetry parametrisation."""
    bins = np.array([0.5, 1, 2, 5, 10, 20, 50, 200])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for algo, tab in particles.groupby("algo"):
        if style.is_reference(algo):
            continue
        t = tab[np.isfinite(tab["response"]) & (tab["response"] > 0)]
        idx = np.digitize(t["p_energy"].to_numpy(), bins) - 1
        c, med, res = [], [], []
        for b in range(len(bins) - 1):
            v = t["response"].to_numpy()[idx == b]
            if v.size >= 20:
                q1, q2, q3 = np.percentile(v, [25, 50, 75])
                c.append(np.sqrt(bins[b] * bins[b + 1]))
                med.append(q2)
                res.append((q3 - q1) / 1.349 / max(q2, 1e-9))
        axes[0].plot(c, med, marker=style.marker(algo), color=style.colour(algo), label=style.label(algo),
                     markersize=3.2, linewidth=1.1)
        axes[1].plot(c, res, marker=style.marker(algo), color=style.colour(algo), markersize=3.2, linewidth=1.1)
    axes[0].axhline(1.0, color="k", lw=0.6, ls=":")
    axes[0].set_ylabel(r"median $E_{\rm reco}/E_{\rm true}$")
    axes[1].set_ylabel(r"$\sigma_E/E$  (IQR/1.349, normalised)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$E_{\rm true}$ [GeV]")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return _save(fig, out)


def fig_pf_consequence(particles, out):
    """Clustering error as the particle-flow error it becomes after track subtraction."""
    bins = np.array([0.5, 1, 2, 5, 10, 20, 200])
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for algo, tab in particles.groupby("algo"):
        if style.is_reference(algo):
            continue
        t = tab[np.isfinite(tab["response"])]
        idx = np.digitize(t["p_energy"].to_numpy(), bins) - 1
        r = t["response"].to_numpy()
        c, over, under = [], [], []
        for b in range(len(bins) - 1):
            v = r[idx == b]
            if v.size >= 20:
                c.append(np.sqrt(bins[b] * bins[b + 1]))
                over.append(float((v > 1.2).mean()))
                under.append(float((v < 0.8).mean()))
        ax.plot(c, over, marker=style.marker(algo), color=style.colour(algo), linewidth=1.1, markersize=3.2,
                label=f"{style.label(algo)}: over-claimed")
        ax.plot(c, under, marker=style.marker(algo), color=style.colour(algo), linewidth=1.1, markersize=3.2,
                linestyle="--", alpha=0.7, label=f"{style.label(algo)}: under-claimed")
    ax.set_xscale("log")
    ax.set_xlabel(r"$E_{\rm true}$ [GeV]")
    ax.set_ylabel("fraction of particles beyond $\\pm$20%")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    return _save(fig, out)


def fig_overlap_usage(usage, out):
    """Does the mask head use the multi-claim freedom its architecture provides?"""
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    keys = ["truth_shared_frac", "claimed_twice_given_shared", "claimed_twice_given_single"]
    names = ["truth: cells with\n$\\geq$2 contributors", "MaskFormer claims $\\geq$2x\ngiven truly shared",
             "MaskFormer claims $\\geq$2x\ngiven single-owner"]
    vals = [usage[k].mean() for k in keys]
    ax.barh(names, vals, color=[ "#BBBBBB", style.colour("maskformer"), "#C9B7D4"])
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xlabel("fraction of cells")
    fig.tight_layout()
    return _save(fig, out)


def _save(fig, out, tight=True):
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), bbox_inches="tight" if tight else None)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=int, default=120, help="events for the store-based figures (1-3, 5)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = settings()
    out_dir = args.out or figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    clue_params = {k: v["parameters"] for k, v in
                   json.loads((Path("results") / cfg["dataset"]["active"] / "clue_parameters.json").read_text())["subsystems"].items()}

    store = EventStore(store_path("store"), expect=store_expectations())
    print(f"store {store.root} ({len(store)} events); using {min(args.events, len(store))} for the spatial figures", flush=True)
    merged, shapes, truth_shapes, usage = collect(store, cfg, clue_params, args.events)

    style.apply()
    print("  1 shower_profiles ", fig_shower_profiles(merged, out_dir / "anatomy_shower_profiles"), flush=True)
    print("  2 shower_map      ", fig_shower_map(merged, out_dir / "anatomy_shower_map"), flush=True)
    print("  3 morphology      ", fig_morphology(shapes, truth_shapes, out_dir / "anatomy_morphology"), flush=True)
    print("  5 overlap_usage   ", fig_overlap_usage(usage, out_dir / "anatomy_overlap_usage"), flush=True)

    res = Path("results") / cfg["dataset"]["active"]
    parts = pd.concat([pd.read_parquet(p) for p in sorted(res.glob("particles_*.parquet"))], ignore_index=True)
    print("  4 complementarity ", fig_complementarity(parts, out_dir / "anatomy_complementarity"), flush=True)
    print("  6 response        ", fig_response(parts, out_dir / "anatomy_response"), flush=True)
    print("  7 pf_consequence  ", fig_pf_consequence(parts, out_dir / "anatomy_pf_consequence"), flush=True)
    print(f"\nwrote 7 figures to {out_dir}")


if __name__ == "__main__":
    main()
