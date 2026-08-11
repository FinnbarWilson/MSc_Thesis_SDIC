"""The five thesis figures, pileup 0 and pileup 200 side by side.

    python -m scripts.make_thesis_figures                 # every dataset that has tables
    python -m scripts.make_thesis_figures --datasets pu200
    python -m scripts.make_thesis_figures --rebuild-anatomy

Writes figures/thesis/*.pdf|png -- one directory, because these are cross-dataset and do not
belong under either dataset's own folder.

WHAT IS HERE AND WHAT IS NOT

Five figures carry the argument:

  1 eff_purity      efficiency and purity against particle energy, with both ceilings
  2 cluster_size    cells recovered against particle energy, against the truth's own growth
  3 shower_profile  where in the shower the energy is lost: transverse, then longitudinal
  4 response        energy response and resolution -- bias against variance
  5 energy_budget   the three fates of a particle's energy, as one bar per method

Everything else that exists under figures/<dataset>/ is either methodology (the working-point
scan, the weighting comparison), a duplicate of one of these at lower information density (the
shower map, the split/merge and efficiency decompositions), or dead (the incidence-head
comparison, which needs a checkpoint that has one). They belong in an appendix or nowhere.

The six training interventions and eleven post-processing methods are a TABLE, not a figure.
Seventeen arms with controls do not plot legibly and read perfectly well as rows.

DATA SOURCES, and the one that has to be built

Figures 1, 4 and 5 read results/<ds>/particles_*.parquet and clusters_*.parquet, so they inherit
exactly the matching and working point behind the headline numbers.

Figures 2 and 3 need cell positions, so they are built from the event store and cached to
results/<ds>/anatomy_particles.parquet. That cache is per (dataset, method) and is rebuilt only
with --rebuild-anatomy, because the pass over the store costs minutes and the figures are
iterated on far more often than the underlying numbers change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.clue.pipeline import cluster_event
from src.config import CONFIG_PATH, settings, store_expectations, store_path
from src.evaluation import anatomy as an
from src.evaluation import jets as jt
from src.evaluation.matching import hungarian_match, overlap_matrix
from src.io.event_store import EventStore
from src.plotting import thesis as th
from src.postproc import chain_labels

#: The comparison the thesis actually makes: one learned method against one classical one.
#:
#: `maskformer_chained` is DELIBERATELY ABSENT. It is a hand-tuned geometric post-process with its
#: own free parameter, so putting it in the headline turns "learned model vs tuned classical
#: algorithm" into "learned model plus classical post-process vs classical algorithm" -- a
#: different question. Its +0.09 efficiency also comes with median response 1.8 at 0.7 GeV, i.e.
#: clusters holding nearly twice their particle's energy, which an efficiency number hides.
#:
#: It belongs in the interventions table, where the gain and its cost sit side by side. Pass
#: --with-chaining to draw it anyway; the value of the post-processing is as EVIDENCE that the
#: masks are fragmented geometrically, not as a competing method.
METHODS_MAIN = ("maskformer", "clue")
METHODS = METHODS_MAIN
OUT = Path("figures/thesis")


# ------------------------------------------------------------------ anatomy cache

def _labels(record, method, cfg, clue_params):
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


def build_anatomy(dataset: str, cfg, n_events: int) -> pd.DataFrame:
    """One row per (particle, method): matched-cluster size, extents, and the fate profile.

    Everything a MATCHED PAIR, so the energy axis in every figure is the true particle's and
    never the cluster's -- the working figures mixed the two on one axis and that is the kind of
    ambiguity a reader should not have to resolve.
    """
    clue_params = {k: v["parameters"] for k, v in
                   json.loads((Path("results") / dataset / "clue_parameters.json").read_text())["subsystems"].items()}
    store = EventStore(store_path("store"), expect=store_expectations())
    rows = []
    for i, record in enumerate(store):
        if i >= n_events:
            break
        truth_ext = an.truth_shape(record)
        truth_cells = np.bincount(record.truth_label[record.truth_label >= 0], minlength=record.n_particles)
        deposit = record.truth_deposit
        for method in METHODS:
            label, n = _labels(record, method, cfg, clue_params)
            shape = an.cluster_shape(record, label, n)
            ov = overlap_matrix(record.truth_label, label, deposit, record.n_particles, n)
            match = hungarian_match(ov, min_overlap=0.0)
            matched = np.full(record.n_particles, -1, dtype=np.int64)
            matched[match.truth_index] = match.pred_index

            cells = an.shower_cells(record, label, n)
            for p in range(record.n_particles):
                if truth_cells[p] == 0:
                    continue
                c = matched[p]
                sel = cells.particle == p
                rec = sel & (cells.fate == 0)
                rows.append({
                    "dataset": dataset, "algo": method, "sample_id": record.sample_id, "particle_row": p,
                    "p_energy": float(record.particle_energy[p]),
                    "n_true": int(truth_cells[p]),
                    "n_recovered": int(rec.sum()),
                    "n_matched_cluster": int(shape["n_cells"][c]) if c >= 0 else 0,
                    "truth_transverse": float(truth_ext["transverse"][p]),
                    "truth_longitudinal": float(truth_ext["longitudinal"][p]),
                    "clus_transverse": float(shape["transverse"][c]) if c >= 0 else np.nan,
                    "clus_longitudinal": float(shape["longitudinal"][c]) if c >= 0 else np.nan,
                })
    return pd.DataFrame(rows)


def build_jets(dataset: str, cfg, n_events: int) -> pd.DataFrame:
    """Anti-k_t jets per event, for every method and for the truth partition."""
    clue_params = {k: v["parameters"] for k, v in
                   json.loads((Path("results") / dataset / "clue_parameters.json").read_text())["subsystems"].items()}
    store = EventStore(store_path("store"), expect=store_expectations())
    rows = []
    for i, record in enumerate(store):
        if i >= n_events:
            break
        rows += jt.event_rows(record, {m: _labels(record, m, cfg, clue_params) for m in METHODS}, dataset)
    return pd.DataFrame(rows)


def load_profiles(dataset: str, cfg, n_events: int):
    """Pooled shower-frame profiles, which are aggregates and so are not worth caching per row."""
    clue_params = {k: v["parameters"] for k, v in
                   json.loads((Path("results") / dataset / "clue_parameters.json").read_text())["subsystems"].items()}
    store = EventStore(store_path("store"), expect=store_expectations())
    per = {m: [] for m in METHODS}
    for i, record in enumerate(store):
        if i >= n_events:
            break
        for method in METHODS:
            label, n = _labels(record, method, cfg, clue_params)
            per[method].append(an.shower_cells(record, label, n))
    return {m: an.ShowerCells(**{f: np.concatenate([getattr(c, f) for c in per[m]])
                                 for f in ("particle", "dr", "depth", "subsystem", "deposit", "fate", "p_energy")})
            for m in METHODS}


# ------------------------------------------------------------------ figures

def fig_eff_purity(tables, datasets, out):
    """Efficiency and purity against particle energy.

    THE TWO "CEILINGS" ARE DELIBERATELY NOT DRAWN, having been tried and removed. Neither is an
    upper bound on both metrics, so both got crossed and the figure raised a question it could not
    answer:

      oracle_geometric assigns each cell to the NEAREST SHOWER AXIS given perfect seeds. That is
      one specific rule and a suboptimal one, so a method that is not nearest-axis-based can beat
      it -- CLUE does, above ~20 GeV.

      oracle_resolution merges truth particles sharing more than half their energy and then
      clusters perfectly. Its EFFICIENCY (0.989) is a real bound; its PURITY is scored against the
      unmerged truth, so its merged clusters are impure by construction and both methods beat it
      at low energy.

    Calling either a ceiling claims more than it supports. The useful content is one number --
    oracle_resolution reaches 0.989 efficiency, so the cells carry the information and the failure
    is algorithmic rather than intrinsic -- and that belongs in the text, not as a flat line at
    0.99 compressing the y-axis.
    """
    fig, axes = th.grid(2, len(datasets), datasets)
    for j, ds in enumerate(datasets):
        parts, clus = tables[ds]
        for algo in METHODS:
            p = parts[parts.algo == algo]
            if p.empty:
                continue
            x, y, lo, hi = th.binned_proportion(p.p_energy.to_numpy(), (p.eff_e >= 0.5).to_numpy(), th.E_BINS)
            th.band(axes[0][j], algo, x, lo, hi)
            th.draw(axes[0][j], algo, x, y)
            # Purity is a per-CLUSTER quantity, but it is binned by the energy of the particle
            # that cluster was matched to, not by the cluster's own energy. Both rows then share
            # one x-axis that means one thing. Binning the rows by different energies under a
            # single "particle energy" label is exactly the ambiguity these figures exist to
            # remove -- and it silently made the two rows non-comparable.
            c = clus[(clus.algo == algo) & (clus.particle_row >= 0)].merge(
                p[["sample_id", "particle_row", "p_energy"]], on=["sample_id", "particle_row"], how="inner")
            if c.empty:
                continue
            x, y, lo, hi = th.binned_proportion(c.p_energy.to_numpy(), (c.pur_e >= 0.5).to_numpy(), th.E_BINS)
            th.band(axes[1][j], algo, x, lo, hi)
            th.draw(axes[1][j], algo, x, y)
        axes[1][j].set_xlabel("particle energy [GeV]")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("efficiency")
    axes[1][0].set_ylabel("purity")
    return th.finish(fig, out)


def fig_cluster_size(anat, datasets, out):
    """Does the cluster keep up with the shower? Fraction of a shower's cells correctly assigned.

    Plotted as a RATIO OF SUMS -- all correctly-assigned cells in an energy bin over all truth
    cells in it -- rather than as a mean of per-particle counts. The per-particle distribution is
    bimodal for CLUE, which usually fragments a large shower and occasionally captures most of
    one: at 50-200 GeV its matched cluster holds a mean of 11.1 cells and a median of 1. A mean
    over those two behaviours describes neither, and it reversed the sign of the trend. Summing
    first asks a question the distribution shape cannot distort.

    A flat line would mean the clusters grow in step with the showers. Both methods fall, so
    neither does; the interesting quantity is how steeply.
    """
    fig, axes = th.grid(1, len(datasets), datasets, sharey=True)
    for j, ds in enumerate(datasets):
        a = anat[anat.dataset == ds]
        for algo in METHODS:
            s = a[a.algo == algo]
            if s.empty:
                continue
            x, y, lo, hi = th.binned_ratio(s.p_energy.to_numpy(), s.n_recovered.to_numpy(),
                                           s.n_true.to_numpy(), s.sample_id.to_numpy(), th.E_BINS)
            th.band(axes[0][j], algo, x, lo, hi)
            th.draw(axes[0][j], algo, x, y)
        axes[0][j].set_xscale("log")
        axes[0][j].set_ylim(0, 0.65)
        axes[0][j].set_xlabel("particle energy [GeV]")
    axes[0][0].set_ylabel("fraction of the shower's cells\ncorrectly assigned")
    return th.finish(fig, out, ncol=len(METHODS))


def fig_shower_profile(profiles, datasets, out):
    """Where the energy is lost: transverse first, then longitudinal."""
    # sharex=False: the rows are different coordinates, not the same one at two scales.
    fig, axes = th.grid(2, len(datasets), datasets, sharex=False)
    for j, ds in enumerate(datasets):
        for row, (coord, edges, xlabel) in enumerate(
            [("dr", an.DR_EDGES, "$\\Delta R$ from shower axis"),
             ("depth", an.DEPTH_EDGES, "layers past shower start")]
        ):
            for algo in METHODS:
                c, truth, fates = an.profile(profiles[ds][algo], coord, edges)
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = np.where(truth > 0, fates["recovered"] / truth, np.nan)
                th.draw(axes[row][j], algo, c, frac)
            axes[row][j].set_ylim(0, 1.0)
            axes[row][j].set_xlabel(xlabel)
    axes[0][0].set_ylabel("fraction of energy\nrecovered")
    axes[1][0].set_ylabel("fraction of energy\nrecovered")
    return th.finish(fig, out)


def fig_response(tables, datasets, out):
    """Bias against variance: the two methods fail in different currencies."""
    fig, axes = th.grid(2, len(datasets), datasets, sharey="row")
    for j, ds in enumerate(datasets):
        parts, _ = tables[ds]
        for algo in METHODS:
            p = parts[(parts.algo == algo) & np.isfinite(parts.response) & (parts.response > 0)]
            if p.empty:
                continue
            e, r, ev = p.p_energy.to_numpy(), p.response.to_numpy(), p.sample_id.to_numpy()
            x, med, lo, hi = th.binned_bootstrap(e, r, ev, th.E_BINS, "median")
            th.band(axes[0][j], algo, x, lo, hi)
            th.draw(axes[0][j], algo, x, med)
            x, res, lo, hi = th.binned_bootstrap(e, r, ev, th.E_BINS, "resolution")
            th.band(axes[1][j], algo, x, lo, hi)
            th.draw(axes[1][j], algo, x, res)
        axes[0][j].axhline(1.0, color="k", lw=0.5, ls=":")
        axes[0][j].set_ylim(0.4, 2.0)
        axes[1][j].set_yscale("log")
        axes[1][j].set_yticks([0.5, 1, 2, 5])
        axes[1][j].set_yticklabels(["0.5", "1", "2", "5"])
        axes[1][j].minorticks_off()
        axes[1][j].set_xlabel("true particle energy [GeV]")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log")
    axes[0][0].set_ylabel("median $E_{\\rm reco}/E_{\\rm true}$")
    axes[1][0].set_ylabel("$\\sigma_E/E$")
    return th.finish(fig, out)


def fig_energy_budget(tables, datasets, out):
    """The three fates of a particle's energy: one full bar per method, darkest to lightest.

    HORIZONTAL AND STACKED, not grouped. A stacked bar is one continuous 0-to-1 line per method,
    so "how is this method's energy divided" is read in a single sweep and the methods sit
    directly above one another for comparison. The grouped version made the within-fate
    comparison easy and the within-method one hard, which is the wrong way round: the point of
    this figure is that the two methods lose the SAME total in DIFFERENT ways.

    Colour carries the method and lightness carries the fate, so the figure needs no second
    colour language -- the blue bar is MaskFormer everywhere it appears, and the palest segment is
    always the energy nobody claimed.
    """
    fig, ax = plt.subplots(figsize=th.figsize_for(1, 1))
    fates = ["recovered", "taken by another cluster", "never claimed"]
    alphas = (1.0, 0.55, 0.25)
    labels, rows = [], []
    for ds in datasets:
        parts, _ = tables[ds]
        for algo in METHODS:
            p = parts[parts.algo == algo]
            if p.empty:
                continue
            tot = p.e_dep_calib.sum()
            # `eff_e * e_dep_calib`, NOT `e_reco_calib`. The latter is the matched cluster's TOTAL
            # energy, contamination included -- it sums to 0.97 of the deposit here, and using it
            # made the three fates add to 1.17. Only the energy-weighted efficiency measures the
            # part of THIS particle's deposit that the matched cluster holds, and with it the
            # decomposition closes exactly.
            vals = [
                (p.eff_e * p.e_dep_calib).sum() / tot,
                p.e_lost_other.sum() / tot,
                p.e_lost_noise.sum() / tot,
            ]
            assert abs(sum(vals) - 1.0) < 1e-6, f"fates for {algo} sum to {sum(vals):.4f}, not 1"
            prefix = f"{th.DATASET_LABELS[ds]}\n" if len(datasets) > 1 else ""
            labels.append(prefix + th.LABELS[algo])
            rows.append((algo, vals))

    y = np.arange(len(rows))
    for i, (algo, vals) in enumerate(rows):
        left = 0.0
        for v, a in zip(vals, alphas, strict=True):
            ax.barh(y[i], v, left=left, color=th.colour(algo), alpha=a, height=0.62)
            if v > 0.06:  # below this the label collides with the segment edges
                ax.text(left + v / 2, y[i], f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if a > 0.8 else "black")
            left += v
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of the particle's deposited energy")
    ax.invert_yaxis()

    # Lightness is the variable here, so the key is drawn in a neutral grey at the same three
    # alphas. Using a method's colour would imply the key belonged to that method.
    from matplotlib.patches import Patch

    handles = [Patch(facecolor="#333333", alpha=a, label=f) for a, f in zip(alphas, fates, strict=True)]
    fig.legend(handles, fates, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    return out


def fig_jets(jets, datasets, out):
    """Do the per-cluster errors survive into jets? Find it, measure it, measure it precisely.

    Three rows, because a jet can fail in three separable ways and an analysis cares about all of
    them: the jet can be missing entirely, its energy scale can be wrong, or its energy can be
    imprecise. The rows mirror the cluster-level response figure, so the question "does the
    cluster-level bias-variance split carry through to an observable" is answered by reading the
    two figures against each other.

    x is the REFERENCE jet pt throughout -- the jet a perfect clusterer would have made from these
    cells -- so every row is binned by the same, method-independent quantity.
    """
    fig, axes = th.grid(3, len(datasets), datasets, sharey="row")
    xmax = 0.0
    for j, ds in enumerate(datasets):
        d = jets[jets.dataset == ds]
        for algo in METHODS:
            a = d[d.algo == algo]
            if a.empty:
                continue
            x, y, lo, hi = th.binned_proportion(a.ref_pt.to_numpy(), a.matched.to_numpy(), th.JET_PT_BINS, min_count=10)
            th.band(axes[0][j], algo, x, lo, hi)
            th.draw(axes[0][j], algo, x, y)
            xmax = max(xmax, float(x.max()) if len(x) else 0.0)

            m = a[a.matched]
            if m.empty:
                continue
            ratio = (m.reco_pt / m.ref_pt).to_numpy()
            ref, ev = m.ref_pt.to_numpy(), m.sample_id.to_numpy()
            x, med, lo, hi = th.binned_bootstrap(ref, ratio, ev, th.JET_PT_BINS, "median", min_count=10)
            th.band(axes[1][j], algo, x, lo, hi)
            th.draw(axes[1][j], algo, x, med)
            x, res, lo, hi = th.binned_bootstrap(ref, ratio, ev, th.JET_PT_BINS, "resolution", min_count=10)
            th.band(axes[2][j], algo, x, lo, hi)
            th.draw(axes[2][j], algo, x, res)
        axes[0][j].set_ylim(0, 1.02)
        # No hardcoded ylim on the response row: an earlier version capped it at 1.3 and silently
        # clipped CLUE's entire line off the top. An axis limit that can hide a series is worse
        # than an untidy one.
        axes[1][j].axhline(1.0, color="k", lw=0.5, ls=":")
        ticks = [t for t in (25, 35, 50, 75, 100, 150, 200, 300, 400) if t <= xmax * 1.12]
        for row in range(3):
            axes[row][j].set_xscale("log")
            axes[row][j].set_xlim(th.JET_MIN_PT * 0.92, xmax * 1.12)
            axes[row][j].set_xticks(ticks)
            axes[row][j].set_xticklabels([str(t) for t in ticks])
            axes[row][j].minorticks_off()
        axes[2][j].set_xlabel(r"reference jet $p_{\rm T}$ [GeV]")
    axes[0][0].set_ylabel("jet efficiency")
    axes[1][0].set_ylabel(r"median $p_{\rm T}^{\rm reco}/p_{\rm T}^{\rm ref}$")
    axes[2][0].set_ylabel(r"jet $\sigma_{p_{\rm T}}/p_{\rm T}$")
    return th.finish(fig, out, ncol=len(METHODS))


# ------------------------------------------------------------------ driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(th.DATASETS))
    ap.add_argument("--events", type=int, default=120, help="events for the store-derived figures")
    ap.add_argument("--rebuild-anatomy", action="store_true")
    ap.add_argument("--rebuild-jets", action="store_true")
    ap.add_argument("--with-chaining", action="store_true",
                    help="also draw maskformer_chained; off by default, see METHODS_MAIN")
    args = ap.parse_args()

    import matplotlib.pyplot as _plt

    globals()["plt"] = _plt
    if args.with_chaining:
        globals()["METHODS"] = ("maskformer", "maskformer_chained", "clue")
    th.apply()
    OUT.mkdir(parents=True, exist_ok=True)

    active = settings()["dataset"]["active"]
    available, tables, anat, profiles, jets_all = [], {}, [], {}, []
    for ds in args.datasets:
        res = Path("results") / ds
        parts = sorted(res.glob("particles_*.parquet"))
        if not parts:
            print(f"  ! {ds}: no particles_*.parquet in {res} -- skipping this column "
                  f"(dump its store and run scripts.score)", flush=True)
            continue
        available.append(ds)
        tables[ds] = (
            pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True),
            pd.concat([pd.read_parquet(p) for p in sorted(res.glob("clusters_*.parquet"))], ignore_index=True),
        )
        jcache = res / "jets.parquet"
        if ds == active and (args.rebuild_jets or not jcache.exists()):
            print(f"  building jet cache for {ds} ({args.events} events) ...", flush=True)
            build_jets(ds, settings(), args.events).to_parquet(jcache)
        if jcache.exists():
            jets_all.append(pd.read_parquet(jcache))
        cache = res / "anatomy_particles.parquet"
        if ds == active:
            if args.rebuild_anatomy or not cache.exists():
                print(f"  building anatomy cache for {ds} ({args.events} events) ...", flush=True)
                build_anatomy(ds, settings(), args.events).to_parquet(cache)
            profiles[ds] = load_profiles(ds, settings(), args.events)
        if cache.exists():
            anat.append(pd.read_parquet(cache))
        else:
            print(f"  ! {ds}: no anatomy cache and it is not the active dataset "
                  f"(dataset.active is {active!r} in {CONFIG_PATH.name}); figures 2-3 will omit it", flush=True)

    if not available:
        raise SystemExit("no dataset had scored tables; nothing to draw")
    anat = pd.concat(anat, ignore_index=True) if anat else pd.DataFrame()
    prof_ds = [d for d in available if d in profiles]

    print("1 eff_purity     ", fig_eff_purity(tables, available, OUT / "fig_eff_purity"), flush=True)
    if not anat.empty:
        ds2 = [d for d in available if d in set(anat.dataset)]
        print("2 cluster_size   ", fig_cluster_size(anat, ds2, OUT / "fig_cluster_size"), flush=True)
    if prof_ds:
        print("3 shower_profile ", fig_shower_profile(profiles, prof_ds, OUT / "fig_shower_profile"), flush=True)
    print("4 response       ", fig_response(tables, available, OUT / "fig_response"), flush=True)
    print("5 energy_budget  ", fig_energy_budget(tables, available, OUT / "fig_energy_budget"), flush=True)
    if jets_all:
        jets = pd.concat(jets_all, ignore_index=True)
        ds6 = [d for d in available if d in set(jets.dataset)]
        print("6 jets           ", fig_jets(jets, ds6, OUT / "fig_jets"), flush=True)
    print(f"\ndatasets drawn: {available}")


if __name__ == "__main__":
    main()
