"""Draw the eight thesis figures, pileup 0 and pileup 200 side by side.

    python -m scripts.make_thesis_figures                       # every dataset with tables
    python -m scripts.make_thesis_figures --datasets pu200
    python -m scripts.make_thesis_figures --from-summary        # no event store needed

Writes ``figures/thesis/*.pdf|png``. Every dataset is first reduced to
``results/<ds>/figure_summary.csv``, a few hundred binned points and about 20 KB. The figures
draw from that and nothing else, so a machine holding only the committed summaries can redraw
every column. :func:`build_summary` is the only place the aggregation happens.

Jets and the shower anatomy cannot be recovered from the per-row tables and are rebuilt from
the event store for the active dataset only; both are folded into the summary. A machine that
has the per-row tables rebuilds the summary rather than trusting the committed one.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import settings, store_expectations, store_path
from src.evaluation import anatomy as an
from src.evaluation import jets as jt
from src.evaluation.differential import clopper_pearson
from src.evaluation.matching import hungarian_match, overlap_matrix
from src.io.event_store import EventStore
from src.plotting import thesis as th

#: One learned method against one classical one.
METHODS = ("maskformer", "clue")
OUT = Path("figures/thesis")


# ------------------------------------------------------------------ anatomy cache

def _labels(record, method, cfg, clue_params):
    mf = cfg["maskformer"]
    if method == "clue":
        # Imported here, not at module scope: src.clue.pipeline imports CLUEstering, which is
        # needed only to re-cluster events and must not become a requirement of --from-summary.
        from src.clue.pipeline import cluster_event

        return cluster_event(record, clue_params, coords=cfg["clue"]["coords"],
                             backend=cfg["clue"]["backend"], min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
                             link_radius=cfg["clue"].get("link_radius", 0.0))
    return record.maskformer_labels(mask_threshold=mf["mask_threshold"],
                                   object_threshold=mf["object_threshold"],
                                   min_cluster_hits=cfg["metrics"]["min_cluster_hits"])


def build_anatomy(dataset: str, cfg, n_events: int) -> pd.DataFrame:
    """One row per (particle, method): matched-cluster size, extents and the fate profile.

    Every row is a matched pair, so the energy axis in every figure is the true particle's and
    never the cluster's.
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
                    "p_pt": float(record.particle_pt[p]),
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
    """Pooled shower-frame profiles. Aggregates, so not worth caching per row."""
    clue_params = {k: v["parameters"] for k, v in
                   json.loads((Path("results") / dataset / "clue_parameters.json").read_text())["subsystems"].items()}
    store = EventStore(store_path("store"), expect=store_expectations())
    per = {m: [] for m in METHODS}
    # The event index is the resampling unit for the profile's band, and `shower_cells` cannot
    # know its own index.
    events = {m: [] for m in METHODS}
    for i, record in enumerate(store):
        if i >= n_events:
            break
        for method in METHODS:
            label, n = _labels(record, method, cfg, clue_params)
            cells = an.shower_cells(record, label, n)
            per[method].append(cells)
            events[method].append(np.full(cells.dr.shape, i, dtype=np.int64))
    return {m: an.ShowerCells(
                event=np.concatenate(events[m]) if events[m] else None,
                **{f: np.concatenate([getattr(c, f) for c in per[m]])
                   for f in ("particle", "dr", "depth", "subsystem", "deposit", "fate", "p_energy", "p_pt")})
            for m in METHODS}


def profile_with_interval(cells, coord: str, edges: np.ndarray, n_boot: int = 200):
    """(centres, {series: (fraction, lo, hi)}) for a shower profile, resampled over events.

    The profile is a ratio of sums in each bin, so the interval is built by resampling whole
    events and recomputing the ratio, as `th.binned_ratio` does elsewhere.

    Two series, the second cumulative: ``recovered_taken`` is recovered plus taken by another
    cluster, so the two curves and the line at 1 partition each bin into the same three fates as
    the energy-budget figure. Both come out of one pass over the cells.
    """
    centres, truth, fates = an.profile(cells, coord, edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = {
            "recovered": np.where(truth > 0, fates["recovered"] / truth, np.nan),
            "recovered_taken": np.where(
                truth > 0, (fates["recovered"] + fates["stolen"]) / truth, np.nan),
        }

    nan = np.full(centres.shape, np.nan)
    if getattr(cells, "event", None) is None:
        return centres, {k: (v, nan, nan) for k, v in point.items()}

    values = getattr(cells, coord)
    idx = np.digitize(values, edges) - 1
    inside = (idx >= 0) & (idx < len(edges) - 1)
    idx, w, fate, ev = idx[inside], cells.deposit[inside], cells.fate[inside], cells.event[inside]
    if idx.size == 0:
        return centres, {k: (v, nan, nan) for k, v in point.items()}

    # Collapse to a (event x bin) set of matrices once; a resample is then a row gather and a
    # column sum rather than a rescan of several million cell rows.
    uniq, row = np.unique(ev, return_inverse=True)
    nb = len(edges) - 1
    dep = np.zeros((uniq.size, nb))
    num = {k: np.zeros((uniq.size, nb)) for k in point}
    np.add.at(dep, (row, idx), w)
    np.add.at(num["recovered"], (row, idx), np.where(fate == 0, w, 0.0))
    np.add.at(num["recovered_taken"], (row, idx), np.where(fate <= 1, w, 0.0))

    out = {}
    for name, value in point.items():
        rng = np.random.default_rng(0)
        boot = np.empty((n_boot, nb))
        for b in range(n_boot):
            pick = rng.integers(0, uniq.size, uniq.size)
            d, r = dep[pick].sum(0), num[name][pick].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[b] = np.where(d > 0, r / d, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column when a bin is empty
            lo = np.nanpercentile(boot, 15.865, axis=0)
            hi = np.nanpercentile(boot, 84.135, axis=0)
        out[name] = (value, lo, hi)

    # The fates are exhaustive, so the cumulative curve cannot exceed 1 nor sit below the
    # recovered one. Either failure would look entirely plausible on the page.
    rec, tak = point["recovered"], point["recovered_taken"]
    ok = np.isfinite(rec) & np.isfinite(tak)
    assert np.all(tak[ok] >= rec[ok] - 1e-9), "recovered exceeds recovered+taken in some bin"
    assert np.all(tak[ok] <= 1.0 + 1e-9), "recovered+taken exceeds the deposited energy in some bin"
    return centres, out


# ------------------------------------------------------------------ the figure summary
#
# Every figure reduces to a handful of binned series, so the aggregation happens once, here, and
# both code paths draw the same numbers, whether rebuilding from local tables or reading the
# committed summary. The format is long/tidy, so adding a method or a panel needs no schema change.

SUMMARY_NAME = "figure_summary.csv"
SUMMARY_COLS = ("dataset", "figure", "panel", "algo", "x", "y", "lo", "hi")

#: Photons and electrons are pooled, because their showers are identical to a calorimeter and
#: the two populations barely share a crowding regime. Charged and neutral hadrons are split,
#: because they differ in how hard they are to cluster and in what a particle-flow chain needs
#: from the calorimeter, and their dR distributions are alike. Muons are not drawn: they deposit
#: well under a percent of the target energy, and stay in the per-class table instead.
PARTICLE_GROUPS: Mapping[str, tuple[int, ...]] = {
    "electromagnetic": (0, 1),
    "charged_hadron": (5,),
    "neutral_hadron": (6,),
}
GROUP_LABELS: tuple[str, ...] = (
    "electromagnetic\n($\\gamma$, e)", "charged\nhadron", "neutral\nhadron",
)

#: Isolation bins. The first edge is 0.005 rather than 0 because bin centres are geometric means.
#: Targets alone in their event have infinite separation and are dropped rather than piled into
#: the last bin. Coarse, because the figure is also cut three ways in pT.
DR_MIN_BINS = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 5.0])

#: The crowding figure is cut in pT, because pooled over all targets the isolation axis selects
#: on energy beyond its peak: isolated targets are predominantly soft. Within a slice the curves
#: are monotonic and the crowding effect is unambiguous. No reference clusterings are drawn.
ISOLATION_ALGOS: tuple[str, ...] = ("maskformer", "clue")

#: (low, high, panel key, label). The panel key goes in figure_summary.csv; the label titles the row.
ISOLATION_PT_SLICES: tuple[tuple[float, float, str, str], ...] = (
    (0.5, 2.0, "pt_soft", r"$0.5 < p_{\rm T} < 2$ GeV"),
    (2.0, 10.0, "pt_mid", r"$2 < p_{\rm T} < 10$ GeV"),
    (10.0, np.inf, "pt_hard", r"$p_{\rm T} > 10$ GeV"),
)


def _rows(dataset, figure, panel, algo, x, y, lo=None, hi=None):
    """One dict per point of a binned series; lo/hi are NaN for series without a band."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = np.full(x.shape, np.nan) if lo is None else np.asarray(lo, dtype=float)
    hi = np.full(x.shape, np.nan) if hi is None else np.asarray(hi, dtype=float)
    return [
        {"dataset": dataset, "figure": figure, "panel": panel, "algo": algo,
         "x": float(xi), "y": float(yi), "lo": float(li), "hi": float(hi_i)}
        for xi, yi, li, hi_i in zip(x, y, lo, hi, strict=True)
    ]


def build_summary(ds, tables, anat, profiles, jets) -> pd.DataFrame:
    """Reduce one dataset's per-row tables to the binned series the eight figures draw.

    Each source is optional: a dataset contributes whatever of `tables`, `anat`, `profiles` and
    `jets` it has.
    """
    rows = []
    parts, clus = tables.get(ds, (None, None))

    if parts is not None:
        # 1 eff_purity. Purity is per-cluster but binned by the pT of the particle its cluster
        # was matched to, so both rows share one x-axis that means one thing.
        for algo in METHODS:
            p = parts[parts.algo == algo]
            if p.empty:
                continue
            x, y, lo, hi = th.binned_proportion(p.p_pt.to_numpy(), (p.eff_e >= 0.5).to_numpy(), th.E_BINS)
            rows += _rows(ds, "eff_purity", "efficiency", algo, x, y, lo, hi)
            c = clus[(clus.algo == algo) & (clus.particle_row >= 0)].merge(
                p[["sample_id", "particle_row", "p_pt"]], on=["sample_id", "particle_row"], how="inner")
            if c.empty:
                continue
            x, y, lo, hi = th.binned_proportion(c.p_pt.to_numpy(), (c.pur_e >= 0.5).to_numpy(), th.E_BINS)
            rows += _rows(ds, "eff_purity", "purity", algo, x, y, lo, hi)

        # 4 response.
        for algo in METHODS:
            p = parts[(parts.algo == algo) & np.isfinite(parts.response) & (parts.response > 0)]
            if p.empty:
                continue
            e, r, ev = p.p_pt.to_numpy(), p.response.to_numpy(), p.sample_id.to_numpy()
            for panel, stat in (("median", "median"), ("resolution", "resolution")):
                x, v, lo, hi = th.binned_bootstrap(e, r, ev, th.E_BINS, stat)
                rows += _rows(ds, "response", panel, algo, x, v, lo, hi)

        # 6 particle_class. x is the class code, so the binned helpers do not apply: they take a
        # geometric bin centre, which is undefined at code 0. The statistics are the ones they use.
        for algo in METHODS:
            p = parts[parts.algo == algo]
            if p.empty or "particle_class" not in p.columns:
                continue
            for code, (_, members) in enumerate(PARTICLE_GROUPS.items()):
                sel = p[p.particle_class.isin(members)]
                if len(sel) < 20:
                    continue
                k = int((sel.eff_e >= 0.5).sum())
                n = int(len(sel))
                lo, hi = clopper_pearson(np.array([k]), np.array([n]))
                rows += _rows(ds, "particle_class", "efficiency", algo, [code], [k / n], lo, hi)

                # `frag_frac`, the share outside the largest piece, rather than the split-rate
                # flag, which cannot see a shower shattered into more than ten pieces.
                f = sel.frag_frac.to_numpy()
                ev = sel.sample_id.to_numpy()
                uniq = np.unique(ev)
                rng = np.random.default_rng(0)
                boot = np.array([
                    f[np.isin(ev, rng.choice(uniq, uniq.size, replace=True))].mean()
                    for _ in range(200)
                ]) if uniq.size > 1 else np.array([f.mean()])
                rows += _rows(ds, "particle_class", "fragmentation", algo, [code], [f.mean()],
                              [np.percentile(boot, 15.865)], [np.percentile(boot, 84.135)])

        # 7 isolation. `dr_min` comes from generator momenta, so crowding is a property of the
        # event. The panel key is the pT slice, since every panel plots the same quantity.
        for lo_pt, hi_pt, key, _ in ISOLATION_PT_SLICES:
            for algo in ISOLATION_ALGOS:
                p = parts[(parts.algo == algo) & np.isfinite(parts.dr_min)
                          & (parts.p_pt >= lo_pt) & (parts.p_pt < hi_pt)]
                if p.empty:
                    continue
                x, y, lo, hi = th.binned_proportion(p.dr_min.to_numpy(), (p.eff_e >= 0.5).to_numpy(),
                                                    DR_MIN_BINS)
                rows += _rows(ds, "isolation", key, algo, x, y, lo, hi)

        # 5 energy_budget: three scalars per method, so x stays NaN. Weighted by
        # `eff_e * e_dep_calib`, not `e_reco_calib`, which is what makes the three fates close.
        for algo in METHODS:
            p = parts[parts.algo == algo]
            if p.empty:
                continue
            tot = p.e_dep_calib.sum()
            vals = [
                (p.eff_e * p.e_dep_calib).sum() / tot,
                p.e_lost_other.sum() / tot,
                p.e_lost_noise.sum() / tot,
            ]
            assert abs(sum(vals) - 1.0) < 1e-6, f"fates for {algo} sum to {sum(vals):.4f}, not 1"
            for panel, v in zip(("recovered", "taken by another cluster", "never claimed"), vals, strict=True):
                rows += _rows(ds, "energy_budget", panel, algo, [np.nan], [v])

    # 2 cluster_size: a ratio of sums per bin, not a mean of per-particle counts.
    if anat is not None and not anat.empty:
        a = anat[anat.dataset == ds]
        for algo in METHODS:
            s = a[a.algo == algo]
            if s.empty:
                continue
            x, y, lo, hi = th.binned_ratio(s.p_pt.to_numpy(), s.n_recovered.to_numpy(),
                                           s.n_true.to_numpy(), s.sample_id.to_numpy(), th.E_BINS)
            rows += _rows(ds, "cluster_size", "size", algo, x, y, lo, hi)
            # The matched cluster's size against the shower's, which unlike the recovered
            # fraction is not bounded above by 1, so fragmenting and merging are distinguishable.
            x, y, lo, hi = th.binned_ratio(s.p_pt.to_numpy(), s.n_matched_cluster.to_numpy(),
                                           s.n_true.to_numpy(), s.sample_id.to_numpy(), th.E_BINS)
            rows += _rows(ds, "cluster_size", "size_ratio", algo, x, y, lo, hi)

    # 3 shower_profile. The `_taken` panel is cumulative, so the pair of curves closes each bin
    # against 1; see profile_with_interval.
    if profiles is not None:
        for panel, edges in (("dr", an.DR_EDGES), ("depth", an.DEPTH_EDGES)):
            for algo in METHODS:
                c, out = profile_with_interval(profiles[algo], panel, edges)
                for name, suffix in (("recovered", ""), ("recovered_taken", "_taken")):
                    frac, lo, hi = out[name]
                    rows += _rows(ds, "shower_profile", panel + suffix, algo, c, frac, lo, hi)

    # 8 jets. x is the reference jet pT throughout, so every row is binned by the same
    # method-independent quantity.
    if jets is not None and not jets.empty:
        d = jets[jets.dataset == ds]
        for algo in METHODS:
            a = d[d.algo == algo]
            if a.empty:
                continue
            x, y, lo, hi = th.binned_proportion(a.ref_pt.to_numpy(), a.matched.to_numpy(),
                                                th.JET_PT_BINS, min_count=10)
            rows += _rows(ds, "jets", "efficiency", algo, x, y, lo, hi)
            m = a[a.matched]
            if m.empty:
                continue
            ratio = (m.reco_pt / m.ref_pt).to_numpy()
            ref, ev = m.ref_pt.to_numpy(), m.sample_id.to_numpy()
            for panel, stat in (("median", "median"), ("resolution", "resolution")):
                x, v, lo, hi = th.binned_bootstrap(ref, ratio, ev, th.JET_PT_BINS, stat, min_count=10)
                rows += _rows(ds, "jets", panel, algo, x, v, lo, hi)

    return pd.DataFrame(rows, columns=list(SUMMARY_COLS))


def series(summary, ds, figure, panel, algo):
    """The (x, y, lo, hi) of one series, or four empty arrays if this dataset never had it."""
    s = summary[(summary.dataset == ds) & (summary.figure == figure)
                & (summary.panel == panel) & (summary.algo == algo)]
    if s.empty:
        e = np.array([])
        return e, e, e, e
    s = s.sort_values("x", kind="stable")
    return s.x.to_numpy(), s.y.to_numpy(), s.lo.to_numpy(), s.hi.to_numpy()


#: The bin edges each (figure, panel) was measured over, so a series can be drawn as the top edge
#: of a histogram. Kept here rather than as summary columns, so a committed figure_summary.csv
#: from another machine redraws unchanged.
EDGES: Mapping[tuple[str, str], np.ndarray] = {
    **{("eff_purity", p): th.E_BINS for p in ("efficiency", "purity")},
    **{("cluster_size", p): th.E_BINS for p in ("size", "size_ratio")},
    **{("response", p): th.E_BINS for p in ("median", "resolution")},
    **{("jets", p): th.JET_PT_BINS for p in ("efficiency", "median", "resolution")},
    ("shower_profile", "dr"): an.DR_EDGES,
    ("shower_profile", "dr_taken"): an.DR_EDGES,
    ("shower_profile", "depth"): an.DEPTH_EDGES,
    ("shower_profile", "depth_taken"): an.DEPTH_EDGES,
}


def _draw_series(ax, summary, ds, figure, panel, edges=None, band: bool = True, **kwargs):
    """Draw every method's series for one panel, band first so the steps sit on top."""
    if edges is None:
        edges = EDGES[(figure, panel)]
    for algo in METHODS:
        x, y, lo, hi = series(summary, ds, figure, panel, algo)
        if len(x) == 0:
            continue
        elo, ehi = th.bin_edges_for(x, edges)
        if band and np.isfinite(lo).any():
            th.band_steps(ax, algo, elo, ehi, lo, hi)
        th.draw_steps(ax, algo, elo, ehi, y, **kwargs)


# ------------------------------------------------------------------ figures

def fig_eff_purity(summary, datasets, out):
    """Efficiency and purity against truth particle pT.

    No reference clusterings are drawn: neither bounds both metrics, so a line for either would be
    crossed and read as a ceiling anyway. The resolution reference is quoted in the report instead.
    """
    fig, axes = th.grid(2, len(datasets), datasets)
    for j, ds in enumerate(datasets):
        _draw_series(axes[0][j], summary, ds, "eff_purity", "efficiency")
        _draw_series(axes[1][j], summary, ds, "eff_purity", "purity")
        axes[1][j].set_xlabel(r"truth particle $p_{\rm T}$ [GeV]")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("efficiency")
    # "matched-cluster", not "purity": the efficiency row is a rate over all targets while this
    # one is conditioned on a match, so no fake cluster appears in it.
    axes[1][0].set_ylabel("matched-cluster purity")
    return th.finish(fig, out)


def fig_cluster_size(summary, datasets, out):
    """How each method gets the cluster wrong: too small, or too large.

    The upper row is the matched cluster's size over the shower's, which is unbounded. Unity is
    right, below it is fragmentation and above it is merging. The lower row is the fraction of the
    shower's cells the cluster holds, which is bounded by 1 and falls under either failure. Read
    together they give the composition. Both are ratios of sums.
    """
    fig, axes = th.grid(2, len(datasets), datasets, sharey="row")
    for j, ds in enumerate(datasets):
        _draw_series(axes[0][j], summary, ds, "cluster_size", "size_ratio")
        # Unity is what the top row is read against.
        axes[0][j].axhline(1.0, color="k", lw=0.5, ls=":")
        _draw_series(axes[1][j], summary, ds, "cluster_size", "size")
        axes[1][j].set_ylim(0, 1.0)
        axes[1][j].set_xlabel(r"truth particle $p_{\rm T}$ [GeV]")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log")
    axes[0][0].set_ylabel("cluster size /\nshower size")
    axes[1][0].set_ylabel("fraction of the shower's\ncells correctly assigned")
    return th.finish(fig, out, ncol=len(METHODS))


def fig_shower_profile(summary, datasets, out):
    """Where in a shower the energy goes: transverse first, then longitudinal.

    Two curves per method: recovered, and cumulatively recovered plus taken by another cluster, so
    each panel partitions into the same three fates the energy-budget figure shows for the event
    as a whole. Only the recovered curve carries a band.

    Colour and dash say which method and weight says which curve, so the key is in two parts: the
    methods, then two split-colour entries naming the curves for both at once.

    The axes are not uniform in energy, most of a shower sitting in the first few dR bins, so
    these panels show shape rather than budget.
    """
    # sharex=False: the rows are different coordinates, not the same one at two scales.
    fig, axes = th.grid(2, len(datasets), datasets, sharex=False)
    for j, ds in enumerate(datasets):
        for row, (coord, xlabel) in enumerate(
            [("dr", "$\\Delta R$ from shower axis"),
             ("depth", "layers past shower start")]
        ):
            ax = axes[row][j]
            # The cumulative curve is a second reading of the same series rather than a third
            # method, so it keeps the method's colour and dash and is separated by weight alone.
            _draw_series(ax, summary, ds, "shower_profile", coord, linewidth=1.3)
            _draw_series(ax, summary, ds, "shower_profile", coord + "_taken", band=False,
                         linewidth=0.6, alpha=0.55, label="_nolegend_")
            # No line at unity: the fraction cannot exceed 1, so the top of the panel is already
            # unity. The limit sits just above so the upper curve clears the spine.
            ax.set_ylim(0, 1.04)
            ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel("fraction of the\nshower's energy")
    axes[1][0].set_ylabel("fraction of the\nshower's energy")

    # These entries describe a weight both methods use, so each is a key split down its middle.
    # Appended after the series so the methods stay first in the legend.
    extra = [(th.SplitKey(METHODS, linewidth=lw), text)
             for lw, text in ((1.3, "recovered"), (0.6, "recovered + taken by another cluster"))]
    # Widened because each swatch carries two segments; at the default, half is too short to
    # show a dash pattern.
    return th.finish(fig, out, ncol=len(METHODS), extra=extra, handlelength=3.0)


def fig_response(summary, datasets, out):
    """Bias against variance: the two methods fail in different currencies."""
    fig, axes = th.grid(2, len(datasets), datasets, sharey="row")
    for j, ds in enumerate(datasets):
        _draw_series(axes[0][j], summary, ds, "response", "median")
        _draw_series(axes[1][j], summary, ds, "response", "resolution")
        axes[0][j].axhline(1.0, color="k", lw=0.5, ls=":")
        axes[0][j].set_ylim(0.4, 2.0)
        axes[1][j].set_yscale("log")
        # Labelled ticks must bracket the data, and unlabelled 2..9-per-decade minors between
        # them are what makes the axis read as logarithmic; a label at every tick removes that
        # cue. The limits are tightened to the range the two datasets occupy.
        axes[1][j].set_ylim(0.13, 1.6)
        axes[1][j].yaxis.set_major_locator(mticker.FixedLocator([0.2, 0.5, 1.0, 1.5]))
        axes[1][j].yaxis.set_major_formatter(mticker.FixedFormatter(["0.2", "0.5", "1.0", "1.5"]))
        axes[1][j].yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100))
        axes[1][j].yaxis.set_minor_formatter(mticker.NullFormatter())
        axes[1][j].set_xlabel(r"truth particle $p_{\rm T}$ [GeV]")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log")
    # E_dep, not E_true: the denominator is the energy the target deposited in the cells being
    # clustered, which is smaller and is what src.evaluation.metrics divides by.
    axes[0][0].set_ylabel("median $E_{\\rm reco}/E_{\\rm dep}$")
    axes[1][0].set_ylabel("$\\sigma_E/E$")
    return th.finish(fig, out)


def fig_energy_budget(summary, datasets, out):
    """The three fates of a target's energy: one stacked bar per method, darkest to lightest.

    Stacked rather than grouped, so how a method divides its energy reads in one sweep. The point
    is that the two methods lose the same total in different ways. Colour carries the method and
    lightness the fate.
    """
    fig, ax = plt.subplots(figsize=th.figsize_for(1))
    fates = ["recovered", "taken by another cluster", "never claimed"]
    alphas = (1.0, 0.55, 0.25)
    labels, rows = [], []
    for ds in datasets:
        for algo in METHODS:
            found = [series(summary, ds, "energy_budget", f, algo)[1] for f in fates]
            vals = [float(v[0]) if len(v) else None for v in found]
            if any(v is None for v in vals):
                continue
            # Re-checked here because a summary can arrive from another machine, and a budget
            # that does not sum to 1 would look entirely plausible on the page.
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
    # "target" rather than "particle": the denominator is one collapsed TARGET as the thesis
    # defines it, which is not every simulated particle in the event.
    ax.set_xlabel("fraction of the target's deposited energy")
    ax.invert_yaxis()

    # Lightness means the same thing on both methods' bars, so each key is split down its middle
    # rather than drawn in one method's colour.
    handles = [th.SplitKey(METHODS, patch=True, alpha=a) for a in alphas]
    fig.legend(handles, fates, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               handler_map=th.HANDLER_MAP, handlelength=2.4)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"))
    plt.close(fig)
    return out


def fig_particle_class(summary, datasets, out):
    """Efficiency and fragmentation by particle type.

    Read the efficiency panel for the level and the fragmentation panel for the failure mode: the
    pairing separates "misses the particle" from "finds it in pieces". Bars rather than a line,
    the x axis being categorical. See PARTICLE_GROUPS for what is pooled and what is not drawn.
    """
    fig, axes = th.grid(2, len(datasets), datasets, sharey="row", sharex=False)
    codes = list(range(len(PARTICLE_GROUPS)))
    width = 0.38
    for j, ds in enumerate(datasets):
        present = [c for c in codes
                   if len(summary[(summary.dataset == ds) & (summary.figure == "particle_class")
                                  & (summary.x == c)])]
        pos = np.arange(len(present), dtype=float)
        for row, panel in enumerate(("efficiency", "fragmentation")):
            for i, algo in enumerate(METHODS):
                x, y, lo, hi = series(summary, ds, "particle_class", panel, algo)
                if len(x) == 0:
                    continue
                order = [list(x).index(c) for c in present if c in list(x)]
                yv = np.asarray(y)[order]
                err = np.vstack([yv - np.asarray(lo)[order], np.asarray(hi)[order] - yv])
                offset = (i - (len(METHODS) - 1) / 2) * width
                axes[row][j].bar(pos + offset, yv, width, color=th.colour(algo),
                                 label=th.LABELS.get(algo, algo), alpha=0.9)
                axes[row][j].errorbar(pos + offset, yv, yerr=np.abs(err), fmt="none",
                                      ecolor="#333333", elinewidth=0.8, capsize=2)
            axes[row][j].set_xticks(pos, [GROUP_LABELS[c] for c in present])
    # One limit per row, set after every bar is drawn: under `sharey="row"` the first set_ylim
    # freezes the row, so setting it inside the dataset loop would pin both columns to whichever
    # was drawn first. 0-1 rather than a data-driven limit, since both rows are fractions.
    for row in range(2):
        axes[row][0].set_ylim(0, 1.0)
    axes[0][0].set_ylabel("efficiency")
    axes[1][0].set_ylabel("fragmentation\n(energy outside largest piece)")
    return th.finish(fig, out, ncol=len(METHODS))


def fig_isolation(summary, datasets, out):
    """Efficiency against how crowded each target is, one row per pT slice.

    The slicing is the argument: pooled over pT the isolation axis selects soft targets rather
    than uncrowded ones beyond its peak. No reference clusterings are drawn.
    """
    # sharex=True, not "col": the dR range each dataset covers is itself a result, and a
    # per-column axis would rescale the truncated pileup-200 panels to fill the width, making the
    # same horizontal position mean a different dR in each column.
    fig, axes = th.grid(len(ISOLATION_PT_SLICES), len(datasets), datasets, sharey=True, sharex=True)
    for j, ds in enumerate(datasets):
        for row, (_, _, key, label) in enumerate(ISOLATION_PT_SLICES):
            ax = axes[row][j]
            for algo in ISOLATION_ALGOS:
                x, y, lo, hi = series(summary, ds, "isolation", key, algo)
                if len(x) == 0:
                    continue
                elo, ehi = th.bin_edges_for(x, DR_MIN_BINS)
                if np.isfinite(lo).any():
                    th.band_steps(ax, algo, elo, ehi, lo, hi)
                th.draw_steps(ax, algo, elo, ehi, y)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            # Named inside the panel: the column already carries the dataset name as its title.
            ax.text(0.03, 0.93, label, transform=ax.transAxes, va="top", ha="left", fontsize="small")
        axes[-1][j].set_xlabel(r"$\Delta R$ to nearest other target")
    for row in range(len(ISOLATION_PT_SLICES)):
        axes[row][0].set_ylabel("efficiency")
    return th.finish(fig, out, ncol=len(ISOLATION_ALGOS))


def fig_jets(summary, datasets, out):
    """Do the per-cluster errors survive into jets?

    Three rows, for the three separable ways a jet can fail: missing entirely, wrong energy scale,
    imprecise energy. They mirror the cluster-level response figure, so the two can be read
    against each other. x is the reference jet pT throughout.
    """
    fig, axes = th.grid(3, len(datasets), datasets, sharey="row")
    xmax = 0.0
    for j, ds in enumerate(datasets):
        for row, panel in enumerate(("efficiency", "median", "resolution")):
            _draw_series(axes[row][j], summary, ds, "jets", panel)
        for algo in METHODS:
            x, *_ = series(summary, ds, "jets", "efficiency", algo)
            if len(x):
                # The upper edge of the last populated bin, not its centre: the series is a step
                # spanning the whole bin, and stopping at the centre would clip it.
                xmax = max(xmax, float(th.bin_edges_for(x, th.JET_PT_BINS)[1].max()))
        axes[0][j].set_ylim(0, 1.02)
        # No hardcoded ylim on the response row: a limit that can hide a series is worse than an
        # untidy axis.
        axes[1][j].axhline(1.0, color="k", lw=0.5, ls=":")
        ticks = [t for t in (25, 35, 50, 75, 100, 150, 200, 300, 400) if t <= xmax]
        for row in range(3):
            axes[row][j].set_xscale("log")
            axes[row][j].set_xlim(th.JET_MIN_PT * 0.96, xmax * 1.04)
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
    ap.add_argument("--events", type=int, default=None,
                    help="events for the store-derived figures; defaults to this dataset's "
                         "windows.eval_events, which the committed summaries were built over")
    ap.add_argument("--rebuild-anatomy", action="store_true")
    ap.add_argument("--rebuild-jets", action="store_true")
    ap.add_argument("--from-summary", action="store_true",
                    help="draw every column from results/<ds>/figure_summary.csv and read nothing "
                         "else; use this on a machine where the analysis was not run")
    ap.add_argument("--latex", action="store_true",
                    help="render text with a real LaTeX installation (requires one to be present)")
    args = ap.parse_args()

    th.apply(latex=True if args.latex else None)
    OUT.mkdir(parents=True, exist_ok=True)

    active = settings()["dataset"]["active"]
    summaries = []
    for ds in args.datasets:
        res = Path("results") / ds
        if not res.is_dir():
            print(f"  ! {ds}: no results/{ds}/, skipping this column", flush=True)
            continue
        summary_file = res / SUMMARY_NAME

        # Derived from the window the store was dumped over, so a rebuild cannot quietly write
        # rows computed on fewer events over a summary built on the full window.
        n_events = args.events if args.events is not None else \
            settings()["dataset"][ds]["windows"]["eval_events"]

        # Each source is optional and they are gathered independently, so a dataset contributes
        # whatever it has rather than being dropped for one missing table.
        tables, anat, profiles, jets = {}, None, None, None

        # A hard short-circuit: neither the store nor the per-row tables are read. Skipping only
        # the store would let a machine with partial tables write an incomplete summary over the
        # committed one.
        if args.from_summary:
            if summary_file.exists():
                print(f"  {ds}: drawing from the committed {summary_file.name}", flush=True)
                summaries.append(pd.read_csv(summary_file))
            else:
                print(f"  ! {ds}: no {SUMMARY_NAME}, skipping this column", flush=True)
            continue

        parts = sorted(res.glob("particles_*.parquet"))
        if parts:
            tables[ds] = (
                pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True),
                pd.concat([pd.read_parquet(p) for p in sorted(res.glob("clusters_*.parquet"))],
                          ignore_index=True),
            )

        # The store is touched only for the active dataset. Jets, the shower anatomy and the
        # profiles cannot be recovered from the per-row tables; `load_profiles` has no cache, so a
        # machine without the store must use --from-summary.
        jcache = res / "jets.parquet"
        if ds == active and (args.rebuild_jets or not jcache.exists()):
            print(f"  building jet cache for {ds} ({n_events} events) ...", flush=True)
            build_jets(ds, settings(), n_events).to_parquet(jcache)
        if jcache.exists():
            jets = pd.read_parquet(jcache)

        cache = res / "anatomy_particles.parquet"
        if ds == active:
            if args.rebuild_anatomy or not cache.exists():
                print(f"  building anatomy cache for {ds} ({n_events} events) ...", flush=True)
                build_anatomy(ds, settings(), n_events).to_parquet(cache)
            profiles = load_profiles(ds, settings(), n_events)
        if cache.exists():
            anat = pd.read_parquet(cache)

        # Rebuild from the per-row tables when this machine has them; local tables are the
        # fresher artefact, and a stale summary plotting over a rescore is the failure to avoid.
        built = build_summary(ds, tables, anat, profiles, jets)
        if not built.empty:
            built.to_csv(summary_file, index=False, float_format="%.6g")
            print(f"  {ds}: summary rebuilt -> {summary_file} ({len(built)} rows)", flush=True)
            summaries.append(built)
        elif summary_file.exists():
            print(f"  {ds}: no local tables; drawing from the committed {summary_file.name}", flush=True)
            summaries.append(pd.read_csv(summary_file))
        else:
            print(f"  ! {ds}: no tables and no {SUMMARY_NAME}, skipping this column "
                  f"(dump its store and run scripts.score, or copy the summary across)", flush=True)

    if not summaries:
        raise SystemExit("no dataset had tables or a committed summary; nothing to draw")
    summary = pd.concat(summaries, ignore_index=True)

    def drawn(figure: str) -> list[str]:
        """The datasets carrying rows for one figure, in the requested order."""
        have = set(summary[summary.figure == figure].dataset)
        return [d for d in args.datasets if d in have]

    for n, (figure, fn, stem) in enumerate([
        ("eff_purity", fig_eff_purity, "fig_eff_purity"),
        ("cluster_size", fig_cluster_size, "fig_cluster_size"),
        ("shower_profile", fig_shower_profile, "fig_shower_profile"),
        ("response", fig_response, "fig_response"),
        ("energy_budget", fig_energy_budget, "fig_energy_budget"),
        ("particle_class", fig_particle_class, "fig_particle_class"),
        ("isolation", fig_isolation, "fig_isolation"),
        ("jets", fig_jets, "fig_jets"),
    ], start=1):
        ds_list = drawn(figure)
        if not ds_list:
            print(f"{n} {figure:<15} skipped: no dataset has rows for it", flush=True)
            continue
        print(f"{n} {figure:<15}", fn(summary, ds_list, OUT / stem), flush=True)

    print(f"\ndatasets drawn: {sorted(set(summary.dataset))}")


if __name__ == "__main__":
    main()
