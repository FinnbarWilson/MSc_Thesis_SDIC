"""The thesis figures, pileup 0 and pileup 200 side by side.

    python -m scripts.make_thesis_figures                 # every dataset that has tables
    python -m scripts.make_thesis_figures --datasets pu200
    python -m scripts.make_thesis_figures --rebuild-anatomy

Writes figures/thesis/*.pdf|png -- one directory, because these are cross-dataset and do not
belong under either dataset's own folder.

WHAT IS HERE

Eight figures, and they are the eight the thesis uses -- nothing here is drawn and unused:

  1 eff_purity      efficiency and purity against particle pT
  2 cluster_size    cluster size against shower size, and cells recovered -- fragmenting vs merging
  3 shower_profile  where in the shower the energy is lost: transverse, then longitudinal
  4 response        energy response and resolution -- bias against variance
  5 energy_budget   the three fates of a particle's energy
  6 particle_class  efficiency and fragmentation per particle type
  7 isolation       efficiency against crowding, with the resolution ceiling
  8 jets            do the per-cluster errors survive into an observable

DATA SOURCES, AND WHY THERE IS A SUMMARY BETWEEN THEM AND THE FIGURES

Nothing here plots from a per-row table directly. Every dataset is first reduced to
results/<ds>/figure_summary.csv -- a few hundred binned points, ~20 KB -- and the figures draw
from that and nothing else. build_summary() is the only place the aggregation happens.

That indirection exists because THE TWO COLUMNS ARE PRODUCED ON DIFFERENT MACHINES: pu0 is
trained and scored on DIAS, pu200 on ce-ai-1, and whichever machine draws the figure needs both.
The per-row tables cannot make that trip -- particles_*.parquet and clusters_*.parquet are ~200 MB
at pu0 and several times that at pu200, .gitignore excludes them, and they have never been in git
history for either dataset. The summary is small enough to commit, so `git push` on one cluster and
`git pull` on the other is the whole transfer.

What still needs local data, and so is rebuilt only for the ACTIVE dataset:

  results/<ds>/anatomy_particles.parquet   figures 2-3; needs cell positions from the event store
  results/<ds>/jets.parquet                figure 8; anti-k_t jets per event

Both are caches rather than results (--rebuild-anatomy / --rebuild-jets), and both are folded into
the summary, so a machine holding only the summary still draws all eight figures. A machine that HAS
the per-row tables always rebuilds the summary rather than trusting the committed one: local tables
are the fresher artefact, and a stale summary quietly surviving a rescore is the failure mode this
arrangement is meant to prevent.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Mapping
from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import settings, store_expectations, store_path
from src.evaluation.differential import clopper_pearson
from src.evaluation import anatomy as an
from src.evaluation import jets as jt
from src.evaluation.matching import hungarian_match, overlap_matrix
from src.io.event_store import EventStore
from src.plotting import thesis as th

#: The comparison the thesis makes: one learned method against one classical one.
METHODS = ("maskformer", "clue")
OUT = Path("figures/thesis")


# ------------------------------------------------------------------ anatomy cache

def _labels(record, method, cfg, clue_params):
    mf = cfg["maskformer"]
    if method == "clue":
        # IMPORTED HERE, NOT AT MODULE SCOPE. src.clue.pipeline imports CLUEstering, a compiled
        # package needed only to re-cluster events from the store. Importing it at the top would
        # make it a hard requirement of drawing the figures, and --from-summary exists precisely so
        # the figures can be drawn on a machine carrying nothing but the repository.
        from src.clue.pipeline import cluster_event

        return cluster_event(record, clue_params, coords=cfg["clue"]["coords"],
                             backend=cfg["clue"]["backend"], min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
                             link_radius=cfg["clue"].get("link_radius", 0.0))
    return record.maskformer_labels(mask_threshold=mf["mask_threshold"],
                                   object_threshold=mf["object_threshold"],
                                   min_cluster_hits=cfg["metrics"]["min_cluster_hits"])


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
    """Pooled shower-frame profiles, which are aggregates and so are not worth caching per row."""
    clue_params = {k: v["parameters"] for k, v in
                   json.loads((Path("results") / dataset / "clue_parameters.json").read_text())["subsystems"].items()}
    store = EventStore(store_path("store"), expect=store_expectations())
    per = {m: [] for m in METHODS}
    # The event index is carried alongside, because it is the resampling unit for the profile's
    # uncertainty band. `shower_cells` returns one event's pairs and cannot know its own index.
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
    """(centres, {series: (fraction, lo, hi)}) for a shower profile, resampled over EVENTS.

    The profile is a RATIO OF SUMS in each bin -- energy of a given fate over deposited energy --
    so the interval has to be built by resampling whole events and recomputing the ratio, exactly
    as `th.binned_ratio` does for the cluster-size figure. Resampling cells or particles would
    understate it: a shower contributes many correlated cells and an event many correlated showers.

    This figure carried no uncertainty at all until now, and the discussion leans on one bin of it
    -- CLUE falling back in the deepest layers -- which is precisely the kind of claim that needs a
    band to be worth making.

    TWO SERIES, NOT ONE, and the second is CUMULATIVE. Drawing only the recovered fraction leaves
    the rest of each bin unaccounted for, so a reader cannot tell energy the method gave to a
    neighbouring cluster from energy it left in no cluster at all -- which is the distinction the
    whole discussion of failure modes rests on, and which figure 5 already makes for the event as a
    whole. `recovered_taken` is recovered PLUS taken by another cluster, so the two curves and the
    line at 1 partition each bin into the same three fates as that figure: recovered below the
    first curve, taken between the two, and unclustered above the second. The cumulative form is
    what makes that reading work; three independent curves per method would be six lines a panel
    and would not visibly close.

    Both series come out of one pass. `an.profile` already splits every bin by fate and the
    bootstrap already builds a per-(event, bin) matrix, so the second series costs one more
    numerator matrix rather than a second scan over several million cell rows.
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

    # The three fates are exhaustive per cell, so the cumulative curve cannot exceed 1 and cannot
    # sit below the recovered one. A figure that silently failed either would be entirely plausible
    # on the page, which is the same reason fig_energy_budget asserts its own decomposition.
    rec, tak = point["recovered"], point["recovered_taken"]
    ok = np.isfinite(rec) & np.isfinite(tak)
    assert np.all(tak[ok] >= rec[ok] - 1e-9), "recovered exceeds recovered+taken in some bin"
    assert np.all(tak[ok] <= 1.0 + 1e-9), "recovered+taken exceeds the deposited energy in some bin"
    return centres, out


# ------------------------------------------------------------------ the figure summary
#
# WHAT THIS IS FOR. pu0 is trained and scored on one cluster, pu200 on another, and the figures put
# them side by side -- so one machine has to hold both columns. The per-row tables cannot travel:
# particles_*.parquet and clusters_*.parquet are ~200 MB at pu0 and several times that at pu200,
# .gitignore excludes them, and they have never been in git history for either dataset. Carrying
# them by hand between clusters for every rescore is the thing this file exists to stop.
#
# Every one of these figures is, in the end, a handful of BINNED SERIES: binned_proportion,
# binned_bootstrap and binned_ratio all return (x, y, lo, hi), the shower profile returns (x, y),
# and the energy budget is three scalars per method. That is a few hundred numbers per dataset --
# ~20 KB of CSV against ~200 MB of parquet, and small enough to commit.
#
# SO THE AGGREGATION HAPPENS EXACTLY ONCE, HERE, and the figures draw from its output and nothing
# else. A machine WITH the per-row tables rebuilds the summary and plots; a machine with only the
# committed summary plots directly. There is no second code path that could drift from this one --
# the numbers a reader sees are the numbers in the file either way.
#
# The format is deliberately long/tidy rather than one column per series: adding a method or a
# panel then costs no schema change.

SUMMARY_NAME = "figure_summary.csv"
SUMMARY_COLS = ("dataset", "figure", "panel", "algo", "x", "y", "lo", "hi")

#: The photon/electron split is NOT drawn, and the charged/neutral hadron split IS. The two look
#: like the same kind of distinction and are not.
#:
#: Photon against electron fails because the two ShowERS are identical -- a calorimeter has no way
#: to tell them apart, the difference lives in the tracker -- and because the split cannot be
#: controlled for anyway. Electrons sit at a median separation of 0.004 from their nearest target
#: against photons' 0.064, and 41.5% of electrons have a neighbour inside dR 0.002 where only ~4%
#: of photons do. The two populations barely share a crowding regime, so the raw 0.208 efficiency
#: gap standardises to +0.094, +0.018 or -0.043 depending on which distribution you standardise to.
#: A quantity whose sign depends on the weighting is not measuring the clustering.
#:
#: Charged against neutral hadron survives both objections. The split is not about what the
#: calorimeter can IDENTIFY -- it cannot -- but about which targets are hard to cluster and which
#: matter downstream, and on both counts they differ. A neutral hadron has no track, so a
#: particle-flow chain must take it from the calorimeter alone; it is the component that limits jet
#: energy resolution. Their dR distributions are also alike (median 0.111 against 0.113), so the
#: comparison is not confounded the way the EM one is.
#:
#: And it shows something the pooled hadronic bar hides: CLUE does BETTER on neutral hadrons than
#: charged (0.551 against 0.504) while MaskFormer does WORSE (0.650 against 0.709). The two methods'
#: hardest hadron is not the same hadron.
#:
#: MUONS ARE NOT DRAWN. They deposit 0.32% of the target energy being clustered -- not the 3.1%
#: that an earlier version of this comment claimed, which was their INCIDENT energy, most of which
#: leaves the calorimeter by definition -- and a calorimeter clustering method is not asked to
#: reconstruct them. The muon result is a real diagnostic of a density criterion and it is worth
#: reporting, so it stays in the per-class table and in the text; it does not earn a third of the
#: width of the headline class figure.
PARTICLE_GROUPS: Mapping[str, tuple[int, ...]] = {
    "electromagnetic": (0, 1),
    "charged_hadron": (5,),
    "neutral_hadron": (6,),
}
GROUP_LABELS: tuple[str, ...] = (
    "electromagnetic\n($\\gamma$, e)", "charged\nhadron", "neutral\nhadron",
)

#: Isolation bins for the crowding figure. The first edge is 0.005 rather than 0 because bin centres
#: are geometric means, which are undefined at zero. Targets with no other target in the event have
#: an infinite separation and are dropped rather than piled into the last bin.
#:
#: Coarser than the nine bins used when this figure was pooled, because it is now cut three ways in
#: pT as well and the bins have to keep enough targets to carry an interval.
DR_MIN_BINS = np.array([0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 5.0])

#: THE CROWDING FIGURE IS CUT IN pT, and that is the whole design.
#:
#: Pooled over all targets the two method curves are NOT monotonic: they peak near dR 0.07 and fall
#: away above it. That shape is an artefact. Targets isolated by more than ~0.1 are predominantly
#: soft -- 39% of targets in the softest pT bin are isolated against 1.4% at 5-10 GeV -- so beyond
#: the peak the isolation axis stops selecting on crowding and starts selecting on energy.
#:
#: The previous version drew two reference clusterings alongside, on the reasoning that their
#: monotonic rise showed the pure crowding effect the method curves could not. That did not work.
#: The resolution ceiling is 0.983 at dR 0.01 and 1.000 everywhere else -- a flat line at the top of
#: the frame carrying no information. The geometric ceiling is worse than uninformative: it assigns
#: each cell to the nearest shower axis given perfect seeds, which is one specific and suboptimal
#: rule that methods beat elsewhere in this study, so it is not a bound, and a dashed line drawn
#: above both methods on a plot about what is achievable will be read as one whatever the caption
#: says.
#:
#: Cutting in pT fixes the confound at its source instead of annotating it. Within a slice the
#: curves are monotonic and the crowding effect is unambiguous -- above 10 GeV MaskFormer runs
#: 0.69 -> 0.98 and CLUE 0.50 -> 0.96 across the dR range -- and the statement rests on thousands of
#: targets rather than on the ~30 isolated ones a given high-pT bin holds.
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
    """Reduce one dataset's per-row tables to the binned series the five figures draw.

    Each block below is the aggregation that used to sit inline in the corresponding fig_*, moved
    here unchanged so the plotted numbers cannot depend on which source a machine had.
    """
    rows = []
    parts, clus = tables.get(ds, (None, None))

    if parts is not None:
        # 1 eff_purity. Purity is a per-CLUSTER quantity binned by the energy of the particle its
        # cluster was matched to, not by the cluster's own energy, so both rows share one x-axis
        # that means one thing. See fig_eff_purity's docstring.
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

        # 7 particle_class. x is the CLASS CODE rather than a continuous variable, so the binned
        # helpers do not apply -- they take a geometric bin centre, which is undefined at code 0.
        # The statistics are the same ones they use: Clopper-Pearson for the efficiency, which is a
        # proportion, and an event-level bootstrap for the fragmentation, which is a mean.
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

                # Fragmentation is `frag_frac`, the share of a particle's energy NOT in its largest
                # piece. Preferred over the split-rate flag because that counts clusters holding
                # more than a tenth of the particle and so cannot see a shower shattered into more
                # than ten -- exactly the failure this panel exists to show.
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

        # 8 isolation. Efficiency against how crowded each target is, which is a property of the
        # EVENT rather than of either method -- `dr_min` comes from generator momenta.
        #
        # CUT IN pT, one panel per slice. Pooling confounds crowding with energy and turns the
        # curves non-monotonic; see ISOLATION_PT_SLICES for the full argument. The panel key is the
        # slice rather than the quantity, since every panel plots the same quantity.
        for lo_pt, hi_pt, key, _ in ISOLATION_PT_SLICES:
            for algo in ISOLATION_ALGOS:
                p = parts[(parts.algo == algo) & np.isfinite(parts.dr_min)
                          & (parts.p_pt >= lo_pt) & (parts.p_pt < hi_pt)]
                if p.empty:
                    continue
                x, y, lo, hi = th.binned_proportion(p.dr_min.to_numpy(), (p.eff_e >= 0.5).to_numpy(),
                                                    DR_MIN_BINS)
                rows += _rows(ds, "isolation", key, algo, x, y, lo, hi)

        # 5 energy_budget: three scalars per method, so x is meaningless and stays NaN.
        # `eff_e * e_dep_calib`, NOT `e_reco_calib` -- see fig_energy_budget's comment; only the
        # energy-weighted efficiency makes the decomposition close to 1 exactly.
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

    # 2 cluster_size: a RATIO OF SUMS per bin, not a mean of per-particle counts -- the per-particle
    # distribution is bimodal for CLUE and a mean over it reversed the sign of the trend.
    if anat is not None and not anat.empty:
        a = anat[anat.dataset == ds]
        for algo in METHODS:
            s = a[a.algo == algo]
            if s.empty:
                continue
            x, y, lo, hi = th.binned_ratio(s.p_pt.to_numpy(), s.n_recovered.to_numpy(),
                                           s.n_true.to_numpy(), s.sample_id.to_numpy(), th.E_BINS)
            rows += _rows(ds, "cluster_size", "size", algo, x, y, lo, hi)
            # The matched cluster's SIZE against the shower's, which unlike the recovered fraction
            # is not bounded above by 1. That is the whole point of carrying both: a method can be
            # wrong by building a cluster too small (fragmenting) or too large (merging), and the
            # recovered fraction alone cannot tell those apart -- it falls in both cases.
            x, y, lo, hi = th.binned_ratio(s.p_pt.to_numpy(), s.n_matched_cluster.to_numpy(),
                                           s.n_true.to_numpy(), s.sample_id.to_numpy(), th.E_BINS)
            rows += _rows(ds, "cluster_size", "size_ratio", algo, x, y, lo, hi)

    # 3 shower_profile: where a target's energy went, against two shower-frame coordinates. The
    # `_taken` panel is CUMULATIVE (recovered plus taken by another cluster), so the pair of curves
    # closes each bin against 1 -- see profile_with_interval.
    if profiles is not None:
        for panel, edges in (("dr", an.DR_EDGES), ("depth", an.DEPTH_EDGES)):
            for algo in METHODS:
                c, out = profile_with_interval(profiles[algo], panel, edges)
                for name, suffix in (("recovered", ""), ("recovered_taken", "_taken")):
                    frac, lo, hi = out[name]
                    rows += _rows(ds, "shower_profile", panel + suffix, algo, c, frac, lo, hi)

    # 6 jets. x is the REFERENCE jet pt throughout, so every row is binned by the same
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
#: of a histogram rather than as a line through bin centres. Every one of these is already a module
#: constant used by build_summary; the registry only records which figure used which, which the
#: summary itself does not carry. Keeping it here rather than adding xlo/xhi columns means a
#: committed figure_summary.csv from another machine redraws in the new style unchanged.
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
        # Purity is a per-CLUSTER quantity, but build_summary bins it by the energy of the particle
        # that cluster was matched to, not by the cluster's own energy. Both rows then share one
        # x-axis that means one thing. Binning the rows by different energies under a single
        # "particle energy" label is exactly the ambiguity these figures exist to remove -- and it
        # silently made the two rows non-comparable.
        _draw_series(axes[0][j], summary, ds, "eff_purity", "efficiency")
        _draw_series(axes[1][j], summary, ds, "eff_purity", "purity")
        axes[1][j].set_xlabel(r"truth particle $p_{\rm T}$ [GeV]")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("efficiency")
    axes[1][0].set_ylabel("purity")
    return th.finish(fig, out)


def fig_cluster_size(summary, datasets, out):
    """How each method gets the cluster wrong: too small, or too large.

    TWO ROWS, AND THE TOP ONE IS THE POINT. The lower row is the fraction of a shower's cells the
    matched cluster actually holds, which is bounded above by 1 and therefore falls whether a
    method fragments a shower or drowns it in a neighbour's cells. It cannot distinguish the two.

    The upper row is the matched cluster's SIZE against the shower's, which is not bounded. Unity
    means the cluster has the right number of cells; below is fragmentation, above is merging. The
    two rows read together give the composition: a method at size 1.9 holding 0.81 of the shower
    has a cluster of which the target is only 43%, and the rest belongs to somebody else.

    Both rows are RATIOS OF SUMS -- all cells in a bin over all truth cells in it -- rather than
    means of per-particle ratios. The per-particle distribution is bimodal for CLUE, which usually
    fragments a large shower and occasionally captures most of one, and a mean over those two
    behaviours describes neither; it reversed the sign of the trend. Summing first asks a question
    the shape of the distribution cannot distort.
    """
    fig, axes = th.grid(2, len(datasets), datasets, sharey="row")
    for j, ds in enumerate(datasets):
        _draw_series(axes[0][j], summary, ds, "cluster_size", "size_ratio")
        # Unity is the reference the top row exists to be read against, so it is drawn.
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

    THREE FATES PER BIN, NOT ONE. The earlier version drew the recovered fraction alone, which left
    the rest of every bin unexplained: a reader could see that a method held half a bin's energy
    and could not see whether the other half went to a neighbouring cluster or into no cluster at
    all. That distinction is the whole of the failure-mode argument, and figure 5 already makes it
    for the event as a whole, so this figure makes it again resolved in the shower frame.

    The second curve per method is CUMULATIVE -- recovered plus taken by another cluster -- so each
    panel partitions the same way figure 5's stacked bar does: below the lower curve is recovered,
    between the two is taken by another cluster, and between the upper curve and 1 is energy no
    cluster claimed. Six independent curves would not close visibly; two curves and a line at unity
    do. Only the recovered curve carries a band, because it is the one the discussion quotes and
    two bands per method is more shading than these panels can hold.

    THE AXES ARE NOT UNIFORM IN ENERGY, and the caption says so, because the figure invites the
    opposite reading. Most of a shower's energy sits in the first two or three dR bins and near the
    shower maximum in depth, so a shortfall in an outer bin costs a small fraction of the energy of
    the same shortfall near the core. The panels are shape, not budget; figure 5 is the budget.
    """
    # sharex=False: the rows are different coordinates, not the same one at two scales.
    fig, axes = th.grid(2, len(datasets), datasets, sharex=False)
    for j, ds in enumerate(datasets):
        for row, (coord, xlabel) in enumerate(
            [("dr", "$\\Delta R$ from shower axis"),
             ("depth", "layers past shower start")]
        ):
            ax = axes[row][j]
            _draw_series(ax, summary, ds, "shower_profile", coord)
            # Thinner, part-transparent, and unlabelled: it is a second reading of the same series
            # rather than a third method, and a legend entry per method per curve would double the
            # key for no information.
            _draw_series(ax, summary, ds, "shower_profile", coord + "_taken", band=False,
                         linewidth=0.7, alpha=0.55, label="_nolegend_")
            # Unity is what the pair of curves is read against: the distance from the upper curve
            # to this line is the energy left in no cluster.
            ax.axhline(1.0, color="k", lw=0.5, ls=":")
            ax.set_ylim(0, 1.08)
            ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel("fraction of the\nshower's energy")
    axes[1][0].set_ylabel("fraction of the\nshower's energy")
    return th.finish(fig, out, ncol=len(METHODS))


def fig_response(summary, datasets, out):
    """Bias against variance: the two methods fail in different currencies."""
    fig, axes = th.grid(2, len(datasets), datasets, sharey="row")
    for j, ds in enumerate(datasets):
        _draw_series(axes[0][j], summary, ds, "response", "median")
        _draw_series(axes[1][j], summary, ds, "response", "resolution")
        axes[0][j].axhline(1.0, color="k", lw=0.5, ls=":")
        axes[0][j].set_ylim(0.4, 2.0)
        axes[1][j].set_yscale("log")
        # TICKS MUST BRACKET THE DATA. These were [0.5, 1, 2, 5], and every MaskFormer point lies
        # between 0.142 and 0.322 -- the entire learned-method curve sat below the lowest label, on
        # a log axis with minor ticks off, so no value on it could be read. An axis whose labels do
        # not span the series it carries is worse than no axis.
        #
        # THEY MUST ALSO LOOK LOGARITHMIC, and the way to do that is UNLABELLED MINOR TICKS rather
        # than more labels. Two previous attempts labelled every tick -- [0.1, 0.2, 0.5, 1, 2],
        # then [0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5] -- and both read as a linear axis someone had
        # written arbitrary numbers on, because a label at every tick destroys the one visual cue
        # that says "log": the minor ticks crowding together as each decade closes, 0.2 to 0.3 wide
        # and 0.8 to 0.9 narrow. `minorticks_off()` had removed exactly that cue.
        #
        # So: round numbers on the labelled ticks, and the standard 2..9-per-decade minors between
        # them carrying no text. 0 cannot be one of the labels -- it does not exist on a log axis --
        # and 0.2 is kept alongside 0.5/1.0/1.5 because the whole pileup-0 MaskFormer curve lives
        # between 0.14 and 0.32, and an axis whose labels do not reach its data is the failure the
        # paragraph above is about.
        #
        # The range is tightened to what the two datasets actually occupy (0.139 to 1.252 including
        # bands) so the curves fill the panel rather than sitting in the middle third of it.
        axes[1][j].set_ylim(0.13, 1.6)
        axes[1][j].yaxis.set_major_locator(mticker.FixedLocator([0.2, 0.5, 1.0, 1.5]))
        axes[1][j].yaxis.set_major_formatter(mticker.FixedFormatter(["0.2", "0.5", "1.0", "1.5"]))
        axes[1][j].yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100))
        axes[1][j].yaxis.set_minor_formatter(mticker.NullFormatter())
        axes[1][j].set_xlabel(r"truth particle $p_{\rm T}$ [GeV]")
        for ax in (axes[0][j], axes[1][j]):
            ax.set_xscale("log")
    axes[0][0].set_ylabel("median $E_{\\rm reco}/E_{\\rm true}$")
    axes[1][0].set_ylabel("$\\sigma_E/E$")
    return th.finish(fig, out)


def fig_energy_budget(summary, datasets, out):
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
        for algo in METHODS:
            vals = [float(series(summary, ds, "energy_budget", f, algo)[1][0])
                    if len(series(summary, ds, "energy_budget", f, algo)[1]) else None
                    for f in fates]
            if any(v is None for v in vals):
                continue
            # The decomposition is built to close exactly -- see build_summary, which weights by
            # eff_e * e_dep_calib rather than e_reco_calib. Re-checked here because a summary can
            # arrive from another machine, and a budget that does not sum to 1 is the one error in
            # this figure that looks entirely plausible on the page.
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


def fig_particle_class(summary, datasets, out):
    """What each method is good at, by particle type. Grouped bars, not curves.

    THE FRAGMENTATION PANEL IS THE POINT, and the ordering in it inverts between the two halves of
    the table. CLUE fragments electromagnetic showers far LESS than MaskFormer and hadronic showers
    far MORE. That is a statement about the algorithms rather than about tuning: a density threshold
    is well matched to a compact single-peaked EM shower and shatters a multi-peaked hadronic one,
    while a learned model encodes no shower shape and is therefore flatter across every class.

    THE EFFICIENCY PANEL CARRIES A SECOND RESULT that the earlier grouping could not show. Split
    charged from neutral and the two methods disagree about which hadron is harder: MaskFormer
    falls from 0.709 to 0.650 going charged -> neutral, CLUE RISES from 0.504 to 0.551. Neutral
    hadrons are the group a particle-flow chain most needs from the calorimeter, having no track,
    so this is the bar with the most downstream weight and the one where the gap is narrowest.

    Read the efficiency panel for the level and the fragmentation panel for the failure mode; the
    pairing is what separates "misses the particle" from "finds it in pieces". See PARTICLE_GROUPS
    for why photon/electron is pooled and muons are not drawn.

    Bars rather than a line: the x axis is categorical, and joining categories with a line would
    imply an ordering between an EM shower and a hadron that does not exist.
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
    # ONE LIMIT PER ROW, SET AFTER EVERY BAR IS DRAWN. `sharey="row"` means the first set_ylim
    # freezes the whole row, so setting it inside the dataset loop pinned both columns to the
    # autoscale of whichever was drawn FIRST. The fragmentation row inherited pu0's 0.41 and cut
    # three pu200 bars off at the top, error bars and all -- CLUE on charged hadrons (0.444) and
    # MaskFormer on neutral (0.431) simply ran off the panel with nothing to say they had.
    #
    # 0-1 for both rows rather than a data-driven limit: efficiency and fragmentation are both
    # fractions of one particle's energy, so the full range is meaningful for each, it cannot clip
    # whatever a future dataset does, and it makes the two rows read on the same scale.
    for row in range(2):
        axes[row][0].set_ylim(0, 1.0)
    axes[0][0].set_ylabel("efficiency")
    axes[1][0].set_ylabel("fragmentation\n(energy outside largest piece)")
    return th.finish(fig, out, ncol=len(METHODS))


def fig_isolation(summary, datasets, out):
    """Efficiency against how crowded each target is, at fixed pT. Crowding is the limitation.

    ONE ROW PER pT SLICE, and the slicing is the argument rather than a refinement of it. Pooled
    over pT these curves peak near dR 0.07 and fall away, which reads as "isolation stops helping"
    and is not true: beyond the peak the isolation axis is selecting soft targets rather than
    uncrowded ones. Cut in pT the confound is gone and both methods rise monotonically in the two
    harder slices, MaskFormer 0.69 -> 0.98 and CLUE 0.50 -> 0.96 above 10 GeV.

    That is the evidence for the study's main claim about where the loss comes from: the flat
    efficiency-against-pT curve in figure 1 is not a model that fails to improve with energy, it is
    reconstruction getting easier while the environment gets harder, very nearly cancelling.

    WHAT THE SOFT SLICE SAYS, and it is a limit on the claim rather than support for it: at
    0.5-2 GeV efficiency does NOT rise with separation, it peaks around dR 0.05 and falls. A soft
    isolated particle is still hard, for reasons that are not crowding -- few cells, and a shower
    that a threshold can lose entirely. Crowding dominates above ~2 GeV and does not explain the
    softest bin, which is where most targets are.

    No reference clusterings: see ISOLATION_PT_SLICES for why they were removed.
    """
    # sharex=True, NOT "col". Every panel measures the same coordinate, and the dR RANGE each
    # dataset covers is itself a result: at pileup 200 no target above 10 GeV has a neighbour
    # beyond dR = 0.2 and only 14 beyond 0.1, so the binned series simply stops. Under a per-column
    # axis those panels rescaled to fill the width, and CLUE's rise to 0.85 read as "large
    # separation" when the axis ended at 0.07 -- the same horizontal position meant a different dR
    # in each column, in a figure whose entire grammar is comparison across columns. Sharing the
    # axis puts the truncation on the page, which is the honest version and the more informative
    # one: the empty right-hand side of the pileup-200 panels is the crowding result.
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
            # The slice is named inside the panel rather than as a title: the top of each column
            # already carries the dataset name, and a second title row would compete with it.
            ax.text(0.03, 0.93, label, transform=ax.transAxes, va="top", ha="left", fontsize="small")
        axes[-1][j].set_xlabel(r"$\Delta R$ to nearest other target")
    for row in range(len(ISOLATION_PT_SLICES)):
        axes[row][0].set_ylabel("efficiency")
    return th.finish(fig, out, ncol=len(ISOLATION_ALGOS))


def fig_jets(summary, datasets, out):
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
        for row, panel in enumerate(("efficiency", "median", "resolution")):
            _draw_series(axes[row][j], summary, ds, "jets", panel)
        for algo in METHODS:
            x, *_ = series(summary, ds, "jets", "efficiency", algo)
            if len(x):
                # The UPPER EDGE of the last populated bin, not its centre. The series is a step
                # spanning the whole bin now, so an axis stopping at the centre cuts the last bin
                # in half and the figure loses its highest-pT measurement without saying so.
                xmax = max(xmax, float(th.bin_edges_for(x, th.JET_PT_BINS)[1].max()))
        axes[0][j].set_ylim(0, 1.02)
        # No hardcoded ylim on the response row: an earlier version capped it at 1.3 and silently
        # clipped CLUE's entire line off the top. An axis limit that can hide a series is worse
        # than an untidy one.
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
    ap.add_argument("--events", type=int, default=120, help="events for the store-derived figures")
    ap.add_argument("--rebuild-anatomy", action="store_true")
    ap.add_argument("--rebuild-jets", action="store_true")
    ap.add_argument("--from-summary", action="store_true",
                    help="draw every column from results/<ds>/figure_summary.csv and read nothing "
                         "else -- no event store, no per-row tables. Use this to redraw the figures "
                         "on a machine where the analysis was not run.")
    ap.add_argument("--latex", action="store_true",
                    help="render text with a real LaTeX installation (requires one to be present)")
    args = ap.parse_args()

    import matplotlib.pyplot as _plt

    globals()["plt"] = _plt
    th.apply(latex=True if args.latex else None)
    OUT.mkdir(parents=True, exist_ok=True)

    active = settings()["dataset"]["active"]
    summaries = []
    for ds in args.datasets:
        res = Path("results") / ds
        if not res.is_dir():
            print(f"  ! {ds}: no results/{ds}/ -- skipping this column", flush=True)
            continue
        summary_file = res / SUMMARY_NAME

        # EACH SOURCE IS OPTIONAL AND THEY ARE GATHERED INDEPENDENTLY. The previous version bailed
        # out of the whole dataset the moment particles_*.parquet was missing, which silently threw
        # away a jets.parquet that IS tracked and had travelled through git perfectly well -- the
        # pu200 column of figure 6 vanished with the data sitting right there on disk. A dataset now
        # contributes whatever it has.
        tables, anat, profiles, jets = {}, None, None, None

        # --from-summary is a hard short-circuit, not a hint. It draws the column from the committed
        # figure_summary.csv and reads nothing else -- not the store, and not the per-row tables
        # either. Both halves matter: skipping only the store would let a machine holding partial
        # tables rebuild a summary missing the jets, anatomy and profile rows, and write that over
        # the complete committed one. This is the mode for drawing the figures somewhere the
        # analysis was not run.
        if args.from_summary:
            if summary_file.exists():
                print(f"  {ds}: drawing from the committed {summary_file.name}", flush=True)
                summaries.append(pd.read_csv(summary_file))
            else:
                print(f"  ! {ds}: no {SUMMARY_NAME} -- skipping this column", flush=True)
            continue

        parts = sorted(res.glob("particles_*.parquet"))
        if parts:
            tables[ds] = (
                pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True),
                pd.concat([pd.read_parquet(p) for p in sorted(res.glob("clusters_*.parquet"))],
                          ignore_index=True),
            )

        # THE STORE IS ONLY TOUCHED FOR THE ACTIVE DATASET. Jets, the shower anatomy and the shower
        # profiles are the three quantities that cannot be recovered from the per-row tables and
        # have to be recomputed from the event store, which lives on the compute cluster.
        # `load_profiles` in particular runs on every invocation for the active dataset, with no
        # cache to fall back on, so a machine without the store must use --from-summary above.
        jcache = res / "jets.parquet"
        if ds == active and (args.rebuild_jets or not jcache.exists()):
            print(f"  building jet cache for {ds} ({args.events} events) ...", flush=True)
            build_jets(ds, settings(), args.events).to_parquet(jcache)
        if jcache.exists():
            jets = pd.read_parquet(jcache)

        cache = res / "anatomy_particles.parquet"
        if ds == active:
            if args.rebuild_anatomy or not cache.exists():
                print(f"  building anatomy cache for {ds} ({args.events} events) ...", flush=True)
                build_anatomy(ds, settings(), args.events).to_parquet(cache)
            profiles = load_profiles(ds, settings(), args.events)
        if cache.exists():
            anat = pd.read_parquet(cache)

        # REBUILD FROM THE PER-ROW TABLES WHEN THIS MACHINE HAS THEM, otherwise fall back to the
        # committed summary. The rebuild wins deliberately: local tables are the fresher artefact,
        # and a stale committed summary silently plotting over a rescore is exactly the failure
        # this repository keeps designing out.
        built = build_summary(ds, tables, anat, profiles, jets)
        if not built.empty:
            built.to_csv(summary_file, index=False, float_format="%.6g")
            print(f"  {ds}: summary rebuilt -> {summary_file} ({len(built)} rows)", flush=True)
            summaries.append(built)
        elif summary_file.exists():
            print(f"  {ds}: no local tables; drawing from the committed {summary_file.name}", flush=True)
            summaries.append(pd.read_csv(summary_file))
        else:
            print(f"  ! {ds}: no tables and no {SUMMARY_NAME} -- skipping this column "
                  f"(dump its store and run scripts.score, or copy the summary across)", flush=True)

    if not summaries:
        raise SystemExit("no dataset had tables or a committed summary; nothing to draw")
    summary = pd.concat(summaries, ignore_index=True)

    def drawn(figure: str) -> list[str]:
        """The datasets that actually carry rows for one figure, in the requested order."""
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
