"""The thesis figure set: five figures that carry the argument, and nothing else.

SEPARATE FROM `figures.py` ON PURPOSE. That module is the working set -- one figure per
deliverable, built while the analysis was still open, and it carries styling decisions
(a custom palette, a grid, a legend placement) that were right for reading on screen. These are
for a 12pt A4 document with 1 inch margins, at 0.8 of a 6.5 inch textwidth, and they use plain
scienceplots so the figures look like the rest of the literature rather than like this project.

LAYOUT CONTRACT

Every figure is 5.2 inches wide, which is 0.8 textwidth at those page settings, so
`\\includegraphics[width=0.8\\textwidth]{...}` reproduces the sizes chosen here rather than
rescaling the type. Two heights only:

    WIDE    5.2 x 2.7   a single row of two panels
    TALL    5.2 x 4.6   two rows of two panels

Fonts are set once, in points that survive that scaling: at 0.8 textwidth a 9pt axis label in the
figure lands at roughly 9pt on the page, so the figure type matches the body type instead of
shrinking to unreadable.

PANEL GRAMMAR, so a reader learns it once

    columns are DATASETS   pileup 0 on the left, pileup 200 on the right
    rows are QUANTITIES    the thing being measured
    x is always the TRUE particle energy where an energy axis appears

That last one is a correction to the working figures, which in one place plotted cluster energy
and particle energy on the same axis. Every energy axis here is the truth, and every cluster
quantity is taken from the cluster MATCHED to that particle, so a reader never has to ask which
energy is meant.

COLOURS come from the scienceplots cycle, assigned by position in `ALGOS`. Nothing is
hand-picked: the methods take the first colours of the cycle in a fixed order, so the identity is
stable across figures without importing a second palette's taste.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

WIDE = (5.2, 2.7)
TALL = (5.2, 4.6)

#: Plot order, which is also colour order in the scienceplots cycle. References last so they take
#: the greys the cycle ends with, and read as context rather than as competitors.
ALGOS: tuple[str, ...] = ("maskformer", "maskformer_chained", "clue", "oracle_geometric", "oracle_resolution")
LABELS: Mapping[str, str] = {
    "maskformer": "MaskFormer",
    "maskformer_chained": "MaskFormer + chaining",
    "clue": "CLUE",
    "oracle_geometric": "Geometric ceiling",
    "oracle_resolution": "Resolution ceiling",
}
MARKERS: Mapping[str, str] = {
    "maskformer": "o", "maskformer_chained": "P", "clue": "s",
    "oracle_geometric": "D", "oracle_resolution": "v",
}
REFERENCES = frozenset({"oracle_geometric", "oracle_resolution"})
DATASETS: tuple[str, ...] = ("pu0", "pu200")
DATASET_LABELS: Mapping[str, str] = {"pu0": "pileup 0", "pu200": "pileup 200"}

#: Energy bins, shared by every differential figure so the panels line up with each other.
E_BINS = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0])

#: Jet bins start at the 25 GeV analysis threshold -- below it there are no jets to bin -- and run
#: to where a ttbar event stops producing them.
JET_PT_BINS = np.array([25.0, 35.0, 50.0, 75.0, 110.0, 160.0, 250.0, 400.0])
JET_MIN_PT = 25.0

_COLOURS: dict[str, str] = {}


def apply() -> None:
    """Plain scienceplots, plus only what the page geometry requires."""
    global _COLOURS
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except (ImportError, OSError):
        import warnings

        warnings.warn("scienceplots missing; falling back to matplotlib defaults", stacklevel=2)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Sized for 0.8 textwidth in a 12pt document: the figure is not rescaled much, so these
        # land close to body-text size on the page.
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Assigned by ROLE, not by position in a list, so removing a method does not repaint the
    # others. The two carrying the comparison take scienceplots' blue and green -- the cycle's
    # first two and the pair that reads best together; chaining takes orange only when drawn.
    _COLOURS = {
        "maskformer": cycle[0],        # blue
        "clue": cycle[1],              # green
        "maskformer_chained": cycle[2],  # orange
        "oracle_geometric": "#5A5A5A",
        "oracle_resolution": "#9E9E9E",
    }


def colour(algo: str) -> str:
    return _COLOURS.get(algo, "#666666")


def draw(ax, algo, x, y, yerr=None, dashed: bool | None = None):
    """One series, styled consistently. References dashed so they survive greyscale."""
    ref = algo in REFERENCES if dashed is None else dashed
    ax.errorbar(x, y, yerr=yerr, color=colour(algo), marker=MARKERS.get(algo, "o"),
                linestyle="--" if ref else "-", linewidth=1.0, markersize=3.0,
                alpha=0.75 if ref else 1.0, label=LABELS.get(algo, algo), capsize=0)


def legend_below(fig, ncol: int = 3, y: float = 0.0) -> None:
    """One legend for the whole figure, under it, so no panel loses area to a key."""
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


def band(ax, algo, x, lo, hi):
    """Shade an uncertainty interval. A band, not caps.

    With seven bins and two methods, capped error bars produce fourteen sets of whiskers that
    collide with the markers and with each other. A filled band reads as one continuous statement
    about where the curve could lie, and stays legible when two of them overlap.
    """
    if len(x) == 0:
        return
    ax.fill_between(x, lo, hi, color=colour(algo), alpha=0.18, linewidth=0, zorder=1)


def binned_proportion(x, passed, bins, min_count: int = 20):
    """(centres, p, lo, hi) for a binomial proportion, with an exact Clopper-Pearson interval.

    Efficiency and purity are counts of successes out of trials, so the interval is exact rather
    than bootstrapped. Clopper-Pearson is conservative and behaves correctly at the edges, where a
    bin with every particle succeeding gets an interval reaching 1 instead of a zero-width bar.
    """
    from src.evaluation.differential import clopper_pearson

    idx = np.digitize(x, bins) - 1
    cs, ps, los, his = [], [], [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n < min_count:
            continue
        k = float(np.asarray(passed)[sel].sum())
        lo, hi = clopper_pearson(np.array([k]), np.array([float(n)]))
        cs.append(np.sqrt(bins[b] * bins[b + 1]))
        ps.append(k / n)
        los.append(float(lo[0]))
        his.append(float(hi[0]))
    return np.array(cs), np.array(ps), np.array(los), np.array(his)


def binned_bootstrap(x, values, events, bins, statistic="median", n_boot: int = 200,
                     min_count: int = 20, seed: int = 0):
    """(centres, stat, lo, hi) with a 68% interval from resampling EVENTS.

    Medians, means of pooled quantities and resolutions are not binomial proportions, so the
    interval comes from a bootstrap. The resampling unit is the EVENT, never the particle:
    particles in one event share cells, occupancy and a shower environment, so treating them as
    independent trials would understate the spread -- often badly, since a busy event contributes
    hundreds of correlated entries at once.
    """
    x = np.asarray(x); values = np.asarray(values); events = np.asarray(events)
    idx = np.digitize(x, bins) - 1
    rng = np.random.default_rng(seed)
    uniq = np.unique(events)
    cs, st, los, his = [], [], [], []

    def _stat(v):
        if v.size == 0:
            return np.nan
        if statistic == "median":
            return float(np.median(v))
        if statistic == "mean":
            return float(np.mean(v))
        if statistic == "resolution":  # IQR/1.349, normalised by the median
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            return float((q3 - q1) / 1.349 / max(q2, 1e-9))
        raise ValueError(statistic)

    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() < min_count:
            continue
        vb, eb = values[sel], events[sel]
        # Group once, then a replicate is a gather over event blocks rather than a re-filter.
        order = np.argsort(eb, kind="stable")
        vb, eb = vb[order], eb[order]
        starts = np.searchsorted(eb, uniq, side="left")
        stops = np.searchsorted(eb, uniq, side="right")
        present = np.flatnonzero(stops > starts)
        if present.size < 2:
            continue
        draws = []
        for _ in range(n_boot):
            pick = rng.choice(present, size=present.size, replace=True)
            v = np.concatenate([vb[starts[i]:stops[i]] for i in pick])
            draws.append(_stat(v))
        draws = np.asarray(draws)
        cs.append(np.sqrt(bins[b] * bins[b + 1]))
        st.append(_stat(vb))
        los.append(float(np.nanpercentile(draws, 16)))
        his.append(float(np.nanpercentile(draws, 84)))
    return np.array(cs), np.array(st), np.array(los), np.array(his)


def binned_ratio(x, numerator, denominator, events, bins, n_boot: int = 200,
                 min_count: int = 20, seed: int = 0):
    """(centres, ratio, lo, hi) for sum(numerator)/sum(denominator) in each bin.

    A RATIO OF SUMS, not a mean of per-object ratios, and the distinction matters here rather
    than being pedantry. The per-particle counts this is built from are heavily skewed: at
    50-200 GeV CLUE's matched cluster recovers a mean of 11.1 cells but a MEDIAN of 1, because
    it usually shatters a large shower and occasionally captures most of one. A mean over that
    distribution describes neither behaviour, and reversed the sign of the measured trend --
    rising on the mean, falling on the median.

    Summing first is immune to that: it asks what fraction of all the truth cells in this energy
    bin ended up correctly assigned, which is a well-posed question whatever the per-particle
    distribution looks like. The interval is a bootstrap over EVENTS, as everywhere else.
    """
    x = np.asarray(x); num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float); events = np.asarray(events)
    idx = np.digitize(x, bins) - 1
    rng = np.random.default_rng(seed)
    uniq = np.unique(events)
    cs, rs, los, his = [], [], [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() < min_count or den[sel].sum() <= 0:
            continue
        nb, db, eb = num[sel], den[sel], events[sel]
        order = np.argsort(eb, kind="stable")
        nb, db, eb = nb[order], db[order], eb[order]
        # Per-event partial sums: a bootstrap replicate is then a sum over drawn events, with no
        # need to re-gather the underlying rows.
        starts = np.searchsorted(eb, uniq, side="left")
        stops = np.searchsorted(eb, uniq, side="right")
        present = np.flatnonzero(stops > starts)
        cn = np.add.reduceat(np.concatenate([nb, [0.0]]), starts[present]) if present.size else np.zeros(0)
        cd = np.add.reduceat(np.concatenate([db, [0.0]]), starts[present]) if present.size else np.zeros(0)
        # reduceat over ragged blocks: recompute exactly rather than trusting the boundary case
        cn = np.array([nb[starts[i]:stops[i]].sum() for i in present])
        cd = np.array([db[starts[i]:stops[i]].sum() for i in present])
        if present.size < 2:
            continue
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, present.size, present.size)
            d = cd[pick].sum()
            draws.append(cn[pick].sum() / d if d > 0 else np.nan)
        draws = np.asarray(draws)
        cs.append(np.sqrt(bins[b] * bins[b + 1]))
        rs.append(float(nb.sum() / db.sum()))
        los.append(float(np.nanpercentile(draws, 16)))
        his.append(float(np.nanpercentile(draws, 84)))
    return np.array(cs), np.array(rs), np.array(los), np.array(his)


def binned(x: np.ndarray, y: np.ndarray, bins: np.ndarray, statistic: str = "mean", min_count: int = 20):
    """(centre, value) per bin, dropping bins too sparse to plot honestly."""
    idx = np.digitize(x, bins) - 1
    cs, vs = [], []
    for b in range(len(bins) - 1):
        sel = (idx == b) & np.isfinite(y)
        if sel.sum() < min_count:
            continue
        cs.append(np.sqrt(bins[b] * bins[b + 1]))
        v = y[sel]
        vs.append(np.median(v) if statistic == "median" else float(np.mean(v)))
    return np.array(cs), np.array(vs)


def figsize_for(nrows: int, ncols: int) -> tuple[float, float]:
    """Width is fixed at 0.8 textwidth; height follows the grid so panels stay near 4:3.

    A single-dataset column is not a temporary curiosity -- it is what these figures look like
    until pu0 is scored -- and a 5.2 x 4.6 figure holding one narrow column of two panels is
    unreadably tall on the page. Height therefore depends on BOTH dimensions.
    """
    # Height is set by the ROW COUNT alone, not by the panel width. The width is fixed at 0.8
    # textwidth, so adding a dataset column narrows the panels rather than making the figure
    # taller -- an earlier version scaled height with panel width and produced a 5.2 x 6.0 figure
    # for a single column, which is most of an A4 page for two panels.
    # 1.75 in of plot per row, plus 0.9 in for the shared x label and the legend beneath.
    # Three rows would be 6.15 in at 1.75 per row, which is a full page. Rows get shorter as
    # they get more numerous; the panels stay legible because the width never changes.
    per_row = 1.75 if nrows < 3 else 1.45
    return (5.2, round(nrows * per_row + 0.9, 2))


def grid(nrows: int, ncols: int, datasets: Sequence[str], sharey: str | bool = "row", sharex: str | bool = "col"):
    """Panel grid with datasets as columns, titled once along the top.

    `sharex="col"` is right when every row measures the same thing against the same x, which is
    the usual case here. It is WRONG when the rows are different coordinates: the shower-profile
    figure puts dR (0 to 0.35) above depth (0 to 47), and sharing the axis collapsed the dR row
    into a vertical line at the origin. Pass sharex=False there.
    """
    size = figsize_for(nrows, ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, sharex=sharex, sharey=sharey, squeeze=False)
    for j, ds in enumerate(datasets):
        axes[0][j].set_title(DATASET_LABELS.get(ds, ds))
    return fig, axes


def finish(fig, path, ncol: int = 3):
    legend_below(fig, ncol=ncol)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return path
