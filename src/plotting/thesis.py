"""Styling and binning for the eight thesis figures, which `scripts.make_thesis_figures` draws.

These are sized for a 12pt A4 document with 1 inch margins, at 0.8 of a 6.5 inch textwidth, and
they use plain scienceplots so the figures look like the rest of the literature rather than like
this project. `src.plotting.style` is the separate, smaller styling for the dataset-composition
figures, which are not in the thesis.

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

COLOURS come from the scienceplots cycle, assigned by role rather than hand-picked: each method
takes a fixed position in the cycle, so its identity is stable across figures without importing a
second palette's taste.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

WIDE = (5.2, 2.7)
TALL = (5.2, 4.6)

LABELS: Mapping[str, str] = {
    "maskformer": "MaskFormer",
    "clue": "CLUE",
    "oracle_resolution": "Resolution ceiling",
}
MARKERS: Mapping[str, str] = {
    "maskformer": "o", "clue": "s",
    "oracle_resolution": "v",
}
#: Line style per method, which carries the identity once the markers are gone. The binned figures
#: are drawn as step outlines (see `draw_steps`), where a marker at a bin centre would sit in the
#: middle of a flat segment and say nothing the segment does not. Two solid steps of similar
#: lightness are hard to follow where they cross, so the second method is dashed instead.
LINESTYLES: Mapping[str, str] = {
    "maskformer": "-", "clue": "--",
    "oracle_resolution": ":",
}
REFERENCES = frozenset({"oracle_resolution"})
DATASETS: tuple[str, ...] = ("pu0", "pu200")
DATASET_LABELS: Mapping[str, str] = {"pu0": "pileup 0", "pu200": "pileup 200"}

#: Transverse-momentum bins, shared by every differential figure so the panels line up.
#:
#: pT RATHER THAN ENERGY, changed 2026-08-13, and the reason is not only that the comparison
#: literature plots pT. On this sample the two axes disagree about the shape: median E/pT is 1.75
#: and 10.9% of targets sit beyond |eta| = 2.44, so binning by ENERGY mixes "energetic" with
#: "forward" and produces a high-energy fall that is partly geometry. Binned by pT the same
#: efficiency curve dips and recovers (0.590 -> 0.682 in the top bin) instead of falling away.
#: Energy remains the metric WEIGHT -- eff_e and pur_e are unchanged; only the binning variable
#: moved.
PT_BINS_DIFFERENTIAL = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0])
E_BINS = PT_BINS_DIFFERENTIAL

#: Jet bins start at the 25 GeV analysis threshold -- below it there are no jets to bin -- and run
#: to where a ttbar event stops producing them.
JET_PT_BINS = np.array([25.0, 35.0, 50.0, 75.0, 110.0, 160.0, 250.0, 400.0])
JET_MIN_PT = 25.0

_COLOURS: dict[str, str] = {}


def apply(latex: bool | None = None) -> None:
    """Plain scienceplots, plus only what the page geometry requires.

    Args:
        latex: render text with a real LaTeX installation rather than matplotlib's mathtext.
            ``None`` (the default) reads the ``CALO_FIGURE_LATEX`` environment variable, so the
            choice can be made once for a whole session. It is **off** by default because neither
            compute cluster this work runs on has a LaTeX installation, and matplotlib raises at
            draw time rather than falling back when one is missing. Turn it on where the thesis is
            typeset, so the figures use the document's own fonts and maths.
    """
    global _COLOURS
    if latex is None:
        latex = os.environ.get("CALO_FIGURE_LATEX", "").lower() in {"1", "true", "yes"}
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science"] if latex else ["science", "no-latex"])
    except (ImportError, OSError):
        import warnings

        warnings.warn("scienceplots missing; falling back to matplotlib defaults", stacklevel=2)
    if latex:
        # scienceplots' own preamble covers siunitx and amsmath; this adds nothing to it and only
        # asserts the mode, so a missing LaTeX fails here with a clear message rather than midway
        # through the eighth figure.
        plt.rcParams.update({"text.usetex": True})
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
    # first two and the pair that reads best together.
    _COLOURS = {
        "maskformer": cycle[0],        # blue
        "clue": cycle[1],              # green
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


class SplitKey:
    """A legend key split down the middle, one method's style on each half.

    THE ENTRY THAT BELONGS TO BOTH METHODS IS THE PROBLEM THIS SOLVES. Two figures use a second
    visual channel on top of colour -- lightness for the three fates in `energy_budget`, weight for
    the two curves in `shower_profile` -- and that channel means the same thing for MaskFormer and
    for CLUE. Drawn in either method's colour the key would claim to belong to that method, so both
    figures previously drew those entries in a neutral dark grey: a swatch in a colour that appears
    nowhere in the panels, which the reader has to work out is standing in for "either of these".

    A key split down its middle says it directly. The left half is drawn in the first method's
    colour and style and the right half in the second's, so the swatch is made of exactly the two
    things it describes and the entry reads as "this, in both colours" without a detour through the
    caption. `algos` is in the same order as the method entries above it in the key, so left-then-
    right matches top-then-bottom.

    Args:
        algos: the methods to split across, left to right.
        patch: draw filled swatches (a stacked-bar key) rather than line segments.
        alpha: opacity, for the figure that carries lightness as its second channel.
        linewidth: line weight, for the figure that carries weight as its second channel.
    """

    def __init__(self, algos: Sequence[str], *, patch: bool = False, alpha: float = 1.0,
                 linewidth: float = 1.0):
        self.algos = tuple(algos)
        self.patch = patch
        self.alpha = alpha
        self.linewidth = linewidth


class _SplitKeyHandler(HandlerBase):
    """Draws a `SplitKey` as n abutting segments across the width of one legend handle."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize,
                       trans):
        artists = []
        n = len(orig_handle.algos)
        for i, algo in enumerate(orig_handle.algos):
            x0 = -xdescent + width * i / n
            if orig_handle.patch:
                art = Rectangle((x0, -ydescent), width / n, height, facecolor=colour(algo),
                                alpha=orig_handle.alpha, linewidth=0)
            else:
                y = -ydescent + height / 2
                art = Line2D([x0, x0 + width / n], [y, y], color=colour(algo),
                             linestyle=LINESTYLES.get(algo, "-"), linewidth=orig_handle.linewidth,
                             alpha=orig_handle.alpha)
            art.set_transform(trans)
            artists.append(art)
        return artists


#: Passed to every legend these figures draw, so a `SplitKey` handle renders anywhere.
HANDLER_MAP = {SplitKey: _SplitKeyHandler()}


def legend_below(fig, ncol: int = 3, y: float = 0.0, extra=None, handlelength: float | None = None) -> None:
    """One legend for the whole figure, under it, so no panel loses area to a key.

    Args:
        extra: ``(handle, label)`` pairs appended after the panel's own entries, for keys that
            describe a second channel rather than a series -- see `SplitKey`. Passing them here
            rather than as empty proxy artists on the panel keeps handles that no axes can produce
            (a split swatch is not something `ax.plot` returns) out of the drawing code.
        handlelength: widen the swatches, in font-size units. A split key needs the room: at the
            default 1.5 each half is about six points, which is too little for a dash pattern to
            show more than one dash.
    """
    handles, labels = fig.axes[0].get_legend_handles_labels()
    for handle, label in extra or ():
        handles.append(handle)
        labels.append(label)
    if handles:
        kw = {} if handlelength is None else {"handlelength": handlelength}
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                   frameon=False, handler_map=HANDLER_MAP, **kw)


def band(ax, algo, x, lo, hi):
    """Shade an uncertainty interval. A band, not caps.

    With seven bins and two methods, capped error bars produce fourteen sets of whiskers that
    collide with the markers and with each other. A filled band reads as one continuous statement
    about where the curve could lie, and stays legible when two of them overlap.
    """
    if len(x) == 0:
        return
    ax.fill_between(x, lo, hi, color=colour(algo), alpha=0.18, linewidth=0, zorder=1)


def _step_path(lo, hi, y):
    """A piecewise-constant path over bins, broken by NaN wherever a bin is missing.

    Every quantity in these figures is measured in bins, and a marker at the geometric centre of
    each bin joined to the next by a straight line says nothing about how wide those bins are. The
    top edge of a histogram does, and because the measurements move smoothly across the range it
    still reads as a curve.

    The NaN break is the part that has to be right. `binned_proportion`, `binned_bootstrap` and
    `binned_ratio` all drop a bin holding fewer than `min_count` entries, so the bins reaching this
    function are not guaranteed contiguous. A single path drawn through them would put a horizontal
    segment across a bin that was never measured, which is the one error this style could introduce
    that a reader could not see. Contiguous runs are therefore emitted separately.

    Args:
        lo, hi: the lower and upper edge of each bin, in plotting coordinates.
        y: the value in each bin.

    Returns:
        ``(x, y)`` arrays for ``ax.plot`` or ``ax.fill_between``.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y = np.asarray(y, dtype=float)
    xs: list[float] = []
    ys: list[float] = []
    i, n = 0, len(lo)
    while i < n:
        j = i
        while j + 1 < n and np.isclose(hi[j], lo[j + 1], rtol=1e-9, atol=1e-12):
            j += 1
        for k in range(i, j + 1):
            xs += [lo[k], hi[k]]
            ys += [y[k], y[k]]
        xs.append(np.nan)
        ys.append(np.nan)
        i = j + 1
    return np.asarray(xs), np.asarray(ys)


def draw_steps(ax, algo, lo, hi, y, dashed: bool | None = None, label: str | None = None,
               alpha: float = 1.0, linewidth: float = 1.0, linestyle: str | None = None):
    """One series as the top edge of a histogram. The step replaces `draw`'s markers and line."""
    ref = algo in REFERENCES if dashed is None else dashed
    style = linestyle if linestyle is not None else ("--" if ref else LINESTYLES.get(algo, "-"))
    x, yy = _step_path(lo, hi, y)
    ax.plot(x, yy, color=colour(algo), linestyle=style, linewidth=linewidth,
            alpha=0.75 * alpha if ref else alpha,
            label=LABELS.get(algo, algo) if label is None else label)


def band_steps(ax, algo, lo, hi, ylo, yhi, alpha: float = 0.18):
    """Shade an uncertainty interval over the same bins, so the band steps with the series."""
    if len(lo) == 0:
        return
    x, low = _step_path(lo, hi, ylo)
    _, high = _step_path(lo, hi, yhi)
    ax.fill_between(x, low, high, color=colour(algo), alpha=alpha, linewidth=0, zorder=1)


def bin_edges_for(x, edges):
    """(lo, hi) edges of the bin each plotted point came from.

    The summary carries one x per bin rather than the edges, and the two figure families use
    different conventions for it: the differential figures place a point at the geometric mean of
    its bin and `anatomy.profile` at the arithmetic midpoint. Both fall strictly inside their own
    bin, so a single digitize recovers the edges for either without the summary having to grow a
    column, and without a committed summary from another machine needing to be regenerated.
    """
    x = np.asarray(x, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    return edges[idx], edges[idx + 1]


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


def finish(fig, path, ncol: int = 3, extra=None, handlelength: float | None = None):
    legend_below(fig, ncol=ncol, extra=extra, handlelength=handlelength)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return path
