"""Styling and binning helpers for the eight thesis figures.

Figures are 5.2 inches wide, which is 0.8 textwidth in a 12pt A4 document with 1 inch margins.
Include them at that width and the type comes out the size set here rather than rescaled.

Columns are datasets, rows are quantities, and every energy axis is the truth particle's.
Colours are assigned by role, so a method keeps its identity across figures.
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
#: Line style per method, which carries the identity in the step outlines `draw_steps` draws,
#: where a marker at a bin centre would say nothing the flat segment does not.
LINESTYLES: Mapping[str, str] = {
    "maskformer": "-", "clue": "--",
    "oracle_resolution": ":",
}
REFERENCES = frozenset({"oracle_resolution"})
DATASETS: tuple[str, ...] = ("pu0", "pu200")
DATASET_LABELS: Mapping[str, str] = {"pu0": "pileup 0", "pu200": "pileup 200"}

#: Transverse-momentum bins, shared by every differential figure so the panels line up. pT
#: rather than energy, because binning by energy mixes "energetic" with "forward" on this
#: sample. Energy remains the metric weight; only the binning variable is pT.
PT_BINS_DIFFERENTIAL = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0])
E_BINS = PT_BINS_DIFFERENTIAL

#: Jet bins start at the 25 GeV analysis threshold and run to where ttbar stops producing jets.
JET_PT_BINS = np.array([25.0, 35.0, 50.0, 75.0, 110.0, 160.0, 250.0, 400.0])
JET_MIN_PT = 25.0

_COLOURS: dict[str, str] = {}


def apply(latex: bool | None = None) -> None:
    """Plain scienceplots, plus only what the page geometry requires.

    Args:
        latex: render text with a real LaTeX installation rather than matplotlib's mathtext.
            ``None`` reads the ``CALO_FIGURE_LATEX`` environment variable. Off by default,
            because matplotlib raises at draw time rather than falling back when TeX is absent.
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
        plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Sized for 0.8 textwidth in a 12pt document, so these land near body-text size.
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # By role, not by position, so removing a method does not repaint the others.
    _COLOURS = {
        "maskformer": cycle[0],        # blue
        "clue": cycle[1],              # green
        "oracle_resolution": "#9E9E9E",
    }


def colour(algo: str) -> str:
    return _COLOURS.get(algo, "#666666")



class SplitKey:
    """A legend key split down the middle, one method's style on each half.

    For entries describing a second visual channel: lightness for the fates in `energy_budget`,
    weight for the two curves in `shower_profile`. Both mean the same thing for either method, so
    a key in one method's colour would claim to belong to it.

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

    # Signature fixed by matplotlib's HandlerBase; not every argument is used.
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,  # noqa: ARG002
                       fontsize, trans):  # noqa: ARG002
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
            describe a second channel rather than a series. See `SplitKey`.
        handlelength: widen the swatches, in font-size units. A split key needs the room, each
            half being about six points at the default.
    """
    handles, labels = fig.axes[0].get_legend_handles_labels()
    for handle, label in extra or ():
        handles.append(handle)
        labels.append(label)
    if handles:
        kw = {} if handlelength is None else {"handlelength": handlelength}
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                   frameon=False, handler_map=HANDLER_MAP, **kw)



def _step_path(lo, hi, y):
    """A piecewise-constant path over bins, broken by NaN wherever a bin is missing.

    The NaN break matters: the binning helpers drop a bin holding fewer than `min_count`
    entries, so the bins reaching here are not contiguous, and a single path through them would
    draw a horizontal segment across a bin that was never measured.

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

    The summary carries one x per bin rather than the edges, and the two figure families place it
    differently, at the geometric mean or the arithmetic midpoint. Both fall strictly inside their
    own bin, so one digitize recovers the edges for either.
    """
    x = np.asarray(x, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    return edges[idx], edges[idx + 1]


def binned_proportion(x, passed, bins, min_count: int = 20):
    """(centres, p, lo, hi) for a binomial proportion, with an exact Clopper-Pearson interval.

    Efficiency and purity are counts of successes out of trials, so the interval is exact rather
    than bootstrapped. Bins holding fewer than `min_count` entries are dropped.
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
    """(centres, stat, lo, hi) with a 68% interval from resampling events.

    Medians and resolutions are not binomial proportions, so the interval is bootstrapped. The
    resampling unit is the event, never the particle: particles in one event share cells and
    occupancy, and treating them as independent trials understates the spread.

    Args:
        statistic: ``"median"``, ``"mean"``, or ``"resolution"`` (IQR/1.349 over the median).

    Raises:
        ValueError: on an unknown `statistic`.
    """
    x = np.asarray(x)
    values = np.asarray(values)
    events = np.asarray(events)
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

    A ratio of sums, not a mean of per-object ratios. The per-particle distribution is bimodal
    for a method that usually fragments a large shower and occasionally captures most of one, and
    a mean over it describes neither behaviour. The interval is a bootstrap over events.
    """
    x = np.asarray(x)
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    events = np.asarray(events)
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
        # Per-event partial sums, so a bootstrap replicate is a sum over drawn events rather
        # than a re-gather of the underlying rows.
        starts = np.searchsorted(eb, uniq, side="left")
        stops = np.searchsorted(eb, uniq, side="right")
        present = np.flatnonzero(stops > starts)
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


def figsize_for(nrows: int) -> tuple[float, float]:
    """Width fixed at 0.8 textwidth; height set by the row count alone.

    Adding a dataset column narrows the panels rather than making the figure taller. Rows get
    shorter as they get more numerous, so three of them still fit on a page.
    """
    # 1.75 in of plot per row, plus 0.9 in for the shared x label and the legend beneath.
    per_row = 1.75 if nrows < 3 else 1.45
    return (5.2, round(nrows * per_row + 0.9, 2))


def grid(nrows: int, ncols: int, datasets: Sequence[str], sharey: str | bool = "row", sharex: str | bool = "col"):
    """Panel grid with datasets as columns, titled once along the top.

    Pass ``sharex=False`` where the rows are different coordinates rather than the same one at
    two scales: sharing the axis there collapses one row against the origin.
    """
    size = figsize_for(nrows)
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
