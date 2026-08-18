"""Styling for the dataset-composition figures.

One colour per particle class, fixed everywhere, so a reader comparing panels never has to
re-read a legend to work out which series is which.

**A legend belongs to a figure, not to a panel.** Several classes drawn inside each of three
panels meant three copies of the same legend, each covering a third of its own axes.
:func:`legend_below` puts one legend under the whole figure instead, which is both smaller and
less ambiguous.

The thesis figures have their own styling in :mod:`src.plotting.thesis`; this module covers only
the figures that describe the DATASET rather than a method.
"""

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Particle classes, for the figures that describe the DATASET rather than a method.
#:
#: Same science palette, and "All" is the same grey the reference clusterings use, for the
#: same reason: it is the total the classes are read against, not one more class competing
#: with them. The three that carry almost all of the calorimeter energy -- charged hadrons,
#: photons and electrons -- take the three loudest hues, and the rare classes take the rest.
PARTICLE_CLASS_COLOURS: Mapping[str, str] = {
    "all": "#474747",
    "charged_hadron": "#0C5DA5",
    "neutral_hadron": "#00B945",
    "electron": "#FF2C00",
    "photon": "#845B97",
    "muon": "#FF9500",
    "tau": "#9e9e9e",
    "neutrino": "#9e9e9e",
    "other": "#9e9e9e",
}

PARTICLE_CLASS_LABELS: Mapping[str, str] = {
    "all": "All",
    "charged_hadron": "Charged hadrons",
    "neutral_hadron": "Neutral hadrons",
    "electron": "Electrons",
    "photon": "Photons",
    "muon": "Muons",
    "tau": "Taus",
    "neutrino": "Neutrinos",
    "other": "Other",
}


def particle_class_colour(name: str) -> str:
    return PARTICLE_CLASS_COLOURS.get(name, "#9e9e9e")


def particle_class_label(name: str) -> str:
    return PARTICLE_CLASS_LABELS.get(name, name.replace("_", " ").capitalize())


#: Applied on top of scienceplots' `science` style. Everything here either overrides
#: something that style gets wrong for this thesis, or states a value the figures depend on.
RC_PARAMS = {
    "figure.figsize": (5.2, 3.9),
    "figure.dpi": 140,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # scienceplots sets 8pt for a two-column paper. A thesis page is wider and these figures
    # are read at full width, so the type can be bigger without crowding.
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    # A hairline grid on y only. Solid, never dashed: a dashed grid reads as a threshold or a
    # projection when it is neither, and these figures have real dashed lines that mean
    # something (the reference clusterings).
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.5,
    "grid.linewidth": 0.4,
    "grid.color": "#d8d8d8",
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.columnspacing": 1.4,
    "legend.handletextpad": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.6,
    "lines.markeredgewidth": 0.0,
    "errorbar.capsize": 0,
    # scienceplots draws ticks inside on all four sides, which is the convention in this
    # field and worth keeping; the top and right spines go with it.
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.6,
}


def apply() -> None:
    """Install the thesis style into matplotlib's global state.

    Layered on scienceplots' `science` style, with `no-latex` because the figures have to
    build on machines without a TeX installation -- an assessor running `make_thesis_figures` should
    not need one. If scienceplots is missing the figures still build, just on matplotlib's
    defaults, so a missing optional dependency degrades the look rather than breaking the
    acceptance criterion that every figure regenerates from the tables alone.
    """
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except (ImportError, OSError):
        import warnings

        warnings.warn(
            "scienceplots not available; figures will use matplotlib defaults. "
            "conda install -c conda-forge scienceplots",
            stacklevel=2,
        )
    mpl.rcParams.update(RC_PARAMS)


def legend_below(fig, handles=None, labels=None, ncol: int | None = None, y: float = 0.02) -> None:
    """Put a single legend under the whole figure.

    Panels then use their full area for data. Entries are collected from the first axes
    unless given explicitly, so every panel must draw its series in the same order -- which
    they do, because they all iterate the same dict.
    """
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(handles),
        frameon=False,
    )


def clear_panel_legends(axes: Sequence) -> None:
    """Remove per-panel legends left behind by the per-panel plotting helpers."""
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def panel_labels(axes, labels: Sequence[str] | None = None) -> None:
    """Label the panels of a multi-panel figure (a), (b), (c).

    The convention a thesis caption is written against: the caption says what each panel
    shows, so the panel itself only has to be identifiable. Descriptive per-panel titles
    duplicate that caption and, on a two-panel figure, tend to be longer than the panel is
    wide. This replaces `annotate_panel`, which drew in-panel notes like "higher is better"
    that either repeated the axis label or stated the obvious.
    """
    tags = labels or [f"({chr(ord('a') + i)})" for i in range(len(axes))]
    for ax, tag in zip(axes, tags, strict=False):
        ax.set_title(tag, loc="left", fontsize=9, pad=4)
