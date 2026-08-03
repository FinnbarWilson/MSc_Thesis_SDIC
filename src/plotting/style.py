"""Shared figure styling.

One colour per algorithm, fixed everywhere. A reader comparing six plots should never have
to re-read a legend to work out which line is which.
"""

from collections.abc import Mapping

import matplotlib as mpl

#: Colourblind-safe and distinguishable in greyscale, which a printed thesis may well be.
ALGO_COLOURS: Mapping[str, str] = {
    "maskformer": "#0173b2",
    "clue": "#de8f05",
    "truth": "#555555",
    # The reference clusterings are deliberately grey and unsaturated: they are the axis the
    # two real methods are read against, not a third competitor.
    "oracle_geometric": "#029e73",
    "oracle_resolution": "#7a7a7a",
}

ALGO_LABELS: Mapping[str, str] = {
    "maskformer": "MaskFormer",
    "clue": "CLUE",
    "truth": "Truth",
    "oracle_geometric": "Geometric ceiling",
    "oracle_resolution": "Resolution ceiling",
}

ALGO_MARKERS: Mapping[str, str] = {
    "maskformer": "o",
    "clue": "s",
    "truth": "^",
    "oracle_geometric": "D",
    "oracle_resolution": "v",
}

#: Reference clusterings are drawn dashed so they read as context even in greyscale.
ALGO_LINESTYLES: Mapping[str, str] = {
    "oracle_geometric": "--",
    "oracle_resolution": ":",
}

#: Drawn behind the real methods, and excluded from "which algorithm won" comparisons.
REFERENCE_ALGOS = frozenset({"oracle_geometric", "oracle_resolution", "truth"})

RC_PARAMS = {
    "figure.figsize": (5.2, 3.9),
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "errorbar.capsize": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply() -> None:
    """Install the thesis style into matplotlib's global state."""
    mpl.rcParams.update(RC_PARAMS)


def colour(algo: str) -> str:
    return ALGO_COLOURS.get(algo, "#333333")


def label(algo: str) -> str:
    return ALGO_LABELS.get(algo, algo)


def marker(algo: str) -> str:
    return ALGO_MARKERS.get(algo, "o")


def linestyle(algo: str) -> str:
    return ALGO_LINESTYLES.get(algo, "-")


def is_reference(algo: str) -> bool:
    return algo in REFERENCE_ALGOS
