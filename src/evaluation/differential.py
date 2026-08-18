"""Binning metrics against particle kinematics, with honest error bars.

This is the point of the whole exercise. An aggregate efficiency is an average dominated by
whatever the sample happens to contain -- here, marginal particles sitting on the pT cut --
and it is the same number for two algorithms that fail in completely different places. The
statement worth making is differential: how efficiency depends on particle energy, and on
how crowded the particle's neighbourhood is.

Two kinds of quantity appear, and they need different error bars:

*   A **fraction** -- "what share of particles reach 50% recovery" -- is a binomial
    proportion. Its uncertainty is *not* sqrt(k)/n, which misbehaves badly near 0 and 1 and
    can put an error bar outside [0, 1]. Clopper-Pearson and Wilson intervals are provided;
    both stay inside the range and remain sensible when a bin is fully efficient.
*   A **pooled ratio** -- "total energy recovered over total energy deposited in this bin" --
    is a ratio of sums, not a proportion, and no binomial interval applies to it. Use the
    bootstrap.

Getting that distinction wrong is the classic way to publish a plot whose error bars are
decorative.

There is a third trap underneath both, and it is the one that actually bites here. Every
binomial interval assumes independent trials, and ~620 target particles from the same event
are nothing of the sort: they share cells, they share the same local occupancy, and they are
reconstructed by one pass of the same algorithm over one detector state. Treating them as
620 independent draws understates the uncertainty by roughly the square root of the number
of particles per event. :func:`binned_fraction` therefore accepts a `cluster` argument -- pass
the event id and the interval comes from resampling *events*, which is the unit that actually
repeats. The binomial forms are kept for bins where a handful of events makes the bootstrap
unstable, and because the difference is itself worth showing.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

#: 1 sigma. A 68.27% interval is what a HEP reader assumes unless told otherwise.
ALPHA_1SIGMA = 0.3173

@dataclass(frozen=True)
class BinnedResult:
    """A binned quantity with asymmetric errors, ready to plot."""

    edges: np.ndarray
    centres: np.ndarray
    value: np.ndarray
    low: np.ndarray
    high: np.ndarray
    count: np.ndarray

    @property
    def yerr(self) -> np.ndarray:
        """``(2, n_bins)`` array in the form matplotlib's errorbar expects."""
        return np.vstack([self.value - self.low, self.high - self.value])


def clopper_pearson(k: np.ndarray, n: np.ndarray, alpha: float = ALPHA_1SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """Exact binomial confidence interval.

    Conservative by construction, and correct at the edges: a bin where every particle
    succeeded gets an interval reaching exactly 1 with a sensible lower bound, rather than
    the zero-width bar a naive sqrt(k)/n would draw.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    low = np.where(k > 0, stats.beta.ppf(alpha / 2, k, np.maximum(n - k + 1, 1e-12)), 0.0)
    high = np.where(k < n, stats.beta.ppf(1 - alpha / 2, k + 1, np.maximum(n - k, 1e-12)), 1.0)
    return np.nan_to_num(low, nan=0.0), np.nan_to_num(high, nan=1.0)


def wilson(k: np.ndarray, n: np.ndarray, alpha: float = ALPHA_1SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval: tighter than Clopper-Pearson, still well behaved at 0 and 1."""
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    low = np.where(n > 0, np.clip(centre - half, 0.0, 1.0), 0.0)
    high = np.where(n > 0, np.clip(centre + half, 0.0, 1.0), 0.0)
    return low, high


def binned_fraction(
    values: np.ndarray,
    passed: np.ndarray,
    bins: np.ndarray,
    interval: str = "clopper_pearson",
    alpha: float = ALPHA_1SIGMA,
    cluster: np.ndarray | None = None,
    n_boot: int = 400,
    seed: int = 42,
) -> BinnedResult:
    """Fraction of entries satisfying `passed`, per bin of `values`.

    Args:
        values: the binning variable, one entry per object.
        passed: boolean, whether each object counts as a success.
        bins: bin edges.
        interval: ``"clopper_pearson"`` or ``"wilson"``. Ignored when `cluster` is given.
        alpha: 1 - confidence level.
        cluster: optional grouping id per object, normally the event id. When given, the
            interval comes from resampling whole groups with replacement instead of assuming
            independent objects. Particles in one event are correlated, so the binomial
            intervals are too narrow; this is the honest version and the one to publish.
        n_boot: bootstrap replicates, when `cluster` is given.
        seed: bootstrap seed, so a figure is reproducible.
    """
    values = np.asarray(values, dtype=np.float64)
    passed = np.asarray(passed, dtype=bool)
    finite = np.isfinite(values)
    index = np.digitize(values[finite], bins) - 1
    inside = (index >= 0) & (index < len(bins) - 1)
    index, hit = index[inside], passed[finite][inside]

    n_bins = len(bins) - 1
    n = np.bincount(index, minlength=n_bins).astype(np.float64)
    k = np.bincount(index[hit], minlength=n_bins).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.divide(k, n, out=np.zeros(n_bins), where=n > 0)

    if cluster is None:
        low, high = (clopper_pearson if interval == "clopper_pearson" else wilson)(k, n, alpha)
        return BinnedResult(edges=bins, centres=_centres(bins), value=value, low=low, high=high, count=n)

    groups = np.asarray(cluster)[finite][inside]
    low, high = _cluster_bootstrap_fraction(index, hit, groups, n_bins, n_boot, seed, alpha)
    # A bin whose entries all come from one or two events has no bootstrap to speak of; fall
    # back to the binomial there rather than drawing a zero-width bar that claims certainty.
    binom_low, binom_high = clopper_pearson(k, n, alpha)
    thin = np.array([np.unique(groups[index == b]).size < 3 for b in range(n_bins)])
    low = np.where(thin, binom_low, low)
    high = np.where(thin, binom_high, high)

    return BinnedResult(edges=bins, centres=_centres(bins), value=value, low=low, high=high, count=n)


def _cluster_bootstrap_fraction(
    index: np.ndarray,
    hit: np.ndarray,
    groups: np.ndarray,
    n_bins: int,
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval on a per-bin fraction, resampling whole groups.

    Every bin is resampled under the *same* draw of events, which is what a repeated
    experiment would look like and keeps neighbouring bins correlated the way they really
    are, rather than letting each bin wobble independently.
    """
    unique, coded = np.unique(groups, return_inverse=True)
    n_groups = int(unique.size)
    rng = np.random.default_rng(seed)

    # Per (event, bin): how many objects and how many passed. Resampling an event then means
    # adding its whole row, which is a single matrix product rather than a re-bin per draw.
    flat = coded * n_bins + index
    totals = np.bincount(flat, minlength=n_groups * n_bins).reshape(n_groups, n_bins).astype(np.float64)
    passes = np.bincount(flat[hit], minlength=n_groups * n_bins).reshape(n_groups, n_bins).astype(np.float64)

    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    lo = np.zeros(n_bins)
    hi = np.zeros(n_bins)
    counts = np.empty((n_boot, n_groups), dtype=np.float64)
    for b in range(n_boot):
        counts[b] = np.bincount(draws[b], minlength=n_groups)
    boot_total = counts @ totals
    boot_pass = counts @ passes
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(boot_pass, boot_total, out=np.zeros_like(boot_total), where=boot_total > 0)
    for b in range(n_bins):
        column = ratios[:, b]
        column = column[np.isfinite(column)]
        if column.size:
            lo[b], hi[b] = np.percentile(column, [15.865, 84.135])
    return lo, hi


def _centres(bins: np.ndarray) -> np.ndarray:
    """Bin centres, geometric where the binning is clearly logarithmic."""
    bins = np.asarray(bins, dtype=np.float64)
    ratios = bins[1:] / np.where(bins[:-1] == 0, np.nan, bins[:-1])
    if np.all(bins > 0) and np.nanstd(ratios) < 1e-6:
        return np.sqrt(bins[:-1] * bins[1:])
    return (bins[:-1] + bins[1:]) / 2.0
