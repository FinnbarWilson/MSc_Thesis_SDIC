"""Binomial and bootstrap intervals for binned quantities.

:func:`clopper_pearson` is the interval the thesis figures use, through
:mod:`src.plotting.thesis`. The rest of the module supports :func:`binned_fraction`, which
exists to demonstrate the difference the resampling unit makes: ~620 target particles from one
event are not independent trials, so a binomial interval on them is too narrow, and passing
`cluster` resamples whole events instead. That comparison is exercised by
``tests/test_ceilings_and_weighting.py``.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

#: 1 sigma: a 68.27% interval, which is what a HEP reader assumes unless told otherwise.
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
    """Exact binomial confidence interval, from `k` successes in `n` trials.

    Conservative, and correct at the edges: a fully efficient bin gets an interval reaching 1
    with a sensible lower bound rather than the zero-width bar sqrt(k)/n would draw.
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
            interval comes from resampling whole groups rather than assuming independent
            objects.
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
    low, high = _cluster_bootstrap_fraction(index, hit, groups, n_bins, n_boot, seed)
    # A bin drawn from fewer than three events has no bootstrap to speak of; fall back to the
    # binomial rather than drawing a zero-width bar.
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
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval on a per-bin fraction, resampling whole groups.

    Every bin is resampled under the same draw of events, so neighbouring bins stay correlated
    the way a repeated experiment would make them.
    """
    unique, coded = np.unique(groups, return_inverse=True)
    n_groups = int(unique.size)
    rng = np.random.default_rng(seed)

    # Per (event, bin) counts, so a replicate is a matrix product rather than a re-bin.
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
