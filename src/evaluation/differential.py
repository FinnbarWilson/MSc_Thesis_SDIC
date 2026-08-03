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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

#: 1 sigma. A 68.27% interval is what a HEP reader assumes unless told otherwise.
ALPHA_1SIGMA = 0.3173

ENERGY_BINS = np.geomspace(0.1, 100.0, 16)
PT_BINS = np.geomspace(0.5, 100.0, 14)
ABS_ETA_BINS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
NHITS_BINS = np.array([1, 3, 5, 8, 12, 20, 35, 60, 100, 200, 500])
DR_BINS = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 5.0])
DENSITY_BINS = np.array([0, 1, 2, 3, 5, 8, 12, 20, 50])


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


def binned_ratio(
    values: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    bins: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
) -> BinnedResult:
    """Pooled ``sum(numerator) / sum(denominator)`` per bin, with a bootstrap band.

    Used for energy ratios such as the response. This is deliberately not given a binomial
    interval: it is a ratio of sums over objects with wildly different weights, so its
    uncertainty is driven by the spread of the heavy objects, not by a count.
    """
    values = np.asarray(values, dtype=np.float64)
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)

    # All three have to be finite, not just the binning variable: a NaN in either sum
    # propagates through np.bincount and silently voids the whole bin.
    finite = np.isfinite(values) & np.isfinite(numerator) & np.isfinite(denominator)
    index = np.digitize(values[finite], bins) - 1
    inside = (index >= 0) & (index < len(bins) - 1)
    index = index[inside]
    num = numerator[finite][inside]
    den = denominator[finite][inside]

    n_bins = len(bins) - 1
    count = np.bincount(index, minlength=n_bins).astype(np.float64)
    total_num = np.bincount(index, weights=num, minlength=n_bins)
    total_den = np.bincount(index, weights=den, minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.divide(total_num, total_den, out=np.zeros(n_bins), where=total_den > 0)

    rng = np.random.default_rng(seed)
    low = np.zeros(n_bins)
    high = np.zeros(n_bins)
    for b in range(n_bins):
        pick = index == b
        if pick.sum() < 2:
            low[b] = high[b] = value[b]
            continue
        n_b, d_b = num[pick], den[pick]
        draws = rng.integers(0, n_b.size, size=(n_boot, n_b.size))
        sums_d = d_b[draws].sum(axis=1)
        ratios = np.divide(n_b[draws].sum(axis=1), sums_d, out=np.zeros(n_boot), where=sums_d > 0)
        low[b], high[b] = np.percentile(ratios, [15.865, 84.135])

    return BinnedResult(edges=bins, centres=_centres(bins), value=value, low=low, high=high, count=count)


def binned_median(values: np.ndarray, quantity: np.ndarray, bins: np.ndarray) -> BinnedResult:
    """Median of `quantity` per bin, with the interquartile range as the band.

    Quote a width this way rather than as an RMS: the tails here are strongly non-Gaussian
    and an RMS would be dominated by a handful of catastrophic mis-clusterings rather than
    describing the core of the distribution.
    """
    values = np.asarray(values, dtype=np.float64)
    quantity = np.asarray(quantity, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(quantity)
    index = np.digitize(values[finite], bins) - 1
    inside = (index >= 0) & (index < len(bins) - 1)
    index, q = index[inside], quantity[finite][inside]

    n_bins = len(bins) - 1
    value = np.zeros(n_bins)
    low = np.zeros(n_bins)
    high = np.zeros(n_bins)
    count = np.bincount(index, minlength=n_bins).astype(np.float64)
    for b in range(n_bins):
        pick = q[index == b]
        if pick.size:
            value[b], low[b], high[b] = np.percentile(pick, [50, 25, 75])

    return BinnedResult(edges=bins, centres=_centres(bins), value=value, low=low, high=high, count=count)


def _centres(bins: np.ndarray) -> np.ndarray:
    """Bin centres, geometric where the binning is clearly logarithmic."""
    bins = np.asarray(bins, dtype=np.float64)
    ratios = bins[1:] / np.where(bins[:-1] == 0, np.nan, bins[:-1])
    if np.all(bins > 0) and np.nanstd(ratios) < 1e-6:
        return np.sqrt(bins[:-1] * bins[1:])
    return (bins[:-1] + bins[1:]) / 2.0


def reference_table(particles, clusters, cuts: Sequence[tuple[str, str]], working_point: float = 0.5) -> "object":
    """Efficiency and purity at a few named selections, for the summary table.

    Args:
        particles: pooled particle table.
        clusters: pooled cluster table.
        cuts: ``(label, pandas query)`` pairs applied to the particle table.
        working_point: recovery a particle must reach to count as efficient.
    """
    import pandas as pd

    rows = []
    for label, query in cuts:
        selected = particles.query(query) if query else particles
        for algo, group in selected.groupby("algo", observed=True):
            same_algo = clusters[clusters["algo"] == algo]

            # Purity needs care here, and it is easy to get wrong in a way that flatters
            # both algorithms. The cuts are on particle properties, so the only clusters
            # they can select are the ones matched to a selected particle -- which silently
            # drops every fake. Reporting that alone is exactly the survivorship bias this
            # analysis is supposed to avoid: an algorithm that emits a thousand junk
            # clusters would look no worse than one that emits none.
            #
            # So both are reported. `pur_matched` is conditional on the cluster having been
            # matched to a selected particle, and `fake_rate` -- the share of ALL that
            # algorithm's clusters matched to nothing -- is what keeps it honest. The
            # brief's unconditional definition is `pur_all`.
            in_scope = set(zip(group["sample_id"], group["particle_row"], strict=True))
            matched_here = same_algo[
                [pair in in_scope for pair in zip(same_algo["sample_id"], same_algo["particle_row"], strict=True)]
            ]

            k = int((group["eff_e"] >= working_point).sum())
            n = int(len(group))
            low, high = _event_clustered_interval(group, "eff_e", working_point)
            rows.append({
                "selection": label,
                "algo": algo,
                "n_particles": n,
                "efficiency": k / n if n else 0.0,
                "eff_low": low,
                "eff_high": high,
                # The working-point fraction is a step function on a continuous variable, so
                # two methods whose median recovery differs by 0.02 either side of the
                # threshold look utterly different while being nearly identical. The mean and
                # median go next to it so a reader can see which is happening.
                "eff_mean": float(group["eff_e"].mean()) if n else 0.0,
                "eff_median": float(group["eff_e"].median()) if n else 0.0,
                # `pur_all_clusters` is the brief's unconditional definition and deliberately
                # does NOT respond to the selection -- the cuts are on particle properties and
                # a cluster matched to nothing has no particle to cut on. It repeats down each
                # algorithm's rows; that is correct, not a bug.
                "pur_all_clusters": float((same_algo["pur_e"] >= working_point).mean()) if len(same_algo) else 0.0,
                "pur_matched": float((matched_here["pur_e"] >= working_point).mean()) if len(matched_here) else 0.0,
                "pur_median": float(same_algo["pur_e"].median()) if len(same_algo) else 0.0,
                "fake_rate": float((~same_algo["matched"]).mean()) if len(same_algo) else 0.0,
                "match_rate": float(group["matched"].mean()) if n else 0.0,
                # Both weightings of the split rate, because they disagree materially and the
                # energy one is primary. See src/evaluation/matching.py:fragmentation.
                "split_rate": float(group["is_split"].mean()) if n else 0.0,
                "split_rate_e": float(group["is_split_e"].mean()) if n and "is_split_e" in group else float("nan"),
                "merge_rate": float(same_algo["is_merge"].mean()) if len(same_algo) else 0.0,
                "merge_rate_e": float(same_algo["is_merge_e"].mean()) if len(same_algo) and "is_merge_e" in same_algo else float("nan"),
                "frag_frac": float(group["frag_frac"].mean()) if n else 0.0,
            })
    return pd.DataFrame(rows)


def _event_clustered_interval(
    group, column: str, working_point: float, n_boot: int = 400, seed: int = 42
) -> tuple[float, float]:
    """1 sigma interval on a pass fraction, resampling events rather than particles.

    Falls back to Clopper-Pearson when there are too few events to resample.
    """
    n = int(len(group))
    if n == 0:
        return 0.0, 0.0

    passed = (group[column].to_numpy() >= working_point)
    events = group["sample_id"].to_numpy()
    unique, coded = np.unique(events, return_inverse=True)
    if unique.size < 3:
        low, high = clopper_pearson(np.array([int(passed.sum())]), np.array([n]))
        return float(low[0]), float(high[0])

    totals = np.bincount(coded, minlength=unique.size).astype(np.float64)
    passes = np.bincount(coded[passed], minlength=unique.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique.size, size=(n_boot, unique.size))
    counts = np.stack([np.bincount(d, minlength=unique.size) for d in draws]).astype(np.float64)
    boot = np.divide(counts @ passes, counts @ totals, out=np.zeros(n_boot), where=(counts @ totals) > 0)
    return float(np.percentile(boot, 15.865)), float(np.percentile(boot, 84.135))
