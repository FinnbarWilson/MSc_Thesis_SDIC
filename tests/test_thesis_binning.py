"""The binning behind every plotted point and every error bar in the thesis."""

import numpy as np
import pytest

from src.plotting import thesis as th

BINS = np.array([1.0, 10.0, 100.0])


# --- which bins survive, and where their centres sit ------------------------


def test_a_sparse_bin_is_dropped_rather_than_plotted():
    """min_count is what stops a two-particle bin appearing as a point with a huge bar."""
    x = np.concatenate([np.full(30, 3.0), np.full(5, 30.0)])
    passed = np.ones(x.size, dtype=bool)

    centres, _, _, _ = th.binned_proportion(x, passed, BINS, min_count=20)

    assert centres.size == 1
    assert centres[0] == pytest.approx(np.sqrt(1.0 * 10.0))


def test_centres_are_geometric_because_the_axis_is_logarithmic():
    """A linear centre on a log axis does not sit in the middle of its own bin."""
    x = np.full(40, 3.0)
    centres, _, _, _ = th.binned_proportion(x, np.ones(40, dtype=bool), BINS)
    assert centres[0] == pytest.approx(3.1622776, rel=1e-6)
    assert centres[0] != pytest.approx((1.0 + 10.0) / 2)


# --- binned_proportion: the efficiency and purity bars ----------------------


def test_a_fully_efficient_bin_gets_an_interval_that_reaches_one():
    """The Clopper-Pearson property. A normal approximation gives zero width here."""
    x = np.full(50, 3.0)
    _, p, lo, hi = th.binned_proportion(x, np.ones(50, dtype=bool), BINS)

    assert p[0] == pytest.approx(1.0)
    assert hi[0] == pytest.approx(1.0)
    assert lo[0] < 1.0          # and the bar has width, unlike sqrt(k)/n
    assert lo[0] > 0.9          # 50/50 successes is still a tight bound


def test_the_interval_brackets_the_proportion_and_narrows_with_more_data():
    small_x, big_x = np.full(40, 3.0), np.full(4000, 3.0)
    small_pass = np.tile([True, False], 20)
    big_pass = np.tile([True, False], 2000)

    _, ps, los, his = th.binned_proportion(small_x, small_pass, BINS)
    _, pb, lob, hib = th.binned_proportion(big_x, big_pass, BINS)

    assert ps[0] == pytest.approx(0.5) and pb[0] == pytest.approx(0.5)
    assert los[0] < 0.5 < his[0]
    assert (hib[0] - lob[0]) < (his[0] - los[0]) / 5


# --- binned_bootstrap: the response and resolution bars ---------------------


def test_resampling_events_is_wider_than_resampling_particles():
    """The correlation this exists for: 40 particles from 4 events are not 40 trials.

    Every particle in an event is given the same value, which is the extreme of the correlation
    that really exists. Treating particles as the unit would collapse the interval; resampling
    events cannot, because there are only four of them.
    """
    events = np.repeat(np.arange(4), 40)
    values = np.repeat([1.0, 1.0, 1.0, 2.0], 40)
    x = np.full(events.size, 3.0)

    _, stat, lo, hi = th.binned_bootstrap(x, values, events, BINS, "median", n_boot=200)
    _, _, lo_p, hi_p = th.binned_bootstrap(x, values, np.arange(events.size), BINS,
                                           "median", n_boot=200)

    assert stat[0] == pytest.approx(1.0)
    assert (hi[0] - lo[0]) > (hi_p[0] - lo_p[0])


def test_a_bin_from_a_single_event_is_dropped_not_bootstrapped():
    """One event cannot be resampled; a point with no honest bar should not be drawn."""
    x = np.full(40, 3.0)
    centres, *_ = th.binned_bootstrap(x, np.ones(40), np.zeros(40, dtype=int), BINS)
    assert centres.size == 0


def test_resolution_is_the_normalised_iqr_not_the_standard_deviation():
    """IQR/1.349 over the median, so one runaway response cannot inflate the resolution."""
    rng = np.random.default_rng(0)
    events = np.repeat(np.arange(20), 20)
    values = rng.normal(1.0, 0.1, size=events.size)
    x = np.full(events.size, 3.0)

    _, clean, _, _ = th.binned_bootstrap(x, values, events, BINS, "resolution")
    spiked = values.copy()
    spiked[0] = 500.0
    _, tainted, _, _ = th.binned_bootstrap(x, spiked, events, BINS, "resolution")

    assert clean[0] == pytest.approx(0.1, abs=0.03)      # sigma/mu for this sample
    assert tainted[0] == pytest.approx(clean[0], abs=0.02)   # the outlier barely moves it


def test_an_unknown_statistic_is_rejected_rather_than_silently_defaulting():
    x = np.full(40, 3.0)
    with pytest.raises(ValueError, match="mode"):
        th.binned_bootstrap(x, np.ones(40), np.repeat(np.arange(4), 10), BINS, "mode")


# --- binned_ratio: the cells-recovered and jet-response panels --------------


def test_a_ratio_of_sums_is_not_a_mean_of_ratios():
    """The skew this exists for: one particle recovering 100 of 100 cells and many recovering 0.

    The mean per-object ratio says 1/21 = 0.048; the pooled ratio says 100/(100 + 20) = 0.833.
    The pooled one is the quantity the figure claims to show.
    """
    events = np.arange(21)
    numerator = np.array([100.0] + [0.0] * 20)
    denominator = np.array([100.0] + [1.0] * 20)
    x = np.full(21, 3.0)

    _, ratio, _, _ = th.binned_ratio(x, numerator, denominator, events, BINS, min_count=20)

    assert ratio[0] == pytest.approx(100.0 / 120.0)
    assert ratio[0] != pytest.approx(float(np.mean(numerator / denominator)))


def test_the_ratio_interval_brackets_the_value():
    rng = np.random.default_rng(1)
    events = np.repeat(np.arange(30), 5)
    denominator = rng.uniform(5.0, 15.0, size=events.size)
    numerator = denominator * rng.uniform(0.4, 0.6, size=events.size)
    x = np.full(events.size, 3.0)

    _, ratio, lo, hi = th.binned_ratio(x, numerator, denominator, events, BINS)

    assert lo[0] <= ratio[0] <= hi[0]
    assert 0.4 < ratio[0] < 0.6


# --- the bin edges the figures are drawn on --------------------------------


def test_the_published_bin_edges_are_monotonic_and_positive():
    """Geometric centres are undefined at zero, and digitize needs increasing edges."""
    for edges in (th.E_BINS, th.JET_PT_BINS):
        assert np.all(np.diff(edges) > 0)
        assert edges[0] > 0
    assert th.JET_PT_BINS[0] == pytest.approx(th.JET_MIN_PT)
