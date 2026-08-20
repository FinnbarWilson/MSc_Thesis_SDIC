"""CLUEstering's periodic metric needs the wrap flags too.

``choose_metric("periodic_euclidean")`` looks sufficient and is not: it changes how far apart
two points are, but CLUE first bins points into a tile grid to decide which pairs to compare at
all, and that grid wraps only when ``wrapped_coords`` says so. Without it a shower straddling
+/-pi is split in two exactly as if the metric had never been set, and the failure is silent.

Both halves are pinned, the negative case deliberately, so the call cannot be "simplified" back
to metric-only. The module skips without CLUEstering rather than failing to collect, or a
missing optional dependency would take the scorer tests down with it.
"""

import numpy as np
import pytest

clue = pytest.importorskip("CLUEstering", reason="CLUEstering is not installed; see environment.yml")

TWO_PI = 2.0 * np.pi


def n_clusters(eta, phi, weights, wrapped, d_c=0.05, rho_c=0.5):
    clusterer = clue.clusterer(d_c, rho_c, d_c)
    data = np.array([eta, phi, weights])
    if wrapped is None:
        clusterer.read_data(data)
    else:
        clusterer.read_data(data, wrapped_coords=wrapped)
    clusterer.choose_metric("periodic_euclidean", parameters=[0.0, TWO_PI])
    clusterer.run_clue(backend="cpu serial")
    ids = clusterer.output_df["cluster_ids"].to_numpy()
    return len(set(ids[ids >= 0].tolist()))


@pytest.fixture
def blob_across_pi():
    """One Gaussian blob centred on phi = pi, so half of it wraps to -pi."""
    rng = np.random.default_rng(0)
    eta = rng.normal(0.0, 0.02, 400)
    phi = np.mod(rng.normal(np.pi, 0.02, 400) + np.pi, TWO_PI) - np.pi
    return eta, phi, np.ones(400)


def test_metric_alone_does_not_wrap(blob_across_pi):
    eta, phi, w = blob_across_pi
    assert n_clusters(eta, phi, w, wrapped=None) == 2, (
        "if this ever returns 1, CLUEstering has changed and the wrapped_coords workaround "
        "in src/clue/pipeline.py can be revisited"
    )


def test_wrapped_coords_joins_the_blob(blob_across_pi):
    eta, phi, w = blob_across_pi
    assert n_clusters(eta, phi, w, wrapped=[0, 1]) == 1


def test_wrapping_does_not_merge_genuinely_separate_blobs():
    """The period comes from the metric, not the data range, so wrapping is safe to leave on."""
    rng = np.random.default_rng(1)
    eta = np.concatenate([rng.normal(0.0, 0.02, 200), rng.normal(0.0, 0.02, 200)])
    phi = np.concatenate([rng.normal(0.1, 0.02, 200), rng.normal(0.6, 0.02, 200)])
    assert n_clusters(eta, phi, np.ones(400), wrapped=[0, 1]) == 2
