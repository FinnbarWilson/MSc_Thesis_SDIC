"""Calorimeter layer geometry, recovered from cell positions.

The evaluation needs a layer index per cell, because the CLUE baseline clusters within a
layer before linking layers into a shower. Nothing in the ColliderML files provides one, so
it has to be derived from the positions.

For the endcaps a layer is a plane of constant \\|z\\| and the derivation is trivial. The barrels
are the interesting case: they are 16-fold polygonal, built from flat staves, so a single
layer spans a *range* of radii and clustering on r alone would smear ~48 layers into several
hundred bands. Projecting onto the stave normal collapses each flat plate back to one
constant depth:

    period    = 2*pi/16
    phi_local = mod(phi + period/2, period) - period/2     # angle within the stave
    depth     = r * cos(phi_local)                         # perpendicular distance to the axis

Pooling 60 events and splitting wherever consecutive depths differ by more than 1 mm gives:

    ecb  48 layers, pitch 5.05 mm      ece  48 layers, pitch 5.05 mm
    hcb  36 layers, pitch 51.00 mm     hce  36 layers, pitch 51.00 mm

ECAL and HCAL each come out with the same layer count and the same pitch in barrel and
endcap, and the pitch is uniform to 0.01 mm. That uniformity is the check that the
projection is right: a wrong stave count gives ragged, unphysical spacings (N=8 -> 32
layers, N=12 -> 15, N=20 -> 7 for ecb). The 1 mm tolerance sits on a plateau -- 2 mm gives
the same answer, 0.5 mm starts over-splitting (54 and 39) -- so it is not a tuned number.

The edges are calibrated ONCE and frozen into the event-store metadata. They must not be
re-derived per event: a sparse subsystem leaves most of its layers unlit in any single
event, which undercounts them (hcb gives 26 per event against its true 36) and would make
the layer index mean different things in different events.
"""

from collections.abc import Mapping, Sequence

import numpy as np

SUBSYSTEMS: tuple[str, ...] = ("ecb", "ece", "hcb", "hce")
BARREL_SUBSYSTEMS = frozenset({"ecb", "hcb"})

# Number of flat staves around phi in the barrels. Measured, not assumed: see the module
# docstring for the scan over candidate symmetries.
STAVE_SYMMETRY = 16

# Gap above which two depths belong to different layers. The smallest real pitch is the
# ECAL's 5.05 mm, and the intra-layer spread is well under 0.2 mm, so 1 mm separates them
# comfortably from either side.
LAYER_GAP_TOLERANCE_M = 1.0e-3

# What calibration should find. Asserted, so a dataset change fails loudly instead of
# silently renumbering every layer in the store.
EXPECTED_NUM_LAYERS: Mapping[str, int] = {"ecb": 48, "ece": 48, "hcb": 36, "hce": 36}


def layer_depth(subsystem: str, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Depth coordinate whose discrete values are the layers of `subsystem`.

    Args:
        subsystem: one of `SUBSYSTEMS`.
        x, y, z: cell positions in metres.

    Returns:
        Depth in metres: perpendicular distance to the beam axis within a stave for the
        barrels, \\|z\\| for the endcaps.
    """
    if subsystem not in EXPECTED_NUM_LAYERS:
        msg = f"Unknown subsystem {subsystem!r}; expected one of {SUBSYSTEMS}"
        raise ValueError(msg)

    if subsystem not in BARREL_SUBSYSTEMS:
        return np.abs(z)

    period = 2.0 * np.pi / STAVE_SYMMETRY
    phi_local = np.mod(np.arctan2(y, x) + period / 2.0, period) - period / 2.0
    return np.hypot(x, y) * np.cos(phi_local)


def calibrate_layer_centres(depth: np.ndarray, tolerance: float = LAYER_GAP_TOLERANCE_M) -> np.ndarray:
    """Group a pooled depth distribution into layers and return each layer's mean depth.

    Args:
        depth: depths pooled over many events, in metres.
        tolerance: minimum gap between layers.

    Returns:
        Sorted layer centres, one per layer.
    """
    ordered = np.sort(np.asarray(depth, dtype=np.float64))
    if ordered.size == 0:
        return np.empty(0, dtype=np.float64)

    cuts = np.flatnonzero(np.diff(ordered) > tolerance)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts + 1, [ordered.size]))
    return np.array([ordered[s:e].mean() for s, e in zip(starts, ends, strict=True)], dtype=np.float64)


def layer_boundaries(centres: Sequence[float] | np.ndarray) -> np.ndarray:
    """Midpoints between consecutive layer centres, for `np.searchsorted`.

    Derived rather than stored so that the writer and the reader cannot disagree: both
    call this on the same `layer_centres_m` from the metadata.
    """
    centres = np.asarray(centres, dtype=np.float64)
    return (centres[:-1] + centres[1:]) / 2.0


def assign_layers(subsystem: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, centres: Sequence[float] | np.ndarray) -> np.ndarray:
    """Layer index per cell, against a frozen set of layer centres.

    Args:
        subsystem: one of `SUBSYSTEMS`.
        x, y, z: cell positions in metres.
        centres: calibrated layer centres for this subsystem.

    Returns:
        uint8 layer index in `[0, len(centres))`.

    Raises:
        ValueError: if any cell sits further than half a pitch from its assigned layer,
            which means the frozen geometry no longer describes the data.
    """
    depth = layer_depth(subsystem, x, y, z)
    centres = np.asarray(centres, dtype=np.float64)
    index = np.searchsorted(layer_boundaries(centres), depth)

    residual = np.abs(depth - centres[index])
    pitch = float(np.median(np.diff(centres))) if centres.size > 1 else np.inf
    if residual.size and residual.max() > pitch / 2.0:
        worst = int(np.argmax(residual))
        msg = (
            f"{subsystem}: cell at depth {depth[worst]:.6f} m is {residual[worst] * 1e3:.2f} mm from "
            f"layer {index[worst]} (pitch {pitch * 1e3:.2f} mm). The frozen layer geometry does not "
            f"describe this data -- re-run the layer calibration."
        )
        raise ValueError(msg)

    return index.astype(np.uint8)


def check_layer_counts(centres_by_subsystem: Mapping[str, np.ndarray]) -> None:
    """Fail loudly if calibration did not recover the expected layer counts."""
    found = {name: len(centres) for name, centres in centres_by_subsystem.items()}
    if found != dict(EXPECTED_NUM_LAYERS):
        msg = f"Layer calibration gave {found}, expected {dict(EXPECTED_NUM_LAYERS)}. Inspect the depth distribution before writing a store."
        raise ValueError(msg)
