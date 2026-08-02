"""Access to the experiment definition in ``config/experiment.yaml``.

Every module reads its settings through this one loader, so the shared decisions
that make the CLUE and MaskFormer comparison fair live in exactly one place. The
file is read once and cached; call :func:`reload` if it changes during a session.

Typical use::

    from src.config import settings, dataset_paths

    cfg = settings()
    detectors = cfg["detectors"]
    calo_path, particles_path = dataset_paths()
"""

from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "experiment.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

_CACHE = None


def reload():
    """Re-read the experiment file from disk and return the fresh settings."""
    global _CACHE
    with open(CONFIG_PATH) as handle:
        _CACHE = yaml.safe_load(handle)
    _validate(_CACHE)
    return deepcopy(_CACHE)


def settings():
    """Return the experiment settings as a nested dictionary.

    A copy is returned each call, so a caller mutating the result cannot alter
    what any other module sees.
    """
    if _CACHE is None:
        return reload()
    return deepcopy(_CACHE)


def dataset_paths():
    """Return ``(calo_hits_path, particles_path)`` for the active dataset.

    Raises:
        ValueError: if the active dataset has no paths configured, which happens
            when switching to a sample that has not been downloaded yet.
    """
    cfg = settings()["dataset"]
    active = cfg["active"]
    entry = cfg[active]
    if not entry["calo_hits"] or not entry["particles"]:
        raise ValueError(
            f"Dataset '{active}' has no paths set in {CONFIG_PATH}. "
            "Fill them in, or switch dataset.active to a sample you have."
        )
    return Path(entry["calo_hits"]), Path(entry["particles"])


def split_bounds(name):
    """Return ``(start_event, n_events)`` for the split ``name``.

    Args:
        name: one of ``"train"``, ``"val"`` or ``"test"``.
    """
    splits = settings()["splits"]
    if name not in ("train", "val", "test"):
        raise ValueError(f"Unknown split '{name}'; expected train, val or test.")
    return splits[f"{name}_start"], splits[f"{name}_events"]


def _validate(cfg):
    """Check the invariants that keep the two pipelines comparable.

    Raises:
        ValueError: if the splits overlap, or the active dataset is unknown.
    """
    dataset = cfg["dataset"]
    if dataset["active"] not in ("pu0", "pu200"):
        raise ValueError(f"dataset.active must be pu0 or pu200, got {dataset['active']!r}")

    splits = cfg["splits"]
    windows = [
        ("train", splits["train_start"], splits["train_events"]),
        ("val", splits["val_start"], splits["val_events"]),
        ("test", splits["test_start"], splits["test_events"]),
    ]
    windows.sort(key=lambda item: item[1])
    for (name, start, count), (next_name, next_start, _) in zip(windows, windows[1:]):
        if start + count > next_start:
            raise ValueError(
                f"Split '{name}' ends at event {start + count} but '{next_name}' "
                f"starts at {next_start}; the windows must not overlap."
            )

    if splits["clue_tune_events"] > splits["train_events"]:
        raise ValueError("splits.clue_tune_events cannot exceed splits.train_events.")
