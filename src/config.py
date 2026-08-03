"""Access to the experiment definition in ``config/experiment.yaml``.

Every module reads its settings through this one loader, so the shared decisions that make
the CLUE and MaskFormer comparison fair live in exactly one place. The file is read once and
cached; call :func:`reload` if it changes during a session.

Note what this module does **not** do. The cuts that define which cells and which particles
exist are not applied from here -- they were applied once, by the hepattn dump, and travel
inside the event store. :func:`store_expectations` collects what this repository *believes*
those cuts to be, and :class:`~src.io.event_store.EventStore` refuses to open a store that
disagrees. A config that has drifted therefore fails loudly at load rather than quietly
producing numbers for a different experiment.

Typical use::

    from src.config import settings, store_path, store_expectations

    cfg = settings()
    store = EventStore(store_path(), expect=store_expectations())
"""

from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "experiment.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

_CACHE: dict | None = None


def reload() -> dict:
    """Re-read the experiment file from disk and return the fresh settings."""
    global _CACHE
    with CONFIG_PATH.open() as handle:
        _CACHE = yaml.safe_load(handle)
    _validate(_CACHE)
    return deepcopy(_CACHE)


def settings() -> dict:
    """Return the experiment settings as a nested dictionary.

    A copy is returned each call, so a caller mutating the result cannot alter what any
    other module sees. Do not call this inside a per-event or per-trial loop -- read it once
    and pass the values down. Use :func:`frozen` where a read-only view will do.
    """
    if _CACHE is None:
        return reload()
    return deepcopy(_CACHE)


def frozen() -> MappingProxyType:
    """A read-only view of the settings, with no copying cost."""
    if _CACHE is None:
        reload()
    assert _CACHE is not None
    return MappingProxyType(_CACHE)


def store_path(kind: str = "store") -> Path:
    """Path to the event store for the active dataset.

    Args:
        kind: ``"store"`` for the evaluation window, ``"tune_store"`` for the smaller
            window CLUE's parameter search runs on.
    """
    cfg = settings()["dataset"]
    active = cfg["active"]
    entry = cfg[active]
    if kind not in entry or not entry[kind]:
        msg = (
            f"dataset.{active}.{kind} is not set in {CONFIG_PATH}. Produce a store with "
            f"hepattn.experiments.colliderml.eval.dump and point this at it."
        )
        raise ValueError(msg)
    return Path(entry[kind])


def window(kind: str = "eval") -> tuple[int, int]:
    """Return ``(start_event, n_events)`` for the ``eval`` or ``tune`` window."""
    if kind not in ("eval", "tune"):
        msg = f"Unknown window {kind!r}; expected 'eval' or 'tune'."
        raise ValueError(msg)
    windows = settings()["windows"]
    return windows[f"{kind}_start"], windows[f"{kind}_events"]


def store_expectations() -> dict:
    """The store metadata this config asserts, for :class:`EventStore` to check.

    Keys are the leaf names of the reader's ``CONTRACT_KEYS``.
    """
    cfg = settings()
    return {
        "length": "m",
        "energy": "GeV",
        **cfg["hit_selection"],
        **cfg["particle_selection"],
        "subsystem_order": list(cfg["detectors"]),
    }


def _validate(cfg: dict) -> None:
    """Check the invariants that keep the two pipelines comparable.

    Raises:
        ValueError: if the active dataset is unknown, or the tuning and evaluation windows
            overlap -- which would mean CLUE had been tuned on the events it is reported on
            while the MaskFormer had not, an advantage that has nothing to do with either
            algorithm.
    """
    dataset = cfg["dataset"]
    if dataset["active"] not in ("pu0", "pu200"):
        msg = f"dataset.active must be pu0 or pu200, got {dataset['active']!r}"
        raise ValueError(msg)

    windows = cfg["windows"]
    eval_start, eval_n = windows["eval_start"], windows["eval_events"]
    tune_start, tune_n = windows["tune_start"], windows["tune_events"]
    if tune_start < eval_start + eval_n and tune_start + tune_n > eval_start:
        msg = (
            f"CLUE's tuning window [{tune_start}, {tune_start + tune_n}) overlaps the "
            f"evaluation window [{eval_start}, {eval_start + eval_n}). Tuning on the reported "
            f"events would flatter CLUE relative to the MaskFormer, which was not."
        )
        raise ValueError(msg)

    expected = {"ecb", "ece", "hcb", "hce"}
    if set(cfg["detectors"]) != expected:
        msg = f"detectors must be exactly {sorted(expected)}, got {cfg['detectors']}"
        raise ValueError(msg)
