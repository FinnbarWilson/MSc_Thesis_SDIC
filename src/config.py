"""Loader for the experiment definition in ``config/experiment.yaml``.

Every module reads its settings through here, so the decisions shared by the CLUE and
MaskFormer pipelines live in one place. The file is read once and cached.

The cuts defining which cells and particles exist are not applied from here: they were applied
by the hepattn dump and travel inside the event store. :func:`store_expectations` collects what
this config asserts them to be, and :class:`~src.io.event_store.EventStore` refuses to open a
store that disagrees.
"""

import os
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "experiment.yaml"

#: A root, not an output directory: a run writes to ``results/<active dataset>/``.
RESULTS_ROOT = REPO_ROOT / "results"

DATASETS = ("pu0", "pu200")

_CACHE: dict | None = None
_RAW: dict | None = None


def reload() -> dict:
    """Re-read the experiment file from disk and return the fresh, merged settings."""
    global _CACHE, _RAW
    with CONFIG_PATH.open() as handle:
        raw = yaml.safe_load(handle)
    _validate_raw(raw)
    _RAW = raw
    _CACHE = _resolve(raw)
    _validate(_CACHE)
    return deepcopy(_CACHE)


def settings() -> dict:
    """Merged settings for the active dataset, as a nested dictionary.

    A copy is returned each call, so a caller mutating the result cannot affect any other
    module. Read it once outside a per-event or per-trial loop.
    """
    if _CACHE is None:
        return reload()
    return deepcopy(_CACHE)


def settings_for(dataset: str) -> dict:
    """Settings resolved against a named dataset, whatever ``dataset.active`` says.

    For scripts that describe both pileup conditions rather than scoring one; the pipelines
    should use :func:`settings`. Nothing is cached, so this cannot change what other callers
    see.

    Raises:
        ValueError: if `dataset` is unknown or has no block in the config.
    """
    if dataset not in DATASETS:
        msg = f"dataset must be one of {list(DATASETS)}, got {dataset!r}"
        raise ValueError(msg)
    if _RAW is None:
        reload()
    assert _RAW is not None
    raw = deepcopy(_RAW)
    raw["dataset"]["active"] = dataset
    if dataset not in raw["dataset"]:
        msg = f"there is no dataset.{dataset} block in {CONFIG_PATH}"
        raise ValueError(msg)
    return _resolve(raw)


def active_dataset() -> str:
    """The name of the dataset every path and override resolves against."""
    if _CACHE is None:
        reload()
    assert _CACHE is not None
    return _CACHE["dataset"]["active"]


def store_path(kind: str = "store", dataset: str | None = None) -> Path:
    """Path to an event store.

    ``CALO_STORE_ROOT`` relocates the directory without editing the config, so one config stays
    valid on two machines. Only the directory moves: the store's name encodes the window and
    format version that :class:`~src.io.event_store.EventStore` checks.

    Args:
        kind: ``"store"`` for the evaluation window, ``"tune_store"`` for the tuning window.
        dataset: name a dataset explicitly instead of using ``dataset.active``.

    Returns:
        The resolved path.

    Raises:
        ValueError: if the requested store is not set in the config.
    """
    active = dataset or active_dataset()
    entry = (settings() if dataset is None else settings_for(dataset))["dataset"][active]
    if kind not in entry or not entry[kind]:
        msg = (
            f"dataset.{active}.{kind} is not set in {CONFIG_PATH}. Produce a store with "
            f"hepattn.experiments.colliderml.eval.dump and point this at it."
        )
        raise ValueError(msg)
    configured = Path(entry[kind])
    root = os.environ.get("CALO_STORE_ROOT")
    if root:
        return Path(root) / configured.name
    return configured


def window(kind: str = "eval") -> tuple[int, int]:
    """``(start_event, n_events)`` for the active dataset's ``eval`` or ``tune`` window.

    Raises:
        ValueError: if `kind` is neither ``"eval"`` nor ``"tune"``.
    """
    if kind not in ("eval", "tune"):
        msg = f"Unknown window {kind!r}; expected 'eval' or 'tune'."
        raise ValueError(msg)
    windows = settings()["dataset"][active_dataset()]["windows"]
    return windows[f"{kind}_start"], windows[f"{kind}_events"]


def results_dir(create: bool = True, dataset: str | None = None) -> Path:
    """Where this dataset's tables go: ``results/<dataset>/``.

    Scoped by dataset so a pu200 run cannot overwrite a pu0 table of the same name.

    Args:
        create: make the directory if it is missing.
        dataset: name a dataset explicitly instead of using ``dataset.active``.
    """
    path = RESULTS_ROOT / (dataset or active_dataset())
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def clue_search(subsystem: str) -> dict:
    """Optuna search ranges for one subsystem, for the active coordinate system.

    An accessor rather than three levels of indexing, because the tuner reads the same ranges
    twice: once to sample from, once to test whether the optimum landed on a bound.
    """
    clue = settings()["clue"]
    return clue["search"][clue["coords"]][subsystem]


def store_expectations(dataset: str | None = None) -> dict:
    """Store metadata this config asserts, keyed by the leaf names of ``CONTRACT_KEYS``.

    Args:
        dataset: name a dataset explicitly. The expectations differ between the two, so a
            caller opening a named store must ask for that store's dataset.
    """
    cfg = settings() if dataset is None else settings_for(dataset)
    return {
        "length": "m",
        "energy": "GeV",
        **cfg["hit_selection"],
        **cfg["particle_selection"],
        "subsystem_order": list(cfg["detectors"]),
    }


def overrides() -> dict:
    """The active dataset's override block as written, before merging."""
    if _RAW is None:
        reload()
    assert _RAW is not None
    return deepcopy(_RAW["dataset"][_RAW["dataset"]["active"]].get("overrides") or {})


def describe() -> str:
    """Summary of what the active dataset resolved to, printed by every entry point."""
    active = active_dataset()
    lines = [f"dataset {active}   ->  results/{active}/"]

    replaced = sorted(_leaf_paths(overrides()))
    if replaced:
        lines.append(f"  overrides    {', '.join(replaced)}")

    # The RESOLVED path, not the configured one: they differ whenever CALO_STORE_ROOT is set,
    # and this function exists so that substitution cannot happen unseen.
    entry = settings()["dataset"][active]
    root = os.environ.get("CALO_STORE_ROOT")
    if root:
        lines.append(f"  store root   {root}  (CALO_STORE_ROOT; names come from the config)")
    for kind in ("store", "tune_store"):
        if not entry.get(kind):
            lines.append(f"  {kind:<12} (unset)")
            continue
        resolved = store_path(kind)
        lines.append(f"  {kind:<12} {resolved}{'' if not root else '   [relocated]'}")
    return "\n".join(lines)


def _leaf_paths(node, prefix: str = "") -> list[str]:
    """Dotted paths of every leaf in a nested mapping."""
    if not isinstance(node, dict):
        return [prefix]
    out: list[str] = []
    for key, value in node.items():
        out.extend(_leaf_paths(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def _merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` into a copy of `base`; mappings recurse, scalars and lists replace.

    A list replaces rather than extends: every list in this config is a complete statement, so
    appending to one produces an invalid value.
    """
    out = deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _resolve(raw: dict) -> dict:
    """Apply the active dataset's overrides to the shared settings."""
    active = raw["dataset"]["active"]
    over = raw["dataset"][active].get("overrides") or {}
    merged = _merge(raw, over)
    # Drop the applied block, so no consumer can read the unmerged value by accident.
    for name in raw["dataset"]:
        if isinstance(merged["dataset"].get(name), dict):
            merged["dataset"][name].pop("overrides", None)
    return merged


def _validate_raw(raw: dict) -> None:
    """Check the file's shape before anything is merged.

    Raises:
        ValueError: if the active dataset is unknown, has no ``windows`` block, or carries a
            key that is not a store path, a window or an override.
    """
    dataset = raw.get("dataset", {})
    active = dataset.get("active")
    if active not in DATASETS:
        msg = f"dataset.active must be one of {list(DATASETS)}, got {active!r}"
        raise ValueError(msg)
    if active not in dataset:
        msg = f"dataset.active is {active!r} but there is no dataset.{active} block in {CONFIG_PATH}"
        raise ValueError(msg)

    entry = dataset[active]
    if "windows" not in entry:
        msg = (
            f"dataset.{active} has no `windows` block. Each dataset carries its own, because "
            f"pu0 and pu200 are different files with different event numbering."
        )
        raise ValueError(msg)

    unknown = set(entry) - {"store", "tune_store", "windows", "overrides"}
    if unknown:
        msg = (
            f"dataset.{active} has unexpected key(s) {sorted(unknown)}. A per-dataset value that "
            f"is not a store path or a window belongs in that dataset's `overrides:` block."
        )
        raise ValueError(msg)


def _validate(cfg: dict) -> None:
    """Check the invariants that keep the two pipelines comparable.

    Raises:
        ValueError: if the tuning and evaluation windows overlap, which would mean CLUE was
            tuned on the events it is reported on while the MaskFormer was not; or if the
            detector list is not the four the store enumerates.
    """
    active = cfg["dataset"]["active"]
    windows = cfg["dataset"][active]["windows"]
    eval_start, eval_n = windows["eval_start"], windows["eval_events"]
    tune_start, tune_n = windows["tune_start"], windows["tune_events"]
    # A zero-length window is the "not configured yet" state, not an overlap.
    if eval_n and tune_n and tune_start < eval_start + eval_n and tune_start + tune_n > eval_start:
        msg = (
            f"CLUE's tuning window [{tune_start}, {tune_start + tune_n}) overlaps the "
            f"evaluation window [{eval_start}, {eval_start + eval_n}) for dataset {active}."
        )
        raise ValueError(msg)

    expected = {"ecb", "ece", "hcb", "hce"}
    if set(cfg["detectors"]) != expected:
        msg = f"detectors must be exactly {sorted(expected)}, got {cfg['detectors']}"
        raise ValueError(msg)
