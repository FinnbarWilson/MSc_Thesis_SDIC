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

**One active dataset at a time.** ``dataset.active`` selects pu0 or pu200, and that choice
reaches further than the store path: the active dataset's ``overrides`` block is deep-merged
over everything else before :func:`settings` returns, and :func:`results_dir` /
:func:`figures_dir` point at a subdirectory named after it. So switching pileup condition is
one edit, no output can land on top of the other condition's tables, and a value that differs
between the two is stated once in the block that owns it rather than being remembered.

:func:`describe` renders what that resolved to, and every entry point prints it, because an
implicit merge that nothing shows you is how the wrong number gets reported.
"""

import os
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "experiment.yaml"

#: Roots, not output directories. Everything a run writes goes under
#: ``<root>/<active dataset>/`` -- see :func:`results_dir` and :func:`figures_dir`.
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"

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
    """Return the experiment settings for the active dataset, as a nested dictionary.

    The active dataset's ``overrides`` block has already been merged in, so a consumer never
    has to ask which pileup condition it is running under.

    A copy is returned each call, so a caller mutating the result cannot alter what any
    other module sees. Do not call this inside a per-event or per-trial loop -- read it once
    and pass the values down. Use :func:`frozen` where a read-only view will do.
    """
    if _CACHE is None:
        return reload()
    return deepcopy(_CACHE)


def settings_for(dataset: str) -> dict:
    """Settings resolved against a NAMED dataset, whatever ``dataset.active`` says.

    Almost nothing should want this. The pipelines all run under one pileup condition at a
    time, and reading the other one's values by accident is the failure ``active`` exists to
    prevent -- which is why the active dataset is not a parameter anywhere else.

    The exception is a script that describes a dataset rather than scoring it. pu200 overrides
    ``particle_max_abs_eta`` to 0.88 for the barrel-only run, so drawing pu0's cuts from the
    ambient config while pu200 is active would put the selection lines in the wrong place on
    the figure and quietly misreport what the cuts cost.

    Nothing is cached: this bypasses the module state rather than replacing it, so a call here
    cannot change what :func:`settings` returns to anyone else.
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


def frozen() -> MappingProxyType:
    """A read-only view of the merged settings, with no copying cost."""
    if _CACHE is None:
        reload()
    assert _CACHE is not None
    return MappingProxyType(_CACHE)


def active_dataset() -> str:
    """The name of the dataset every path and every override resolves against."""
    if _CACHE is None:
        reload()
    assert _CACHE is not None
    return _CACHE["dataset"]["active"]


def store_path(kind: str = "store") -> Path:
    """Path to the event store for the active dataset.

    The path in the config is ce-ai-1's, under /mnt/ai-datastore. ``CALO_STORE_ROOT``
    relocates it without editing the config, which is what lets one config stay valid on
    two machines -- the same reason ``src/maskformer/dias/train.sh`` overrides the data
    directories rather than rewriting ``configs/pu0.yaml``.

    Only the DIRECTORY moves; the store's NAME is taken from the config unchanged, because
    ``eval/dump.py`` builds it as ``<prefix>_<start>_<end>_v<FORMAT_VERSION>`` and that name
    is the contract :class:`~src.io.event_store.EventStore` checks. Relocating a store must
    not be able to change which window or format version is being opened.

    Args:
        kind: ``"store"`` for the evaluation window, ``"tune_store"`` for the smaller
            window CLUE's parameter search runs on.
    """
    active = active_dataset()
    entry = settings()["dataset"][active]
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
    """Return ``(start_event, n_events)`` for the active dataset's ``eval`` or ``tune`` window.

    The store carries its own window and asserts it is disjoint from the checkpoint's
    training range, so this is the convenience copy; the check that has teeth is in
    :class:`~src.io.event_store.EventStore`.
    """
    if kind not in ("eval", "tune"):
        msg = f"Unknown window {kind!r}; expected 'eval' or 'tune'."
        raise ValueError(msg)
    windows = settings()["dataset"][active_dataset()]["windows"]
    return windows[f"{kind}_start"], windows[f"{kind}_events"]


def results_dir(create: bool = True) -> Path:
    """Where this dataset's tables go: ``results/<active dataset>/``.

    Scoped by dataset rather than tagged within one directory, because the tables are read
    back by name (``particles_clue.parquet``) and a pu200 run writing beside a pu0 one would
    silently replace it -- and, worse, ``make_thesis_figures`` would then draw one column from
    two pileup conditions without anything looking wrong.
    """
    path = RESULTS_ROOT / active_dataset()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(create: bool = True) -> Path:
    """Where this dataset's figures go: ``figures/<active dataset>/``."""
    path = FIGURES_ROOT / active_dataset()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def clue_search(subsystem: str) -> dict:
    """The Optuna search ranges for one subsystem, resolved for the active coordinates.

    An accessor rather than three levels of indexing at each call site: the tuner needs the
    same ranges twice, once to sample from and once to test whether the winning value came
    back pressed against a bound, and those two must not be able to disagree.
    """
    clue = settings()["clue"]
    return clue["search"][clue["coords"]][subsystem]


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


def overrides() -> dict:
    """The active dataset's override block, as written -- before merging.

    Useful for reporting. :func:`settings` has already applied it.
    """
    if _RAW is None:
        reload()
    assert _RAW is not None
    return deepcopy(_RAW["dataset"][_RAW["dataset"]["active"]].get("overrides") or {})


def describe() -> str:
    """A one-paragraph summary of what the active dataset resolved to.

    Printed by every entry point. The override merge is convenient and invisible, and an
    invisible thing that changes reported numbers has to be shown somewhere.
    """
    active = active_dataset()
    lines = [f"dataset {active}   ->  results/{active}/  figures/{active}/"]

    replaced = sorted(_leaf_paths(overrides()))
    if replaced:
        lines.append(f"  overrides    {', '.join(replaced)}")

    # The RESOLVED path, not the configured one, and they differ whenever CALO_STORE_ROOT is set.
    # This function exists precisely so an invisible substitution cannot change reported numbers
    # unseen, so printing the config's own string here would defeat it: show_config is the first
    # thing run when a path looks wrong, and it would have confirmed a path nothing was reading.
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
    """Dotted paths of every leaf in a nested mapping, for :func:`describe`."""
    if not isinstance(node, dict):
        return [prefix]
    out: list[str] = []
    for key, value in node.items():
        out.extend(_leaf_paths(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def _merge(base: dict, over: dict) -> dict:
    """Deep-merge `over` into a copy of `base`; scalars and lists replace, mappings recurse.

    A list replaces rather than extends on purpose. Every list in this file is a complete
    statement -- the subsystem order, a scan's grid, a search range's two endpoints -- and
    appending to any of them produces something that is not a valid value of that key.
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
    # The block has been applied; leaving a copy in the resolved settings invites a consumer
    # to read the unmerged value by accident.
    for name in raw["dataset"]:
        if isinstance(merged["dataset"].get(name), dict):
            merged["dataset"][name].pop("overrides", None)
    return merged


def _validate_raw(raw: dict) -> None:
    """Check the shape of the file before anything is merged.

    Runs first so that a misspelled dataset name is reported as itself, rather than as
    whatever the merge then fails to find.
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
            f"dataset.{active} has no `windows` block. Event windows moved from the top level "
            f"into each dataset when pu200 was added -- pu0 and pu200 are different files with "
            f"different event numbering, so one shared window could only be right for one."
        )
        raise ValueError(msg)

    unknown = set(entry) - {"store", "tune_store", "windows", "overrides"}
    if unknown:
        msg = (
            f"dataset.{active} has unexpected key(s) {sorted(unknown)}. A per-dataset value that "
            f"is not a store path or a window belongs in that dataset's `overrides:` block, so "
            f"that it lands where the shared setting it replaces is documented."
        )
        raise ValueError(msg)


def _validate(cfg: dict) -> None:
    """Check the invariants that keep the two pipelines comparable.

    Raises:
        ValueError: if the tuning and evaluation windows overlap -- which would mean CLUE had
            been tuned on the events it is reported on while the MaskFormer had not, an
            advantage that has nothing to do with either algorithm.
    """
    active = cfg["dataset"]["active"]
    windows = cfg["dataset"][active]["windows"]
    eval_start, eval_n = windows["eval_start"], windows["eval_events"]
    tune_start, tune_n = windows["tune_start"], windows["tune_events"]
    # A window of zero events is the "not configured yet" state, not an overlap: pu200 ships
    # with zeros so that the numbers have to be filled in from the store that gets dumped,
    # and failing here would block even loading the config to look at it.
    if eval_n and tune_n and tune_start < eval_start + eval_n and tune_start + tune_n > eval_start:
        msg = (
            f"CLUE's tuning window [{tune_start}, {tune_start + tune_n}) overlaps the "
            f"evaluation window [{eval_start}, {eval_start + eval_n}) for dataset {active}. "
            f"Tuning CLUE on the reported events would flatter it relative to the MaskFormer, "
            f"which was not."
        )
        raise ValueError(msg)

    expected = {"ecb", "ece", "hcb", "hce"}
    if set(cfg["detectors"]) != expected:
        msg = f"detectors must be exactly {sorted(expected)}, got {cfg['detectors']}"
        raise ValueError(msg)
