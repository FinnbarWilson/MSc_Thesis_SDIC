"""The experiment definition: the overrides merge, and the guard on the event windows.

Two things :mod:`src.config` does are load-bearing for claims the report makes. The windows must
not overlap, or CLUE would have been tuned on the events it is scored over while the MaskFormer
was not. And the overrides merge must be invisible to consumers, since pu200 replaces the eta
acceptance, the thresholds and the checkpoint through it.

These run on dictionaries rather than on ``config/experiment.yaml``, so they pin the behaviour
and not the file's current contents, apart from two cases that deliberately load the real file.
"""

import copy

import pytest

from src import config


def _raw(tune_start=0, tune_events=50, eval_start=100, eval_events=500, overrides=None):
    """A minimal but structurally valid experiment file."""
    entry = {
        "store": "/tmp/store",
        "tune_store": "/tmp/tune_store",
        "windows": {
            "eval_start": eval_start, "eval_events": eval_events,
            "tune_start": tune_start, "tune_events": tune_events,
        },
    }
    if overrides is not None:
        entry["overrides"] = overrides
    return {
        "dataset": {"active": "pu0", "pu0": entry},
        "detectors": ["ecb", "ece", "hcb", "hce"],
        "metrics": {"min_cluster_hits": 1, "working_points": [0.5]},
        "maskformer": {"mask_threshold": 0.05, "object_threshold": 0.5},
    }


# --- the window guard -------------------------------------------------------


def test_disjoint_windows_are_accepted():
    config._validate(config._resolve(_raw(tune_start=0, tune_events=50, eval_start=100)))


def test_overlapping_windows_raise():
    """The one that matters: tuning CLUE on the events it is reported on."""
    raw = _raw(tune_start=90, tune_events=50, eval_start=100, eval_events=500)
    with pytest.raises(ValueError, match="overlaps the evaluation window"):
        config._validate(config._resolve(raw))


def test_a_tune_window_inside_the_eval_window_raises():
    """Containment is overlap too, and is the shape a careless edit actually produces."""
    raw = _raw(tune_start=200, tune_events=50, eval_start=100, eval_events=500)
    with pytest.raises(ValueError, match="overlaps"):
        config._validate(config._resolve(raw))


def test_windows_touching_end_to_start_are_disjoint():
    """[0, 50) and [50, 550) share no event; an off-by-one here would reject a valid setup."""
    config._validate(config._resolve(_raw(tune_start=0, tune_events=50, eval_start=50)))


def test_a_zero_length_window_is_unconfigured_not_overlapping():
    """pu200 ships with zeros so the numbers must come from the store that gets dumped.

    Failing here would block even loading the config to look at it.
    """
    config._validate(config._resolve(_raw(tune_events=0, tune_start=100, eval_start=100)))


def test_the_detector_list_is_pinned():
    """`detectors` is contract-checked against the store's subsystem order, so it is not free."""
    raw = _raw()
    raw["detectors"] = ["ecb", "hcb"]
    with pytest.raises(ValueError, match="detectors must be exactly"):
        config._validate(config._resolve(raw))


# --- the shape check that runs before the merge -----------------------------


def test_an_unknown_dataset_is_named_in_the_error():
    raw = _raw()
    raw["dataset"]["active"] = "pu100"
    with pytest.raises(ValueError, match="pu100"):
        config._validate_raw(raw)


def test_a_dataset_with_no_window_block_is_rejected():
    raw = _raw()
    del raw["dataset"]["pu0"]["windows"]
    with pytest.raises(ValueError, match="no `windows` block"):
        config._validate_raw(raw)


def test_a_stray_per_dataset_key_is_rejected_rather_than_ignored():
    """A per-dataset value that is not a store or a window belongs in `overrides:`.

    Accepting it silently is how a setting ends up somewhere no consumer reads it.
    """
    raw = _raw()
    raw["dataset"]["pu0"]["mask_threshold"] = 0.2
    with pytest.raises(ValueError, match="unexpected key"):
        config._validate_raw(raw)


# --- the overrides merge ----------------------------------------------------


def test_an_override_replaces_the_shared_value():
    resolved = config._resolve(_raw(overrides={"maskformer": {"mask_threshold": 0.5}}))
    assert resolved["maskformer"]["mask_threshold"] == 0.5


def test_merging_is_deep_and_leaves_siblings_alone():
    """The failure this prevents: overriding one threshold dropping the other."""
    resolved = config._resolve(_raw(overrides={"maskformer": {"mask_threshold": 0.5}}))
    assert resolved["maskformer"]["object_threshold"] == 0.5   # untouched sibling survives
    assert resolved["metrics"]["min_cluster_hits"] == 1


def test_a_list_is_replaced_not_extended():
    """Every list in the config is a complete statement: a grid, a range, an order.

    Appending to one produces something that is not a valid value of that key.
    """
    resolved = config._resolve(_raw(overrides={"metrics": {"working_points": [0.75]}}))
    assert resolved["metrics"]["working_points"] == [0.75]


def test_the_overrides_block_is_removed_after_it_is_applied():
    """Leaving a copy in the resolved settings invites reading the unmerged value by accident."""
    resolved = config._resolve(_raw(overrides={"maskformer": {"mask_threshold": 0.5}}))
    assert "overrides" not in resolved["dataset"]["pu0"]


def test_resolving_does_not_mutate_the_input():
    raw = _raw(overrides={"maskformer": {"mask_threshold": 0.5}})
    before = copy.deepcopy(raw)
    config._resolve(raw)
    assert raw == before


# --- the real file ----------------------------------------------------------


def test_the_committed_config_loads_and_validates():
    """Catches an edit to config/experiment.yaml that no script would notice until it ran."""
    cfg = config.reload()
    assert cfg["dataset"]["active"] in config.DATASETS
    assert set(cfg["detectors"]) == {"ecb", "ece", "hcb", "hce"}


def test_settings_hands_out_copies_so_one_caller_cannot_affect_another():
    """`settings()` is called by every script; a shared mutable dict would leak between them."""
    first = config.settings()
    first["metrics"]["min_cluster_hits"] = 999
    assert config.settings()["metrics"]["min_cluster_hits"] != 999


def test_settings_for_names_a_dataset_without_disturbing_the_active_one():
    """Used by the dataset-composition script, which describes pu0 while pu200 may be active."""
    active = config.active_dataset()
    other = next(d for d in config.DATASETS if d != active)
    config.settings_for(other)
    assert config.active_dataset() == active


def test_settings_for_rejects_an_unknown_dataset():
    with pytest.raises(ValueError, match="must be one of"):
        config.settings_for("pu100")
