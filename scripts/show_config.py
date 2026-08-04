"""Print what the active dataset actually resolved to, and stop.

    python -m scripts.show_config
    python -m scripts.show_config --full

The experiment definition has one implicit step in it: the active dataset's ``overrides``
block is deep-merged over everything else before any consumer sees the settings. That is
what makes switching pileup condition a one-line edit, and it is also the one thing in the
config that cannot be checked by reading the file. This prints the result, so "which value is
this run using" is a command rather than a deduction.

It opens no store and imports no CLUEstering, so it works before either dataset is dumped and
is the first thing to run when a path looks wrong.
"""

import argparse

import yaml

from src import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="dump the whole merged configuration")
    args = parser.parse_args()

    print(config.describe())

    cfg = config.settings()
    active = cfg["dataset"]["active"]
    eval_window, tune_window = config.window("eval"), config.window("tune")
    print(f"  eval window  [{eval_window[0]}, {eval_window[0] + eval_window[1]})   "
          f"tune window [{tune_window[0]}, {tune_window[0] + tune_window[1]})")

    mf = cfg["maskformer"]
    print(f"  maskformer   mask {mf['mask_threshold']}  object {mf['object_threshold']}  "
          f"incidence floor {mf.get('incidence_floor', 0.0)}")
    print(f"  checkpoint   {str(mf['checkpoint']).rsplit('/', 1)[-1] or '(unset)'}")
    print(f"  clue         {cfg['clue']['coords']}, {cfg['clue']['optuna_trials']} trials, "
          f"backend {cfg['clue']['backend']!r}")

    # The search ranges are the values most likely to be wrong on a new dataset, and the most
    # tedious to confirm by eye through two levels of YAML nesting, so they are printed in full.
    print("  clue search ranges")
    for subsystem in cfg["detectors"]:
        ranges = config.clue_search(subsystem)
        formatted = "  ".join(f"{k}=[{lo:g}, {hi:g}]" for k, (lo, hi) in ranges.items())
        print(f"    {subsystem:<4} {formatted}")

    params = config.results_dir(create=False) / "clue_parameters.json"
    print(f"  clue params  {params}  {'(present)' if params.exists() else '(not tuned yet)'}")

    if args.full:
        print("\n--- merged configuration ---")
        print(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    elif active != "pu0":
        print("\n  --full prints the whole merged configuration.")


if __name__ == "__main__":
    main()
