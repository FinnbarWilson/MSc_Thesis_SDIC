"""Regenerate every figure from the pooled tables.

This is the only script an assessor needs to run. It touches no checkpoint, no GPU and no
ColliderML file -- only the parquet tables written by ``scripts.score``, which are small
enough to sit beside the thesis.

    python -m scripts.make_figures --tags maskformer clue
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR, settings, store_path
from src.evaluation.differential import reference_table
from src.plotting import figures, style

REFERENCE_CUTS = [
    ("all particles", ""),
    ("E > 1 GeV", "p_energy > 1"),
    ("E > 5 GeV", "p_energy > 5"),
    ("E > 20 GeV", "p_energy > 20"),
    ("isolated (dR > 0.2)", "dr_min > 0.2"),
    ("jet core (dR < 0.1)", "dr_min < 0.1"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tags", nargs="+",
        default=["maskformer", "maskformer_incidence", "clue", "oracle_geometric", "oracle_resolution"],
        help="table suffixes to load; missing ones are skipped with a warning",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="output formats")
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="store meta.json for the definitions paragraph; defaults to the active store's",
    )
    args = parser.parse_args()

    style.apply()
    args.out.mkdir(parents=True, exist_ok=True)

    # The reference clusterings are optional: the thesis figures are still meaningful without
    # them, they are just far harder to read, so a missing one is a warning and not a stop.
    available = [t for t in args.tags if (args.results / f"particles_{t}.parquet").exists()]
    for missing in [t for t in args.tags if t not in available]:
        print(f"  ! no tables for {missing!r}; run  python -m scripts.score --algo {missing}")

    particles = pd.concat([pd.read_parquet(args.results / f"particles_{t}.parquet") for t in available], ignore_index=True)
    clusters = pd.concat([pd.read_parquet(args.results / f"clusters_{t}.parquet") for t in available], ignore_index=True)
    print(f"{len(particles)} particle rows, {len(clusters)} cluster rows, algorithms {sorted(particles['algo'].unique())}")

    working_point = settings()["metrics"]["working_points"][0]

    # Two figures were dropped rather than restyled, because neither carried anything its
    # neighbours did not. `reference_ceiling` drew the same two panels as `eff_pur_vs_energy`
    # against the same references; `fake_and_match_rates` drew the cluster match rate, now
    # panel 3 of `efficiency_decomposition`, beside a particle match rate that was already
    # panel 1 of it with a different x-axis.
    panels = {
        "eff_pur_vs_energy": lambda p: figures.efficiency_and_purity_vs_energy(particles, clusters, working_point, out=p),
        "efficiency_decomposition": lambda p: figures.efficiency_decomposition(particles, clusters, working_point, out=p),
        "performance_vs_density": lambda p: figures.performance_vs_density(particles, working_point, out=p),
        "split_and_merge": lambda p: figures.split_and_merge(particles, clusters, out=p),
        "energy_decomposition": lambda p: figures.energy_decomposition(particles, out=p),
        "incidence_head_comparison": lambda p: figures.incidence_head_comparison(particles, clusters, working_point, out=p),
        "weighting_comparison": lambda p: figures.weighting_comparison(particles, clusters, out=p),
    }

    # The multi-owner capability study no longer produces a figure. Its result is two
    # numbers -- 0.56 of otherwise-unreachable particles recovered against CLUE's 0.007 --
    # and a bar chart of two numbers is a sentence with axes. They stay in
    # results/capability_summary.csv and are quoted in the README, which is the right home
    # for them; scripts.score_soft still writes the tables.
    # PDF as well as PNG: vector output is what goes into the thesis, and matplotlib embeds
    # the fonts, so the figures survive being dropped into LaTeX at any scale.
    for name, draw in panels.items():
        for suffix in args.formats:
            draw(args.out / f"{name}.{suffix}")
    print(f"wrote {len(panels)} figures to {args.out} as {', '.join(args.formats)}")

    table = reference_table(particles, clusters, REFERENCE_CUTS, working_point)
    table.to_csv(args.results / "reference_table.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # The definitions paragraph is deliverable 7 and is generated from the store's metadata so
    # it cannot contradict the run. It used to be written only when --meta was passed by hand,
    # which is a footgun rather than an option: pointing it at a previous store produced a
    # paragraph describing a DIFFERENT checkpoint from the one in the tables beside it, and
    # nothing complained. Defaulting to the active store removes the chance to get it wrong,
    # and the check below catches the remaining case where the config has moved on from the
    # store without it being re-dumped.
    meta_path = args.meta or (store_path() / "meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cfg = settings()

        store_ckpt = str(meta["maskformer"]["checkpoint"]).rsplit("/", 1)[-1]
        config_ckpt = str(cfg["maskformer"]["checkpoint"]).rsplit("/", 1)[-1]
        if store_ckpt != config_ckpt:
            print(
                f"\n  ! the store was dumped from {store_ckpt} but config/experiment.yaml names "
                f"{config_ckpt}.\n"
                f"    The tables come from the STORE, so the paragraph describes {store_ckpt}. "
                f"Re-dump or fix the config."
            )

        paragraph = figures.definitions_paragraph(meta, cfg["metrics"])
        (args.results / "definitions.txt").write_text(paragraph + "\n")
        print("\n" + paragraph)
    else:
        print(f"\n  ! no store metadata at {meta_path}; definitions.txt not regenerated")


if __name__ == "__main__":
    main()
