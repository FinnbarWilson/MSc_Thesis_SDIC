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

from src.config import FIGURES_DIR, RESULTS_DIR, settings
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
        default=["maskformer", "clue", "oracle_geometric", "oracle_resolution"],
        help="table suffixes to load; missing ones are skipped with a warning",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="output formats")
    parser.add_argument("--meta", type=Path, default=None, help="store meta.json, for the definitions paragraph")
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

    panels = {
        "reference_ceiling": lambda p: figures.reference_ceiling(particles, clusters, working_point, out=p),
        "eff_pur_vs_energy": lambda p: figures.efficiency_and_purity_vs_energy(particles, clusters, working_point, out=p),
        "performance_vs_density": lambda p: figures.performance_vs_density(particles, working_point, out=p),
        "split_and_merge": lambda p: figures.split_and_merge(particles, clusters, out=p),
        "weighting_comparison": lambda p: figures.weighting_comparison(particles, clusters, out=p),
        "energy_decomposition": lambda p: figures.energy_decomposition(particles, out=p),
        "efficiency_decomposition": lambda p: figures.efficiency_decomposition(particles, working_point, out=p),
        "fake_and_match_rates": lambda p: figures.fake_and_match_rates(particles, clusters, out=p),
    }

    # The capability study is optional: it needs scripts.score_soft, which re-clusters rather
    # than reading the head-to-head tables, so it is not part of the default pipeline.
    soft_paths = sorted(args.results.glob("soft_particles_*.parquet"))
    if soft_paths:
        soft = pd.concat([pd.read_parquet(p) for p in soft_paths], ignore_index=True)
        panels["multiowner_capability"] = lambda p: figures.multiowner_capability(soft, working_point, out=p)
    else:
        print("  ! no soft tables; run  python -m scripts.score_soft  for the capability study")
    # PDF as well as PNG: vector output is what goes into the thesis, and matplotlib embeds
    # the fonts, so the figures survive being dropped into LaTeX at any scale.
    for name, draw in panels.items():
        for suffix in args.formats:
            draw(args.out / f"{name}.{suffix}")
    print(f"wrote {len(panels)} figures to {args.out} as {', '.join(args.formats)}")

    table = reference_table(particles, clusters, REFERENCE_CUTS, working_point)
    table.to_csv(args.results / "reference_table.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.meta and args.meta.exists():
        paragraph = figures.definitions_paragraph(json.loads(args.meta.read_text()), settings()["metrics"])
        (args.results / "definitions.txt").write_text(paragraph + "\n")
        print("\n" + paragraph)


if __name__ == "__main__":
    main()
