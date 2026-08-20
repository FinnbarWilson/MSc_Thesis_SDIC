"""Turn the per-event timing dumps into the table that goes in the report.

    python -m scripts.make_bench_table

Reads every ``bench_*.json`` under ``results/<dataset>/``, writes
``results/<dataset>/timing_summary.csv`` and prints one markdown table across both conditions.

Throughput is ``1 / median latency`` for one event on one device with no pipelining, which is
the reading a batch-1 measurement supports, and a lower bound on a served system. The p95 column is
there because a trigger has a per-event deadline that a comfortable mean can still miss. The
``ms/1000 cells`` column is the only one comparable across the two pileup conditions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATASETS, RESULTS_ROOT

#: Row order and the names the table prints. Keyed by (method, stage-1 backend, stage-2 backend),
#: where a stage-2 entry of None means both stages ran on the same backend.
ROWS = {
    ("clue", "cpu serial", None): (0, "CLUE — CPU serial", "1 core, Xeon Gold 6230R"),
    ("clue", "cpu openmp", None): (1, "CLUE — CPU parallel", "16 threads, Xeon Gold 6230R"),
    ("clue", "gpu cuda", None): (2, "CLUE — GPU, all stages", "A100 80GB"),
    ("clue", "cpu serial", "gpu cuda"): (3, "CLUE — CPU 2D + GPU 3D", "1 core + A100 80GB"),
    ("maskformer", "cuda bf16", None): (4, "MaskFormer", "A100 80GB, bfloat16"),
}


def summarise(path: Path) -> dict:
    """One row: the distribution of per-event latencies, plus what produced it."""
    blob = json.loads(path.read_text())
    frame = pd.DataFrame(blob["samples"])
    elapsed_ms = frame["elapsed_ns"].to_numpy() / 1e6

    key = (blob["method"], blob["backend"], blob.get("backend_3d"))
    if key not in ROWS:
        msg = f"{path.name} holds an unrecognised configuration {key}; add it to ROWS"
        raise SystemExit(msg)
    order, name, hardware = ROWS[key]

    # Per-pass medians: a spread wider than a few percent means something else was on the
    # machine and the row is not a measurement of this code.
    per_pass = frame.groupby("repeat")["elapsed_ns"].median().to_numpy() / 1e6
    median = float(np.median(elapsed_ms))

    return {
        "dataset": blob["dataset"],
        "configuration": name,
        "hardware": hardware,
        "n_events": int(frame["sample_id"].nunique()),
        "n_passes": int(frame["repeat"].nunique()),
        "mean_cells": float(frame["n_hits"].mean()),
        "mean_clusters": float(frame["n_clusters"].mean()),
        "median_ms": median,
        "p95_ms": float(np.percentile(elapsed_ms, 95)),
        "throughput_hz": 1000.0 / median,
        "ms_per_1k_cells": float(np.median(elapsed_ms / frame["n_hits"].to_numpy() * 1e3)),
        "pass_spread": float(per_pass.max() / per_pass.min() - 1.0),
        "sort_key": order,
    }


def markdown(table: pd.DataFrame) -> str:
    header = (
        "| Pile-up | Configuration | Hardware | Median ms | p95 ms | Throughput (events/s) |\n"
        "|---|---|---|---:|---:|---:|\n"
    )
    lines = []
    for dataset in DATASETS:
        block = table[table["dataset"] == dataset]
        label = {"pu0": "pileup 0", "pu200": "pileup 200"}.get(dataset, dataset)
        for i, row in enumerate(block.itertuples()):
            lines.append(
                f"| {label if i == 0 else ''} | {row.configuration} | {row.hardware} | "
                f"{row.median_ms:,.1f} | {row.p95_ms:,.1f} | {row.throughput_hz:,.2f} |"
            )
    return header + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()

    rows = []
    for dataset in DATASETS:
        for path in sorted((args.results / dataset).glob("bench_*.json")):
            rows.append(summarise(path))
    if not rows:
        raise SystemExit(f"no bench_*.json under {args.results}/<dataset>/; run scripts.bench_clue first")

    table = pd.DataFrame(rows).sort_values(["dataset", "sort_key"]).drop(columns="sort_key")

    for dataset in table["dataset"].unique():
        out = args.results / dataset / "timing_summary.csv"
        table[table["dataset"] == dataset].to_csv(out, index=False)
        print(f"wrote {out}")

    loose = table[table["pass_spread"] > 0.05]
    if not loose.empty:
        print("\n! these rows' three passes disagreed by more than 5%, so the machine was not quiet:")
        for row in loose.itertuples():
            print(f"    {row.dataset} {row.method} {row.hardware}: spread {row.pass_spread:.1%}")

    print()
    print(markdown(table))


if __name__ == "__main__":
    main()
