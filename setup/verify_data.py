#!/usr/bin/env python
"""Verify downloaded ColliderML shards are complete, readable and correctly paired.

    python setup/verify_data.py                          # pu200, the 100 shards download_data.py fetches
    python setup/verify_data.py --pileup pu0 --shards 21

Checks the three things that would otherwise fail late, inside training or the dump: that every
shard is present in both collections with no leftover ``.incomplete`` files; that the filenames
match across the two, since an unmatched shard is dropped silently rather than reported; and that
every parquet footer parses and reports the expected row count, a truncated download usually
still having a plausible size.

Pass the same ``--pileup`` and ``--shards`` you gave ``download_data.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

COLLECTIONS = ("calo_hits", "particles")
#: Events per shard, which differs by an order of magnitude between the two conditions.
ROWS_PER_SHARD = {"pu0": 1000, "pu200": 100}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pileup", default="pu200", choices=sorted(ROWS_PER_SHARD))
    p.add_argument("--shards", type=int, default=100, help="how many were downloaded")
    p.add_argument("--data-root", default=os.environ.get("COLLIDERML_DATA", "external/ColliderML_data"))
    args = p.parse_args()

    root = Path(args.data_root) / "data"
    expected_rows = ROWS_PER_SHARD[args.pileup]
    failures: list[str] = []
    names: dict[str, set[str]] = {}

    for c in COLLECTIONS:
        d = root / f"ttbar_{args.pileup}_{c}"
        if not d.is_dir():
            failures.append(f"missing directory {d}")
            names[c] = set()
            continue

        unfinished = list(d.glob("*.incomplete")) + list(d.glob("*.lock"))
        if unfinished:
            failures.append(f"{c}: {len(unfinished)} unfinished files still present")

        files = sorted(d.glob("*.parquet"))
        names[c] = {f.name for f in files}
        print(f"{c}: {len(files)} shards, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB")

        if len(files) != args.shards:
            failures.append(f"{c}: expected {args.shards} shards, found {len(files)}")

        rows = 0
        for f in files:
            try:
                meta = pq.ParquetFile(f).metadata
            except Exception as exc:  # noqa: BLE001 - report the file, keep checking the rest
                failures.append(f"{c}/{f.name}: unreadable footer ({type(exc).__name__}: {exc})")
                continue
            if meta.num_rows != expected_rows:
                failures.append(f"{c}/{f.name}: {meta.num_rows} rows, expected {expected_rows}")
            rows += meta.num_rows
        print(f"{c}: {rows} events across readable shards")

    shared = names[COLLECTIONS[0]] & names[COLLECTIONS[1]]
    print(f"\nshared shard names (what the loader will actually use): {len(shared)}")
    print(f"-> {len(shared) * expected_rows} events")

    for c in COLLECTIONS:
        unmatched = names[c] - shared
        if unmatched:
            failures.append(f"{c}: {len(unmatched)} shards have no counterpart and will be dropped")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nOK: {args.shards} matched shards per collection, {args.shards * expected_rows:,} events.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
