#!/usr/bin/env python
"""Verify the downloaded pu200 shards are complete, readable and correctly paired.

Checks the three things that would otherwise fail late, inside training:

1. All 100 shards present in BOTH collections, with no leftover .incomplete files.
2. Filenames match across the two collections. ColliderMLDataset pairs a particles
   shard with the calo_hits shard of the same name and takes the INTERSECTION of the
   two listings, so an unmatched shard is dropped silently -- the event count just
   comes out lower than expected with no error.
3. Every file's parquet footer parses and reports 100 rows, so the event total is
   really 10,000. A truncated download usually still has a plausible size.

    python setup/verify_data.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(os.environ.get("COLLIDERML_DATA", "/mnt/ai-datastore/finnbar/ColliderML_data")) / "data"
COLLECTIONS = ("calo_hits", "particles")
EXPECTED_SHARDS = 100
EXPECTED_ROWS_PER_SHARD = 100

failures: list[str] = []
names: dict[str, set[str]] = {}

for c in COLLECTIONS:
    d = ROOT / f"ttbar_pu200_{c}"
    if not d.is_dir():
        failures.append(f"missing directory {d}")
        names[c] = set()
        continue

    incomplete = list(d.glob("*.incomplete")) + list(d.glob("*.lock"))
    if incomplete:
        failures.append(f"{c}: {len(incomplete)} unfinished files still present")

    files = sorted(d.glob("*.parquet"))
    names[c] = {p.name for p in files}
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"{c}: {len(files)} shards, {total_bytes / 1e9:.1f} GB")

    if len(files) != EXPECTED_SHARDS:
        failures.append(f"{c}: expected {EXPECTED_SHARDS} shards, found {len(files)}")

    rows = 0
    for p in files:
        try:
            md = pq.ParquetFile(p).metadata
        except Exception as exc:  # noqa: BLE001 - report the file, keep checking the rest
            failures.append(f"{c}/{p.name}: unreadable footer ({type(exc).__name__}: {exc})")
            continue
        if md.num_rows != EXPECTED_ROWS_PER_SHARD:
            failures.append(f"{c}/{p.name}: {md.num_rows} rows, expected {EXPECTED_ROWS_PER_SHARD}")
        rows += md.num_rows
    print(f"{c}: {rows} events across readable shards")

shared = names[COLLECTIONS[0]] & names[COLLECTIONS[1]]
print(f"\nshared shard names (what the loader will actually use): {len(shared)}")
print(f"-> {len(shared) * EXPECTED_ROWS_PER_SHARD} events")

for c in COLLECTIONS:
    unmatched = names[c] - shared
    if unmatched:
        failures.append(f"{c}: {len(unmatched)} shards have no counterpart and will be dropped")

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("\nOK: 100 matched shards per collection, 10,000 events.")
