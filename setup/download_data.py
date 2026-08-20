#!/usr/bin/env python
"""Download ColliderML ttbar shards from the HuggingFace Hub.

    python setup/download_data.py                    # pu200, first 100 shards = 10,000 events
    python setup/download_data.py --pileup pu0 --shards 20
    COLLIDERML_DATA=/somewhere/else python setup/download_data.py

The default is ~297 GB and is resumable; re-running skips what is already complete. The two
collections are downloaded under matching filenames because ``ColliderMLDataset`` pairs a
particles shard with the calo_hits shard of the same name and takes the intersection of the two
listings, so an unmatched shard is dropped silently. ``setup/verify_data.py`` checks for that.

Needs `huggingface_hub` and nothing else.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "CERN/ColliderML-Release-1"
TOTAL_SHARDS = 1000  # the "-of-01000" in every filename

# Rows per shard, which is events per shard, and is not the same for the two pileup
# conditions. Read from the parquet footers on the Hub rather than assumed:
#
#   ttbar_pu0    1,000 rows/shard  ->  1,000,000 events in the release,  ~1.06 GB/shard
#   ttbar_pu200    100 rows/shard  ->    100,000 events in the release,  ~2.97 GB/shard
#
# pu0 packs 10x more events into a shard of a third the size, a pu0 event holding ~22k calo hits
# against pu200's ~532k. One constant for both would misreport the event count by 10x, which is
# the line anyone sizing a download reads.
EVENTS_PER_SHARD: dict[str, int] = {"pu0": 1000, "pu200": 100}
COLLECTIONS = ("calo_hits", "particles")


def main() -> None:
    default_root = os.environ.get("COLLIDERML_DATA", "external/ColliderML_data")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pileup", default="pu200", choices=["pu0", "pu200"])
    p.add_argument("--event-type", default="ttbar")
    p.add_argument("--shards", type=int, default=100, help="number of shards per collection")
    p.add_argument("--dest", default=default_root, help=f"destination (default: {default_root})")
    args = p.parse_args()

    prefix = f"{args.event_type}_{args.pileup}"
    patterns = [
        f"data/{prefix}_{c}/train-{i:05d}-of-{TOTAL_SHARDS:05d}.parquet"
        for c in COLLECTIONS
        for i in range(args.shards)
    ]

    print(f"repo      : {REPO}")
    print(f"dataset   : {prefix}")
    print(f"dest      : {args.dest}")
    print(f"files     : {len(patterns)} ({args.shards} shards x {len(COLLECTIONS)} collections)")
    per_shard = EVENTS_PER_SHARD[args.pileup]
    print(f"events    : {args.shards * per_shard:,} ({per_shard:,}/shard for {args.pileup})", flush=True)

    # Keep the Hub's own caches beside the data rather than in ~/.cache: the Xet chunk cache is
    # large and / is the small disk on this machine.
    dest = Path(args.dest)
    os.environ.setdefault("HF_HOME", str(dest.parent / ".hf"))
    os.environ.setdefault("HF_XET_CACHE", str(dest.parent / ".hf" / "xet"))

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
        max_workers=8,
    )

    # The loader wants "<dirpath>/<prefix>_<collection>", while the Hub lays the collections out
    # under "data/". A symlink gives that layout without moving 300 GB, and mirrors how the pu0
    # data was addressed on the old cluster (<root>/ttbar_pu0/ttbar_pu0_calo_hits).
    link = dest / prefix
    if not link.exists():
        link.symlink_to("data")
        print(f"\nlinked {link} -> data")

    print(f"\nDONE. Point train_dir at {link}/ with dataset_prefix: {prefix}")
    print("Now run: python setup/verify_data.py")


if __name__ == "__main__":
    main()
