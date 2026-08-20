"""Download the trained MaskFormer checkpoints from the GitHub release.

    python -m scripts.fetch_checkpoints                 # every published checkpoint
    python -m scripts.fetch_checkpoints --datasets pu200
    python -m scripts.fetch_checkpoints --list          # what exists, and what is already local

They are release assets rather than git objects: each is ~112 MB, past GitHub's 100 MB limit for
an ordinary object, and someone who only wants to redraw the figures should not pay for the
download. Each lands at the exact path ``config/experiment.yaml`` already names for it, under
the gitignored ``external/`` tree, so nothing in the config changes when one is fetched.

Every asset carries its SHA-256 here and is verified after transfer. A local file whose digest
disagrees is reported and left alone rather than replaced, since it is more likely a locally
trained checkpoint than a corrupt download. ``--force`` overrides that.

A checkpoint is needed only to dump a new event store.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Assets are addressed by tag rather than release id, so the URL survives the release being
#: edited.
DEFAULT_TAG = "checkpoints-v1"
REPO = "FinnbarWilson/MSc_Thesis_SDIC"


@dataclass(frozen=True)
class Checkpoint:
    """One published checkpoint.

    `sha256` is None until the asset is published: whoever uploads it replaces the None with the
    digest `sha256sum` prints, and nothing else here changes.
    """

    dataset: str
    asset: str
    dest: Path
    sha256: str | None
    run: str
    note: str


CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        dataset="pu200",
        asset="maskformer_pu200.ckpt",
        dest=Path("external/hepattn/src/hepattn/experiments/colliderml/logs"
                  "/hepattn_20260813-T134117/ckpts/epoch=003-val_loss=2.06527.ckpt"),
        sha256="c8f78ca64b5b95fc01e94d4f42a6f1b41376503a1f2bfd2c93faef9d50e7b2a1",
        run="hepattn_20260813-T134117",
        note="4 epochs x 6,000 events at batch 1, 1600 queries, shower-level truth "
             "(particle_min_num_calohits 10). Trained on ce-ai-1, finished 2026-08-14 17:05.",
    ),
    Checkpoint(
        dataset="pu0",
        asset="maskformer_pu0.ckpt",
        dest=Path("external/hepattn/src/hepattn/experiments/colliderml/logs"
                  "/hepattn_20260813-T145153/ckpts/epoch=005-val_loss=1.65500.ckpt"),
        sha256="72a3b058cc540b8fdd65c565ba7354d03fb93ed270c9f7be4e269a0c70225029",
        run="hepattn_20260813-T145153",
        note="6 epochs / 120,000 steps, 400 queries, shower-level truth "
             "(particle_min_num_calohits 10). Trained on DIAS.",
    ),
)


def digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def url_for(ckpt: Checkpoint, tag: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{ckpt.asset}"


def download(ckpt: Checkpoint, tag: str, root: Path) -> bool:
    """Fetch one checkpoint. Returns True if the file is present and verified afterwards."""
    dest = root / ckpt.dest
    if ckpt.sha256 is None:
        print(f"  {ckpt.dataset}: not published yet ({ckpt.run}), skipping")
        return False

    if dest.exists():
        have = digest(dest)
        if have == ckpt.sha256:
            print(f"  {ckpt.dataset}: already present and verified")
            return True
        print(f"  ! {ckpt.dataset}: {dest} exists but its digest does not match the release.")
        print(f"    local  {have}\n    release {ckpt.sha256}")
        print("    Left alone; this is most likely a locally trained checkpoint. "
              "Pass --force to replace it.")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url_for(ckpt, tag)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  {ckpt.dataset}: downloading {ckpt.asset} ...", flush=True)
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        extra = ("  The release or asset does not exist yet. See --list."
                 if exc.code == 404 else "")
        print(f"  ! {ckpt.dataset}: HTTP {exc.code} fetching {url}.{extra}")
        return False
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        print(f"  ! {ckpt.dataset}: {exc}")
        return False

    # Verify before moving into place: a half-written checkpoint at the configured path is worse
    # than none, since the config still resolves and the failure surfaces as a torch load error.
    got = digest(tmp)
    if got != ckpt.sha256:
        tmp.unlink(missing_ok=True)
        print(f"  ! {ckpt.dataset}: digest mismatch after download, discarded.")
        print(f"    expected {ckpt.sha256}\n    got      {got}")
        return False
    tmp.replace(dest)
    print(f"  {ckpt.dataset}: verified -> {ckpt.dest}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=[c.dataset for c in CHECKPOINTS])
    ap.add_argument("--tag", default=DEFAULT_TAG, help=f"release tag (default: {DEFAULT_TAG})")
    ap.add_argument("--list", action="store_true", help="show what exists and what is local")
    ap.add_argument("--force", action="store_true",
                    help="replace a local file whose digest does not match the release")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wanted = [c for c in CHECKPOINTS if c.dataset in args.datasets]
    if not wanted:
        raise SystemExit(f"no checkpoint for {args.datasets}; "
                         f"known: {[c.dataset for c in CHECKPOINTS]}")

    if args.list:
        for c in wanted:
            dest = root / c.dest
            state = ("published" if c.sha256 else "not published")
            local = "present" if dest.exists() else "absent"
            print(f"{c.dataset:<6} {state:<14} local: {local:<8} {c.run}")
            print(f"       {c.note}")
            if c.sha256:
                print(f"       {url_for(c, args.tag)}")
        return

    if args.force:
        for c in wanted:
            dest = root / c.dest
            if c.sha256 and dest.exists() and digest(dest) != c.sha256:
                print(f"  {c.dataset}: --force, removing mismatched {c.dest}")
                dest.unlink()

    ok = [download(c, args.tag, root) for c in wanted]
    if not all(ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
