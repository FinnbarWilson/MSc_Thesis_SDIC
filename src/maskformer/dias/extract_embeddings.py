"""Save the encoder's per-cell embeddings, so a pair classifier can be trained on them offline.

WHY
---
The learned per-cell attribution model in `src/postproc/attribute.py` reached AUC 0.765 and its
dominant feature was distance-to-nearest-cell by 3x, which is why stacking it on geometric chaining
changed nothing. That result was used to argue that every post-hoc signal reduces to proximity --
but it was measured WITHOUT the encoder's embeddings, because those are not in the event store. The
one feature most likely to falsify the conclusion was missing from the test.

Raw cosine similarity of those embeddings already separates same-particle cell pairs at AUC 0.688
against plain distance's 0.661, with no training pointed at that task at all. A learned readout
should do better than raw cosine, since 256-dim cosine is dominated by variance that has nothing to
do with co-membership.

So this extracts the embeddings once, and the question becomes a CPU experiment: does a classifier
with embeddings beat the same classifier with geometry alone? That is the clean version of the
affinity question, separated from the confound that sank the jointly-trained head (arm 5), where a
randomly-initialised head and the main objective pulled on the encoder simultaneously for one epoch.

INFERENCE ONLY -- fits a 20 GB MIG slice, so it runs on LIGHTGPU while the GPU partition is busy.
Verified by job 48314, which did the same forward pass and hook on LIGHTGPU in 2:35.

    python extract_embeddings.py --ckpt <ckpt> --start 20000 --events 50 --out <dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

EXP_DIR = Path("/home/xucapfwi/hepattn/src/hepattn/experiments/colliderml")
sys.path.insert(0, str(EXP_DIR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--events", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from eval.dump import build_dataset  # noqa: E402
    from hepattn.experiments.colliderml.model import ColliderMLModel  # noqa: E402

    ckpt = Path(args.ckpt)
    model = ColliderMLModel.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval().cuda()

    captured: dict[str, torch.Tensor] = {}
    model.model.encoder.register_forward_hook(lambda m, i, o: captured.__setitem__("embed", o.detach()))

    dataset = build_dataset(ckpt.parent.parent / "config.yaml", args.start, args.events)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        device_inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.model(device_inputs)
        embed = captured["embed"].float().squeeze(0).cpu().numpy()

        xyz = np.column_stack([inputs[f"calohit_{c}"].squeeze(0).numpy() for c in ("x", "y", "z")])
        if embed.shape[0] != xyz.shape[0]:
            print(f"  ! event {i}: {embed.shape[0]} embedding rows against {xyz.shape[0]} cells; skipped")
            continue

        valid = targets["particle_calohit_valid"].squeeze(0).numpy()
        owner = np.where(valid.any(axis=0), valid.argmax(axis=0), -1) if valid.size else np.full(len(xyz), -1)
        energy = inputs["calohit_energy"].squeeze(0).numpy() if "calohit_energy" in inputs else np.zeros(len(xyz))

        # float16 for the embeddings: 22k cells x 256 dims is 11 MB per event at half precision
        # against 22 MB at single, and cosine similarity does not need the extra mantissa.
        np.savez_compressed(
            out / f"event_{args.start + i:06d}.npz",
            embed=embed.astype(np.float16), xyz=xyz.astype(np.float32),
            owner=owner.astype(np.int32), energy=energy.astype(np.float32),
        )
        written += 1

    print(f"wrote {written} events to {out}")


if __name__ == "__main__":
    main()
