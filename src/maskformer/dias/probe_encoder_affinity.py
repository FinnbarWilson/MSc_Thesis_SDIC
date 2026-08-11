"""Does the encoder already know which cells share a particle?

THE QUESTION
------------
The mask head decides membership with an independent sigmoid per (query, cell), so it can express
"this cell resembles that query" and cannot express "this cell is connected to a cell that does".
That is the structural gap behind every measurement in `docs/HIGH_ENERGY_STATUS.md`: a
fixed ~6-cell aperture whatever the shower's size, and 48.6% of assigned cells ending up in the
wrong cluster after growth, 82.1% of them cells with a single unambiguous owner.

But the encoder is four layers of attention over the cells. It may already build representations in
which two cells of one shower are close, even though the mask head has no way to use that. If so,
the information is already in the trained model and the fix is a way to read it -- not a retrain.

WHAT THIS MEASURES
------------------
For pairs of nearby cells, the cosine similarity of their post-encoder embeddings, split by whether
they belong to the same truth particle. Reported as an AUC: 0.5 means the embedding says nothing
about shared ownership, high values mean the encoder has learned exactly the relation the mask head
cannot express.

Pairs are restricted to cells within `--radius` of each other, which is the honest comparison --
distant cells are trivially different particles, and including them would inflate the AUC while
telling us nothing about the decision that actually matters (which of the two showers reaching this
cell owns it). The geometric baseline AUC over the same pairs is printed alongside, so the question
is whether the embedding beats plain distance rather than whether it beats chance.

WHAT FOLLOWS FROM THE ANSWER
----------------------------
* Clearly above the geometric baseline -> affinity-driven chaining is available with NO retraining:
  replace `chain.py`'s nearest-neighbour tie-break with nearest-in-embedding, after storing the
  embeddings in the event store.
* Level with geometry -> the encoder has not learned the relation either, and getting it needs a
  training signal that asks for it: an auxiliary cell-cell affinity head, which is a small addition
  to the task list rather than a new architecture.

Either answer is worth having, and neither requires committing to a new model.

    sbatch --partition=GPU --gres=gpu:a100:2 --cpus-per-task=8 --mem=64G --time=00:40:00 \
      --wrap "..."   # see dias/README.md; or run through probe_encoder_affinity.sh
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
    ap.add_argument("--events", type=int, default=20)
    ap.add_argument("--start", type=int, default=20250)
    ap.add_argument("--radius", type=float, default=0.06, help="metres; pairs closer than this")
    ap.add_argument("--max-pairs", type=int, default=400_000)
    args = ap.parse_args()

    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score

    from eval.dump import build_dataset  # noqa: E402
    from hepattn.experiments.colliderml.model import ColliderMLModel  # noqa: E402

    ckpt = Path(args.ckpt)
    run_config = ckpt.parent.parent / "config.yaml"
    model = ColliderMLModel.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval().cuda()

    # The encoder's output is `x["key_embed"]` inside MaskFormer.forward, which is not returned.
    # A forward hook is the least invasive way to reach it: no change to the model, and no change
    # to eval/dump.py, which stays byte-identical to what produced the reported stores.
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["embed"] = output.detach()

    model.model.encoder.register_forward_hook(hook)

    # THE PROJECTION IS WHERE THE TRAINED STRUCTURE LIVES, and measuring only the raw encoder output
    # would have reported a false negative for the affinity arm. ConstituentAffinityTask trains the
    # cosine similarity of a 32-dim PROJECTION of the encoder embedding, not of the 256-dim
    # embedding itself, so a model that has learned the relation perfectly in that subspace can look
    # unchanged in the raw space. Both are reported: the raw number is comparable to the
    # pre-affinity baseline, the projected one is the trained quantity and the one a downstream
    # chainer would actually use.
    affinity_task = None
    for task in getattr(model.model, "encoder_tasks", []) or []:
        if hasattr(task, "project") and hasattr(task, "log_scale"):
            affinity_task = task
            print(f"affinity head found: {task.name} -> projected space reported alongside raw")
            break
    if affinity_task is None:
        print("no affinity head on this checkpoint; reporting the raw encoder space only")

    dataset = build_dataset(run_config, args.start, args.events)
    rng = np.random.default_rng(0)
    emb_scores, geo_scores, proj_scores, labels = [], [], [], []

    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        device_inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.model(device_inputs)
        embed = captured["embed"].float().squeeze(0).cpu().numpy()

        x = inputs["calohit_x"].squeeze(0).numpy()
        y = inputs["calohit_y"].squeeze(0).numpy()
        z = inputs["calohit_z"].squeeze(0).numpy()
        xyz = np.column_stack([x, y, z])
        if embed.shape[0] != xyz.shape[0]:
            print(f"  ! event {i}: embedding has {embed.shape[0]} rows against {xyz.shape[0]} cells; skipping")
            continue

        # Truth ownership per cell: the particle with the largest share, matching the exclusive
        # partition everything else in this repository is scored against.
        valid = targets["particle_calohit_valid"].squeeze(0).numpy()
        if valid.size == 0:
            continue
        owner = np.where(valid.any(axis=0), valid.argmax(axis=0), -1)

        pairs = np.array(list(cKDTree(xyz).query_pairs(args.radius)), dtype=np.int64)
        if pairs.size == 0:
            continue
        keep = (owner[pairs[:, 0]] >= 0) & (owner[pairs[:, 1]] >= 0)
        pairs = pairs[keep]
        if pairs.shape[0] > args.max_pairs // max(len(dataset), 1):
            pairs = pairs[rng.choice(pairs.shape[0], args.max_pairs // max(len(dataset), 1), replace=False)]
        if pairs.size == 0:
            continue

        a, b = pairs[:, 0], pairs[:, 1]

        def cosine(mat, i, j):
            u, v = mat[i], mat[j]
            return (u * v).sum(1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-9)

        emb_scores.append(cosine(embed, a, b))
        if affinity_task is not None:
            # `zproj`, not `z`: `z` is the cell coordinate a few lines above, and reusing the name
            # here shadowed it. Harmless as written because xyz is built first, but one reordering
            # away from silently correlating the projection against the wrong geometry.
            with torch.no_grad():
                zproj = affinity_task.project(captured["embed"].float())
                zproj = torch.nn.functional.normalize(zproj, dim=-1).squeeze(0).cpu().numpy()
            proj_scores.append(cosine(zproj, a, b))
        geo_scores.append(-np.linalg.norm(xyz[a] - xyz[b], axis=1))  # closer = more likely same
        labels.append((owner[a] == owner[b]).astype(int))

    if not labels:
        raise SystemExit("no usable pairs; check the checkpoint and the event window")

    emb = np.concatenate(emb_scores)
    geo = np.concatenate(geo_scores)
    lab = np.concatenate(labels)
    print(f"\npairs: {len(lab):,}  within {args.radius} m  |  same particle: {lab.mean():.1%}")
    print(f"  AUC, encoder embedding cosine : {roc_auc_score(lab, emb):.3f}")
    if proj_scores:
        proj = np.concatenate(proj_scores)
        print(f"  AUC, AFFINITY-PROJECTED cosine: {roc_auc_score(lab, proj):.3f}   <- the trained quantity")
    print(f"  AUC, plain 3D distance        : {roc_auc_score(lab, geo):.3f}   <- the bar to beat")
    print("\n  baseline, checkpoint without the affinity head: embedding 0.683, distance 0.661")
    print("\nThe embedding is only useful if it beats the distance baseline: chaining already uses")
    print("distance, so an embedding that merely reproduces it adds nothing -- which is exactly how")
    print("the learned per-cell attribution model failed (it rediscovered proximity).")


if __name__ == "__main__":
    main()
