"""Is the mask head predicting anything, or has it collapsed to the target prior?

    python eval/diagnose_mask.py <run_dir> [--events 2] [--ckpt <path>]

WHY THIS IS NOT A METRIC IN model.py

`p{wp}_calohit_eff` answers "did the model reconstruct the particle", which is a strict working
point: it needs >=50% of a particle's cells correctly assigned AND the query accepted by
`flow_valid` AND a non-empty predicted mask. When it reads 0 that is compatible with two very
different states -- a model that is learning but not yet accurate, and a model whose mask head has
collapsed so that every predicted mask is empty and the metric is 0 by construction. Those need
opposite responses, and the efficiency alone cannot tell them apart.

This separates them by looking at the raw sigmoid output instead of the thresholded one. The
signature of collapse is unmistakable: the mean predicted probability equals the positive rate of
the target, i.e. the head has learned the marginal and nothing else, and no cell anywhere exceeds
the 0.5 prediction threshold. Measured on two pu200 runs:

    dice 5 + focal 20, 7 epochs   max prob 0.0000   mean == prior   ->  total collapse
    dice 1 + bce 1,    1500 steps max prob 0.0995   mean == prior   ->  partial collapse
                                                                        (nothing above 0.1)

READ `max prob` FIRST. Anything above 0.5 means the head is making real predictions and a zero
efficiency is an accuracy problem. Everything below ~0.1 with mean ~= prior means it is not, and no
amount of further training will fix it -- that is a converged degenerate solution, not an
undertrained one. At initialisation logits are ~0 so probabilities start near 0.5; a collapsed model
has actively driven every one of them down.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.dump import build_dataset  # noqa: E402
from hepattn.experiments.colliderml.model import ColliderMLModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a logs/ColliderML_Calo_Clustering_* directory")
    parser.add_argument("--ckpt", type=Path, default=None, help="defaults to the newest ckpt in <run_dir>/ckpts")
    parser.add_argument("--events", type=int, default=2)
    parser.add_argument("--start-event", type=int, default=6000, help="inside the val window, i.e. not trained on")
    parser.add_argument("--device", default="cuda")
    # A cap, so this can be run against a checkpoint WHILE a training job holds the rest of the card.
    parser.add_argument("--memory-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if args.device.startswith("cuda") and args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    config = args.run_dir / "config.yaml"
    ckpt = args.ckpt or sorted((args.run_dir / "ckpts").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)[-1]
    print(f"checkpoint : {ckpt.name}")

    dataset = build_dataset(config, args.start_event, args.events)
    model = ColliderMLModel.load_from_checkpoint(ckpt, map_location=args.device).eval().to(args.device)
    object_task = next(t for t in model.model.tasks if t.name == "flow_valid")
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=2)

    # bf16 autocast is mandatory outside Lightning: FlashAttention refuses fp32.
    with torch.no_grad(), torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16):
        for index, (inputs, targets) in enumerate(loader):
            outputs = model.model({k: v.to(args.device) for k, v in inputs.items()})["final"]
            probs = outputs["flow_calohit_assignment"]["flow_calohit_logit"][0].float().sigmoid()
            valid_prob = object_task.predict(outputs["flow_valid"])["flow_valid_prob"][0].float()

            true_valid = targets["particle_valid"][0]
            true_mask = targets["particle_calohit_valid"][0]
            prior = true_mask.float().mean().item()

            print(f"--- event {index}: {probs.shape[0]} queries x {probs.shape[1]} hits")
            print(f"  mask prob : max {probs.max():.4f}  mean {probs.mean():.6f}  (target prior {prior:.2e})")
            print(f"  cells >0.5: {(probs >= 0.5).sum().item()}   >0.3: {(probs >= 0.3).sum().item()}   >0.1: {(probs >= 0.1).sum().item()}")
            print(f"  queries with >=1 cell >0.5: {((probs >= 0.5).sum(-1) > 0).sum().item()} of {probs.shape[0]}")
            print(f"  flow_valid: max {valid_prob.max():.4f} mean {valid_prob.mean():.4f}  above 0.2: {(valid_prob >= 0.2).sum().item()}")
            print(f"  truth     : {true_valid.sum().item()} particles, mean {true_mask.sum(-1).float()[true_valid].mean():.1f} cells each")

            collapsed = probs.max().item() < 0.5
            near_prior = abs(probs.mean().item() - prior) < 0.5 * prior
            print(f"  VERDICT   : {'COLLAPSED to prior' if collapsed and near_prior else 'collapsed (no cell above threshold)' if collapsed else 'PREDICTING -- efficiency is an accuracy question'}")


if __name__ == "__main__":
    main()
