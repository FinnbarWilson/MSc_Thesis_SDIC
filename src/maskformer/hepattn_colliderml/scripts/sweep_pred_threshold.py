"""Sweep the mask prediction threshold on a trained clustering checkpoint.

`ObjectHitMaskTask.pred_threshold` (default 0.5) decides how confident the model must be before it
claims a hit for a cluster. Every configuration measured so far assigns hits too conservatively —
purity has consistently come out well above efficiency (0.909 vs 0.685 even when deliberately
overfitting) — which means there is efficiency to buy by lowering it, and purity to spare.

Nothing here retrains anything: one forward pass per event produces the mask logits, and every
threshold is then evaluated on those same logits. The reported numbers use exactly the formula in
ColliderMLModel.log_custom_metrics, so they are directly comparable with the val/p0.5_calohit_eff
you see in Comet.

Note that the threshold ALSO feeds the decoder's mask attention when mask_attention_threshold is
left at its default, so the value chosen here should be set in the config and confirmed with a real
validation run before being reported. This sweep isolates the output thresholding only.

Usage (inside the container, from the experiment directory):
    python scripts/sweep_pred_threshold.py <clustering ckpt> [--num-events 100] [--config <cfg>]
"""

import argparse
from pathlib import Path

import torch
import yaml

from hepattn.experiments.colliderml.data import ColliderMLDataModule
from hepattn.experiments.colliderml.model import ColliderMLModel

# The low end (< 0.5) is here for the earlier over-conservative configs the docstring describes.
# The 4-epoch num_train=20000 run (job 48169) inverted that: purity 0.270 BELOW efficiency 0.389 at
# p0.5. Extended to 0.95 to probe the high side; the answer was that it does not matter. F1 is flat
# at 0.304-0.322 across the whole 0.2-0.9 range (best 0.322 at 0.70 vs 0.319 at the 0.50 default),
# i.e. threshold choice buys nothing on that checkpoint - it only slides along the eff/pur trade.
# Diagnostic that actually mattered: hits/flow at 0.5 is 19.6 against 19.0 true hits per particle,
# so the masks are the right SIZE and simply contain the wrong hits. A flat F1 plateau plus correct
# average mask size means the logit RANKING is the problem, and no cut point can fix ranking.
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
WORKING_POINTS = (0.5, 1.0)

# The object head's own working point. ObjectClassificationTask.predict takes a `threshold`
# argument but never uses it — validity comes from an argmax over [p_valid, 1 - p_valid], which is
# exactly `flow_valid_prob >= 0.5`. So the effective object threshold is hardcoded at 0.5, and
# thresholding flow_valid_prob directly is the generalisation of it.
OBJ_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
OBJ_SWEEP_MASK_THRESHOLDS = (0.05, 0.2, 0.5)


def evaluate(masks: torch.Tensor, flow_valid: torch.Tensor, true_masks: torch.Tensor, true_valid: torch.Tensor):
    """Per-event efficiency/purity, mirroring ColliderMLModel.log_custom_metrics.

    Also decomposes efficiency into its two independent failure modes, because the headline number
    conflates them:

      * `eff`      the reported metric. A true particle counts only if the query matched to it
                   recovered >= wp of its hits AND the object head declared that query a real
                   object. Two ways to fail, one number.
      * `eff_mask` the same hit requirement with the object-head veto removed — what the masks
                   achieve on their own, i.e. the ceiling if the object head were perfect.
      * `obj_eff`  the fraction of true particles whose matched query the object head accepted.
                   This is a hard ceiling on `eff`: no matter how good the masks get, efficiency
                   cannot exceed it.

    If eff_mask is far above eff, the object head is the bottleneck and no amount of mask tuning
    will help. If they are close, the masks are the bottleneck.
    """
    pred_valid = flow_valid & (masks.sum(-1) > 0)
    true_ok = true_valid & (true_masks.sum(-1) > 0)

    # Keep the ungated masks. Gating them by pred_valid first would make the object-head-free
    # measurement below meaningless: a rejected query has an all-False mask, so its overlap is 0 and
    # it fails the hit requirement anyway — the two numbers would come out identical by construction
    # rather than because the object head is harmless.
    raw_masks = masks
    masks = masks & pred_valid.unsqueeze(-1)
    true_masks = true_masks & true_ok.unsqueeze(-1)

    hit_tp = (masks & true_masks).sum(-1)
    hit_tp_raw = (raw_masks & true_masks).sum(-1)
    hit_p = masks.sum(-1)
    hit_t = true_masks.sum(-1)
    both_valid = true_ok & pred_valid
    n_true = true_ok.float().sum(-1) + 1e-8

    out = {}
    for wp in WORKING_POINTS:
        hit_ok = (hit_tp / (hit_t + 1e-8)) >= wp
        effs = hit_ok & both_valid
        purs = ((hit_tp / (hit_p + 1e-8)) >= wp) & both_valid
        out[f"eff{wp}"] = (effs.float().sum(-1) / n_true).mean().item()
        out[f"eff_mask{wp}"] = ((((hit_tp_raw / (hit_t + 1e-8)) >= wp) & true_ok).float().sum(-1) / n_true).mean().item()
        # Purity is already cluster-based: the denominator counts PREDICTED clusters.
        out[f"pur{wp}"] = (purs.float().sum(-1) / (pred_valid.float().sum(-1) + 1e-8)).mean().item()

    out["obj_eff"] = (both_valid.float().sum(-1) / n_true).mean().item()
    out["hits_per_flow"] = hit_p[pred_valid].float().mean().item() if bool(pred_valid.any()) else 0.0
    out["hits_per_part"] = hit_t[true_ok].float().mean().item() if bool(true_ok.any()) else 0.0
    out["num_flows"] = pred_valid.sum(-1).float().mean().item()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt", type=Path, help="a trained CLUSTERING checkpoint (not the hit filter)")
    parser.add_argument("--config", type=Path, default=None, help="defaults to config.yaml beside the checkpoint")
    parser.add_argument("--num-events", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = args.config or args.ckpt.parent.parent / "config.yaml"
    with config_path.open() as f:
        data_config = yaml.safe_load(f)["data"]
    data_config["num_val"] = args.num_events
    data_config.setdefault("num_test", args.num_events)

    datamodule = ColliderMLDataModule(**data_config)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()

    model = ColliderMLModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval().to(args.device)

    totals: dict[float, list[dict]] = {t: [] for t in THRESHOLDS}
    grid: dict[tuple[float, float], list[dict]] = {(m, o): [] for m in OBJ_SWEEP_MASK_THRESHOLDS for o in OBJ_THRESHOLDS}
    n_seen = 0

    # bf16 autocast, matching the trainer's `precision: bf16-mixed`. Without it the model runs in
    # fp32 and FlashAttention refuses outright ("only support fp16 and bf16 data type"), since
    # Lightning is what normally supplies the autocast context.
    with torch.no_grad(), torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16):
        for inputs, targets in loader:
            if n_seen >= args.num_events:
                break
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            targets = {k: v.to(args.device) for k, v in targets.items()}

            outputs = model.model(inputs)
            # Runs the Hungarian matcher and returns targets permuted onto the query ordering, which
            # is what makes the index-wise comparison below valid.
            outputs, targets, _ = model.model.loss(outputs, targets)
            preds = model.model.predict(outputs)

            logits = outputs["final"]["flow_calohit_assignment"]["flow_calohit_logit"]
            probs = logits.sigmoid()
            flow_valid = preds["final"]["flow_valid"]["flow_valid"]
            true_masks = targets["particle_calohit_valid"]
            true_valid = targets["particle_valid"]

            for threshold in THRESHOLDS:
                totals[threshold].append(evaluate(probs >= threshold, flow_valid, true_masks, true_valid))

            valid_prob = preds["final"]["flow_valid"]["flow_valid_prob"]
            for mask_t in OBJ_SWEEP_MASK_THRESHOLDS:
                masks_t = probs >= mask_t
                for obj_t in OBJ_THRESHOLDS:
                    grid[mask_t, obj_t].append(evaluate(masks_t, valid_prob >= obj_t, true_masks, true_valid))
            n_seen += 1

    print(f"\n{n_seen} events from {config_path}")
    true_size = sum(r["hits_per_part"] for r in totals[THRESHOLDS[0]]) / max(n_seen, 1)
    print(f"true hits per particle: {true_size:.1f}\n")
    header = (
        f"  {'thresh':>7} {'p0.5 eff':>9} {'p0.5 pur':>9} {'p0.5 F1':>8} "
        f"{'eff(mask)':>10} {'obj eff':>8} {'p1.0 eff':>9} {'hits/flow':>10} {'flows':>7}"
    )
    print(header)
    best = None
    means = {}
    for threshold in THRESHOLDS:
        rows = totals[threshold]
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        means[threshold] = mean
        f1 = 2 * mean["eff0.5"] * mean["pur0.5"] / max(mean["eff0.5"] + mean["pur0.5"], 1e-9)
        print(
            f"  {threshold:>7.2f} {mean['eff0.5']:>9.3f} {mean['pur0.5']:>9.3f} {f1:>8.3f} "
            f"{mean['eff_mask0.5']:>10.3f} {mean['obj_eff']:>8.3f} "
            f"{mean['eff1.0']:>9.3f} {mean['hits_per_flow']:>10.1f} {mean['num_flows']:>7.0f}"
        )
        if best is None or f1 > best[1]:
            best = (threshold, f1)

    print("\nBOTH working points together (mask threshold x object-head threshold)")
    print("  The object head is the second gate: a true particle counts only if its query is also")
    print("  accepted as a real object, so its threshold is an independent free knob.")
    print(f"  {'mask':>6} {'obj':>6} {'p0.5 eff':>9} {'p0.5 pur':>9} {'p0.5 F1':>8} {'flows':>7}")
    for mask_t in OBJ_SWEEP_MASK_THRESHOLDS:
        for obj_t in OBJ_THRESHOLDS:
            rows = grid[mask_t, obj_t]
            m = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            f1 = 2 * m["eff0.5"] * m["pur0.5"] / max(m["eff0.5"] + m["pur0.5"], 1e-9)
            print(
                f"  {mask_t:>6.2f} {obj_t:>6.2f} {m['eff0.5']:>9.3f} {m['pur0.5']:>9.3f} "
                f"{f1:>8.3f} {m['num_flows']:>7.0f}"
            )

    top = means[best[0]]
    print(f"\nBest p0.5 F1 at threshold {best[0]:.2f} (F1 {best[1]:.3f}).")
    print("Read hits/flow against the true hits per particle above: a threshold that inflates")
    print("efficiency while blowing up hits/flow is painting blobs, not clustering better.")
    print(
        f"\nAt that threshold the object head accepts {top['obj_eff']:.3f} of true particles, so "
        f"efficiency cannot exceed that.\nMasks alone would give {top['eff_mask0.5']:.3f} against "
        f"the reported {top['eff0.5']:.3f} — "
        + (
            "the OBJECT HEAD is the bottleneck, not the clustering."
            if top["eff_mask0.5"] - top["eff0.5"] > 0.05
            else "the MASKS are the bottleneck, so the object head is not what to fix."
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
