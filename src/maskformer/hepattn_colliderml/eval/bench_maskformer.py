"""Time MaskFormer's clustering of one event, with nothing else in the measurement.

    python -m hepattn.experiments.colliderml.eval.bench_maskformer \
        --store <event store> --dataset pu200

The counterpart to `scripts.bench_clue`, with the boundary drawn in the same place: cell arrays
already in host RAM, to an int32 label per cell back in host RAM. The host-to-device copy, the
forward pass, the thresholding into an exclusive partition and the copy back are inside the
clock; reading the event out of the store is outside it. No truth is touched and no Hungarian
matching runs, that matcher being a training-time device with no inference-time analogue.

Inputs come from `ColliderMLDataset` rather than from the event store, even though the store
holds the same cells and is cheaper to read, because the order differs: the store keeps cells in
geometry order while the encoder uses windowed attention over hits sorted by phi, so a
permutation decides which hits share a window and the model is not permutation-invariant.
Feeding store-order cells was measured to move 2.9% of cells to a different cluster. Decoding
happens once, up front, so the timed region sees only host tensors already in RAM.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hepattn.experiments.colliderml.eval.dump import build_dataset
from hepattn.experiments.colliderml.model import ColliderMLModel
from torch.utils.data import DataLoader


# Found by walking up rather than counting parents: this file runs from the copy
# install_training_env.sh drops into the hepattn checkout, which sits deeper than its source of
# record, so a fixed parents[n] would be right in one location and wrong in the other.
def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src" / "io" / "event_store.py").exists() and (candidate / "config" / "experiment.yaml").exists():
            return candidate
    msg = "cannot locate the thesis repository root from " + str(Path(__file__).resolve())
    raise SystemExit(msg)


REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))
from src.io.event_store import EventStore  # noqa: E402

INPUT_FIELDS = ("x", "y", "z", "r", "eta", "phi", "log_energy", "valid")


def host_inputs(sample_inputs: dict) -> dict[str, torch.Tensor]:
    """The tensors the model consumes, pinned to the host and nothing else.

    `ColliderMLDataset` returns a great deal more than the InputNet reads: detector ids,
    contribution sums, the truth side. Carrying it into the timed loop would inflate
    both the host-to-device copy and the memory footprint of holding 100 events at once.
    """
    return {f"calohit_{f}": sample_inputs[f"calohit_{f}"] for f in INPUT_FIELDS}


def store_permutation(loader_xyz: np.ndarray, store_xyz: np.ndarray) -> np.ndarray:
    """Index array mapping loader-order cells onto store-order cells.

    Needed only by `--check`. The two orders hold the same cells, so they are matched on
    position: both sides are lexsorted on (z, y, x), which pairs them off exactly because the
    coordinates are the identical float32 values on both sides of the dump.
    """
    a = np.lexsort((loader_xyz[:, 0], loader_xyz[:, 1], loader_xyz[:, 2]))
    b = np.lexsort((store_xyz[:, 0], store_xyz[:, 1], store_xyz[:, 2]))
    perm = np.empty(a.size, dtype=np.int64)
    perm[a] = b
    return perm


def labels_from_logits(
    mask_logit: torch.Tensor,
    valid_prob: torch.Tensor,
    mask_threshold: float,
    object_threshold: float,
    min_cluster_hits: int,
) -> tuple[torch.Tensor, int]:
    """Exclusive cell -> cluster labels, entirely on the device.

    Same rule as `EventRecord.maskformer_labels`: a cell joins the accepted query that claims
    it most strongly, and a query is accepted only if the object head believes in it. The
    probability cut is applied to the logit, since sigmoid is monotonic and evaluating it
    over a [n_queries, n_hits] matrix would be pure waste.
    """
    logit_floor = float(np.log(mask_threshold / (1.0 - mask_threshold)))
    accepted = valid_prob >= object_threshold
    if not bool(accepted.any()):
        return torch.full((mask_logit.shape[1],), -1, dtype=torch.int32, device=mask_logit.device), 0

    scores = mask_logit.masked_fill(~accepted.unsqueeze(1), float("-inf"))
    best_score, best_query = scores.max(dim=0)
    label = torch.where(best_score >= logit_floor, best_query, torch.full_like(best_query, -1))

    if min_cluster_hits > 1:
        counts = torch.bincount(label[label >= 0], minlength=mask_logit.shape[0])
        too_small = counts < min_cluster_hits
        label = torch.where((label >= 0) & too_small[label.clamp_min(0)], torch.full_like(label, -1), label)

    claimed = label >= 0
    if not bool(claimed.any()):
        return torch.full_like(label, -1, dtype=torch.int32), 0
    used, compact = torch.unique(label[claimed], return_inverse=True)
    out = torch.full_like(label, -1, dtype=torch.int32)
    out[claimed] = compact.to(torch.int32)
    return out, int(used.numel())


def competing_gpu_processes() -> dict:
    """How many other processes held the GPU while this ran.

    A benchmark sharing the card measures the sharing as much as the code, and contention that is
    steady rather than bursty does not show up as spread between passes. Recording what else was
    on the device is the only defence.
    """
    import os
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
    except Exception:
        return {"probed": False}
    mine = os.getpid()
    others = [line for line in out.strip().splitlines() if line.strip() and int(line.split(",")[0]) != mine]
    return {"probed": True, "n_other_processes": len(others), "detail": others}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="event store directory")
    parser.add_argument("--dataset", required=True, choices=["pu0", "pu200"], help="stamped into the output")
    parser.add_argument("--ckpt", type=Path, default=None, help="defaults to the checkpoint the store names")
    parser.add_argument("--run-config", type=Path, default=None, help="defaults to the run config the store names")
    parser.add_argument("--out", type=Path, default=None, help="defaults to results/<dataset>/")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8, help="loader workers; decode only, outside the clock")
    parser.add_argument("--mask-threshold", type=float, default=0.05)
    parser.add_argument("--object-threshold", type=float, default=0.5)
    parser.add_argument("--min-cluster-hits", type=int, default=1)
    parser.add_argument("--check", type=int, default=10, help="events to validate against the stored labels")
    args = parser.parse_args()

    # No contract check: this script reads the store only for provenance and for --check.
    # EventStore still validates the format.
    store = EventStore(args.store, expect=None)
    ckpt = args.ckpt or Path(store.meta["maskformer"]["checkpoint"])
    run_config = args.run_config or Path(store.meta["maskformer"]["run_config"])
    if not ckpt.exists():
        raise SystemExit(f"checkpoint {ckpt} does not exist; pass --ckpt")
    if not run_config.exists():
        raise SystemExit(f"run config {run_config} does not exist; pass --run-config")

    gpu_before = competing_gpu_processes()
    if gpu_before.get("n_other_processes"):
        print(f"  ! {gpu_before['n_other_processes']} other process(es) are using the GPU")

    print(f"dataset {args.dataset}  store {store.root}")
    print(f"checkpoint {ckpt.name}")

    model = ColliderMLModel.load_from_checkpoint(ckpt, map_location=args.device)
    model.eval().to(args.device)
    object_task = next(t for t in model.model.tasks if t.name == "flow_valid")
    n_queries = int(getattr(model.model.decoder, "_num_queries", -1))
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"  {n_parameters / 1e6:.1f} M parameters, {n_queries} queries")

    n_needed = args.warmup + args.limit
    if len(store) < n_needed:
        raise SystemExit(f"store holds {len(store)} events, need {n_needed}")

    # The same window the store was dumped over, so event i here is event i there.
    window = store.meta["event_window"]
    dataset = build_dataset(run_config, int(window["start_event"]), n_needed)
    # Truth associations are the expensive half of a load and are pure waste here: nothing in
    # this script reads targets. Disabling it leaves the input side untouched.
    dataset.build_calohit_associations = False

    # Through a DataLoader, as eval/dump.py did, and this is not cosmetic: indexing the dataset
    # directly in this process shifts object probabilities enough to flip queries across the
    # decision boundary, while going through a worker reproduces the store bit-exactly. Timing an
    # event whose clustering does not match the scored one would be the wrong measurement.
    print(f"decoding {n_needed} events through ColliderMLDataset ...", flush=True)
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    hosts, records = [], []
    for i, (sample_inputs, _) in enumerate(loader):
        if i >= n_needed:
            break
        hosts.append(host_inputs(sample_inputs))
        records.append(store[i])

    def infer(host: dict[str, torch.Tensor], return_valid: bool = False):
        device_inputs = {k: v.to(args.device, non_blocking=True) for k, v in host.items()}
        outputs = model.model(device_inputs)
        final = outputs["final"]
        mask_logit = final["flow_calohit_assignment"]["flow_calohit_logit"][0].float()
        valid_prob = object_task.predict(final["flow_valid"])["flow_valid_prob"][0].float()
        label, n = labels_from_logits(
            mask_logit, valid_prob, args.mask_threshold, args.object_threshold, args.min_cluster_hits
        )
        if return_valid:
            return label.cpu().numpy(), n, valid_prob.cpu().numpy().astype(np.float64)
        return label.cpu().numpy(), n

    # bf16 autocast is mandatory, not an optimisation: outside Lightning the model is fp32 and
    # FlashAttention refuses it ("only support fp16 and bf16 data type").
    autocast = torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16)

    agreement = valid_prob_delta = None
    if args.check:
        n_same = n_total = 0
        deltas = []
        with torch.no_grad(), autocast:
            for record, host in zip(records[: args.check], hosts[: args.check]):
                live, _, live_valid = infer(host, return_valid=True)
                # The sharp, order-free test: object probability per query against what the
                # dump wrote. Cell labels agree only to the store's uint8 half-step, so a
                # partition comparison alone could not tell a correct input from a near one.
                q = record.mf_query_index.astype(np.int64)
                deltas.append(float(np.abs(live_valid[q] - record.mf_valid_prob.astype(np.float64)).max()))

                stored, _ = record.maskformer_labels(args.mask_threshold, args.object_threshold, args.min_cluster_hits)
                xyz = np.stack([host["calohit_x"][0], host["calohit_y"][0], host["calohit_z"][0]], axis=1).astype(np.float32)
                perm = store_permutation(xyz, np.stack([record.x, record.y, record.z], axis=1).astype(np.float32))
                # Compare which cells are claimed, not which id they carry: query indices and
                # stored-CSR row indices number the same clusters differently.
                in_store_order = np.empty_like(live)
                in_store_order[perm] = live
                n_same += int(((in_store_order < 0) == (stored < 0)).sum())
                n_total += live.size
        agreement = n_same / n_total
        valid_prob_delta = max(deltas)
        print(f"  valid_prob live-vs-stored: max |delta| = {valid_prob_delta:.2e}")
        print(f"  claimed-cell agreement over {args.check} events: {agreement:.5f}")

    print(f"warm-up over {args.warmup} events ...", flush=True)
    with torch.no_grad(), autocast:
        for host in hosts[: args.warmup]:
            infer(host)
    torch.cuda.synchronize()

    samples = []
    with torch.no_grad(), autocast:
        for rep in range(args.repeats):
            for record, host in zip(records[args.warmup :], hosts[args.warmup :]):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                _, n_clusters = infer(host)
                torch.cuda.synchronize()
                elapsed = time.perf_counter_ns() - start
                samples.append(
                    {
                        "repeat": rep,
                        "sample_id": int(record.sample_id),
                        "n_hits": int(record.n_hits),
                        "n_clusters": int(n_clusters),
                        "elapsed_ns": int(elapsed),
                    }
                )
            median = np.median([s["elapsed_ns"] for s in samples if s["repeat"] == rep]) / 1e6
            print(f"  pass {rep + 1}/{args.repeats}: median {median:.2f} ms/event", flush=True)

    out_dir = args.out or (REPO_ROOT / "results" / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bench_maskformer.json"
    out_path.write_text(
        json.dumps(
            {
                "method": "maskformer",
                "dataset": args.dataset,
                "backend": "cuda bf16",
                "device": torch.cuda.get_device_name(0),
                "store": str(store.root),
                "checkpoint": str(ckpt),
                "n_queries": n_queries,
                "n_parameters": n_parameters,
                "run_config": str(run_config),
                "mask_threshold": args.mask_threshold,
                "object_threshold": args.object_threshold,
                "min_cluster_hits": args.min_cluster_hits,
                "label_agreement_vs_store": agreement,
                "valid_prob_max_delta_vs_store": valid_prob_delta,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "torch_version": torch.__version__,
                "git_sha": git_sha(),
                "gpu_contention_before": gpu_before,
                "gpu_contention_after": competing_gpu_processes(),
                "samples": samples,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
