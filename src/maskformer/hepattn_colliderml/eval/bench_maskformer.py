"""Time MaskFormer's clustering of one event, with nothing else in the measurement.

    python -m hepattn.experiments.colliderml.eval.bench_maskformer \
        --store /mnt/ai-datastore/finnbar/eventstore_pu200_v2/ttbar_pu200_7500_8000_v2 --dataset pu200

The counterpart to `scripts.bench_clue`, and the boundary is drawn in the same place:

    cell arrays already in host RAM  ->  an int32 cluster label per cell, back in host RAM

so the host-to-device copy, the forward pass, the thresholding into an exclusive partition
and the copy of the labels back are all inside the clock, while reading the event out of the
store is outside it. No truth is touched, nothing is scored, and no Hungarian matching runs --
the matcher in `model.py` is a training-time device for permuting the loss and has no
inference-time analogue, which is exactly why a timing comparison against CLUE is meaningful
at all.

TWO THINGS THIS DOES NOT DO THE OBVIOUS WAY.

*   Inputs come from `ColliderMLDataset`, not from the event store, even though the store
    holds the very same cells and reading it would be far cheaper. THE ORDER IS THE REASON.
    The store keeps cells in geometry order, `lexsort((phi, layer, subsystem))`, while the
    encoder uses windowed attention over hits sorted by phi -- so a permutation of the input
    decides which hits share a window, and the model is not permutation-invariant. Feeding it
    store-order cells was measured to move 2.9% of cells to a different cluster and to shift
    object probabilities by up to 0.55. The cost of a forward pass would have been the same
    either way, but the clustering being timed would not have been the clustering the thesis
    scored. Decoding is done once, up front, through a DataLoader worker -- see the comment
    in `main` for why the worker is load-bearing -- and the timed region sees only host
    tensors that are already sitting in RAM.

*   Thresholding runs live on the GPU against the raw logits. `EventRecord.maskformer_labels`
    does the same arithmetic in numpy against the stored uint8-quantised CSR, but that store
    is an artefact of this study; a detector-fed system would have the logits in device
    memory and would threshold them there. Because sigmoid is monotonic the probability cut
    is applied as a cut on the logit directly, so no sigmoid is evaluated over the
    [n_queries, n_hits] matrix at all.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hepattn.experiments.colliderml.eval.dump import build_dataset
from hepattn.experiments.colliderml.model import ColliderMLModel

# The event store reader lives in the analysis half of the repository, which is plain numpy
# on purpose. Importing it here is the one place the two halves meet, and it is a read of a
# file format rather than a dependency on any of the scoring code.
#
# Found by walking up rather than counting parents: this file is executed from the copy that
# install_training_env.sh drops into the hepattn checkout, which sits deeper than its source of
# record in src/maskformer/, so a fixed parents[n] is right in one location and wrong in the other.
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

    `ColliderMLDataset` returns a great deal more than the InputNet reads -- detector ids,
    contribution sums, the truth side -- and carrying it into the timed loop would inflate
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
    """How many OTHER processes held the GPU while this ran.

    RECORDED BECAUSE IT ONCE INVALIDATED A WHOLE ROW. CLUE's GPU path is latency-bound -- it
    launches one small kernel per detector layer and waits -- so it degrades under time-slicing
    far worse than a throughput-bound workload does. A first pass at this benchmark ran while
    three other jobs held the card and measured 9.9 s per event; the same code on a quieter
    machine measures 0.38 s. Nothing in the timing distribution revealed it: the three passes
    agreed with each other to 8%, because the contention was steady rather than bursty. The
    only defence is to write down what else was on the device.
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
        raise SystemExit(f"checkpoint {ckpt} does not exist -- pass --ckpt")
    if not run_config.exists():
        raise SystemExit(f"run config {run_config} does not exist -- pass --run-config")

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

    # THROUGH A DATALOADER, exactly as eval/dump.py did, and this is not cosmetic. Indexing
    # the dataset directly in this process gives predictions that differ from the ones in the
    # store -- measured at mean 0.010 and up to 0.41 in object probability, flipping 32 of
    # 1600 queries across the 0.5 decision boundary -- while going through a worker reproduces
    # them BIT-EXACTLY. The tensors are numerically identical either way; what changes is that
    # a worker's arrive through shared memory as freshly allocated buffers, and some kernel in
    # the bf16 stack is sensitive to that. Timing an event whose clustering does not match the
    # scored one would have been the wrong measurement, so the loader stays.
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
                # The sharp test, and an order-free one: the object head's probability per
                # query, against what the dump wrote for the same query. Cell labels can only
                # ever agree up to the +/-0.031 logit half-step of the store's uint8
                # quantisation, so a partition comparison alone would not distinguish a
                # correct input from a slightly wrong one.
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
