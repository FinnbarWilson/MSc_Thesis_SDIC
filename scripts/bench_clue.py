"""Time CLUE's clustering of one event, with nothing else in the measurement.

    python -m scripts.bench_clue --backend "cpu serial"
    python -m scripts.bench_clue --backend "cpu openmp" --dataset pu0

The timed region is exactly ``cluster_event``: cell arrays already in host RAM, to an int32
label per cell. No truth is read and nothing is scored. Reading the event out of the store sits
outside the clock, so every record is materialised before it starts, and
``eval/bench_maskformer.py`` draws the boundary in the same place, which is the only reason the
two numbers can go in one table.

The arguments handed to ``cluster_event`` come from the same config keys `scripts.score` uses.
Writes ``bench_clue_<backend>.json``.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import src.clue.pipeline as pipeline
from src.clue.pipeline import cluster_event
from src.config import DATASETS, active_dataset, results_dir, settings, settings_for, store_expectations, store_path
from src.io.event_store import EventStore

BACKENDS = ("cpu serial", "cpu openmp", "gpu cuda")


def backend_slug(backend: str) -> str:
    return backend.replace(" ", "_")


def check_backend(backend: str) -> None:
    """Refuse to time a backend CLUEstering cannot actually run.

    `run_clue` handles a missing backend by printing "CUDA module not found" and returning with
    `cluster_ids` untouched; it does not raise. The conda-forge wheel ships no GPU module, so
    timing `gpu cuda` against a stock install measures a no-op and reports it as a large
    speed-up. See setup/build_clue_cuda.sh.

    Raises:
        SystemExit: if CLUEstering was built without `backend`.
    """
    import CLUEstering as clue

    if backend not in clue.backends:
        msg = (
            f"CLUEstering was built without the {backend!r} backend; it advertises "
            f"{clue.backends}. run_clue would print a warning and silently return unclustered "
            f"points, so this would time nothing. Build it with setup/build_clue_cuda.sh."
        )
        raise SystemExit(msg)


def canonical(label: np.ndarray) -> np.ndarray:
    """Relabel by order of first appearance, so two equal partitions compare equal.

    CLUE numbers its clusters in whatever order it happens to find seeds, and the parallel
    backends find them in a different order from the serial one. Comparing raw ids therefore
    reports ~35% agreement between backends that in fact produced the identical partition.
    """
    out = np.full(label.shape, -1, dtype=np.int64)
    clustered = label >= 0
    if not clustered.any():
        return out
    values, first = np.unique(label[clustered], return_index=True)
    rank = np.empty(values.size, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(values.size)
    out[clustered] = rank[np.searchsorted(values, label[clustered])]
    return out


def verify_against_serial(records, params, kwargs, backend: str) -> None:
    """A faster backend that solves a different problem is not a faster backend.

    Checked rather than assumed because the failure is silent: `run_clue` leaves `cluster_ids`
    untouched when a backend misbehaves, so a wrong answer arrives as a plausible one.
    """
    for record in records:
        reference, _ = cluster_event(record, params, backend="cpu serial", **kwargs)
        candidate, _ = cluster_event(record, params, backend=backend, **kwargs)
        if not np.array_equal(canonical(reference), canonical(candidate)):
            differing = float((canonical(reference) != canonical(candidate)).mean())
            msg = (
                f"backend {backend!r} disagrees with 'cpu serial' on event {record.sample_id}: "
                f"{differing:.4f} of cells land in a different cluster. Timing it would compare "
                f"two different algorithms."
            )
            raise SystemExit(msg)
    print(f"  verified: {backend!r} reproduces 'cpu serial' partitions on {len(records)} events")


def competing_gpu_processes() -> dict:
    """How many other processes held the GPU while this ran.

    CLUE's GPU path is latency-bound, launching one small kernel per detector layer and waiting,
    so it degrades under time-slicing far worse than a throughput-bound workload does. A first
    pass at this benchmark ran with three other jobs on the card and measured 9.9 s per event
    against 0.38 s on a quiet machine, and nothing in the timing distribution revealed it: the
    three passes agreed to 8%, the contention being steady rather than bursty. Recording what
    else was on the device is the only defence.
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
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=None, choices=BACKENDS, help="defaults to clue.backend")
    parser.add_argument("--dataset", default=None, choices=DATASETS, help="defaults to dataset.active")
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=100, help="events to time")
    parser.add_argument("--warmup", type=int, default=10, help="events run and discarded first")
    parser.add_argument("--repeats", type=int, default=3, help="timed passes over the same events")
    parser.add_argument("--verify", type=int, default=3, help="events to check against the cpu serial partition")
    parser.add_argument("--backend-3d", default=None, choices=BACKENDS,
                        help="send stage 2 (the 3D centroid pass) to a different backend from stage 1")
    args = parser.parse_args()

    # Stage routing by wrapping the pipeline's entry point rather than changing it. Stage 1 is
    # ~160 calls of a few hundred points and stage 2 a handful over tens of thousands, so the
    # device suiting one need not suit the other.
    if args.backend_3d:
        untouched = pipeline._run_clue

        def routed(coord_a, coord_b, weights, params, suffix, coords, backend, depth=None):
            chosen = args.backend_3d if suffix == "3d" else backend
            return untouched(coord_a, coord_b, weights, params, suffix, coords, chosen, depth)

        pipeline._run_clue = routed

    dataset = args.dataset or active_dataset()
    cfg = settings() if dataset == active_dataset() else settings_for(dataset)
    backend = args.backend or cfg["clue"]["backend"]
    check_backend(backend)
    if args.backend_3d:
        check_backend(args.backend_3d)

    store = EventStore(
        args.store or store_path(dataset=dataset),
        expect=store_expectations(dataset=dataset),
    )

    params_path = args.params or (results_dir(dataset=dataset) / "clue_parameters.json")
    tuned = json.loads(params_path.read_text())
    params = {name: entry["parameters"] for name, entry in tuned["subsystems"].items()}
    if tuned.get("dataset") != dataset:
        raise SystemExit(f"{params_path} was tuned on {tuned.get('dataset')!r}, not {dataset!r}")

    # Exactly the call scripts.score makes, from exactly the same config keys.
    kwargs = {
        "subsystems": tuple(cfg["detectors"]),
        "coords": cfg["clue"]["coords"],
        "min_cluster_hits": cfg["metrics"]["min_cluster_hits"],
        "link_radius": cfg["clue"].get("link_radius", 0.0),
    }

    gpu_before = competing_gpu_processes()
    if backend.startswith("gpu") and gpu_before.get("n_other_processes"):
        print(f"  ! {gpu_before['n_other_processes']} other process(es) are using the GPU; "
              f"a latency-bound backend will be slowed by an unknown factor")

    print(f"dataset {dataset}  backend {backend!r}  subsystems tuned: {sorted(params)}")
    print(f"store   {store.root}")

    n_needed = args.warmup + args.limit
    if len(store) < n_needed:
        raise SystemExit(f"store holds {len(store)} events, need {n_needed}")

    # Decode every event up front: an npz read inside the timed region would be charged to
    # CLUE, and the first read of a chunk is far slower than the rest.
    print(f"materialising {n_needed} events ...", flush=True)
    records = [store[i] for i in range(n_needed)]
    warm, timed = records[: args.warmup], records[args.warmup :]

    if backend != "cpu serial" and args.verify:
        verify_against_serial(records[: args.verify], params, kwargs, backend)

    print(f"warm-up over {len(warm)} events ...", flush=True)
    for record in warm:
        cluster_event(record, params, backend=backend, **kwargs)

    samples = []
    for rep in range(args.repeats):
        for record in timed:
            start = time.perf_counter_ns()
            _, n_clusters = cluster_event(record, params, backend=backend, **kwargs)
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
        print(f"  pass {rep + 1}/{args.repeats}: median {median:.1f} ms/event", flush=True)

    out_dir = args.out or results_dir(dataset=dataset)
    suffix = backend_slug(backend) + (f"__3d_{backend_slug(args.backend_3d)}" if args.backend_3d else "")
    out_path = out_dir / f"bench_clue_{suffix}.json"
    out_path.write_text(
        json.dumps(
            {
                "method": "clue",
                "dataset": dataset,
                "backend": backend,
                "backend_3d": args.backend_3d,
                "device": "A100 80GB PCIe" if backend.startswith("gpu") else cpu_model(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "store": str(store.root),
                "params": str(params_path),
                "subsystems_tuned": sorted(params),
                "clue_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
                "warmup": args.warmup,
                "repeats": args.repeats,
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
