# `dias/` — running this repository's config on DIAS, under Slurm

Job scripts for **DIAS** (`dias.hpc.phys.ucl.ac.uk`), the RHEL7 Slurm cluster, where the **pu0**
work runs. The counterpart to [`../ce_ai_1/`](../ce_ai_1/README.md), which is a different machine
entirely — no scheduler, its own A100, and where the **pu200** work runs.

| | DIAS (here) | ce-ai-1 |
|---|---|---|
| pileup | **pu0** | **pu200** |
| scheduling | Slurm: `sbatch`, queueing, walltime caps | run it directly, `nohup` for long runs |
| OS | RHEL7, glibc 2.17 → apptainer container | Ubuntu 24.04, glibc 2.39 → no container |
| python env | pixi env inside the container, from the hepattn checkout | `external/venv-hepattn` |
| GPU | `GPU`: 3 × A100 **80 GB**, one with 844 uncorrected ECC errors<br>`LIGHTGPU`: 6 × ~20 GB MIG instances | 1 × A100 80 GB |
| host | 128 CPUs, 515 GB, **shared** | 64 CPUs, 1.5 TB |
| dataset | `/home/xucapfwi/ColliderML_data` — pu0 only | `/mnt/ai-datastore/finnbar` — pu200 |

Two cluster facts worth having in one place, both checked rather than assumed:

- **The GPU partition's cards are 80 GB.** Job 48244 reported `NVIDIA A100 80GB PCIe, 81920 MiB`,
  MIG disabled. The [cluster documentation](https://uclphysast.github.io/clusters/dias/) says 40 GB
  and is out of date on this point. It is right about LIGHTGPU being MIG instances — which
  `/etc/slurm/gpu_variables.sh` confirms, by mapping `CUDA_VISIBLE_DEVICES` to six MIG UUIDs.
- **`--mem` is not enforced.** `TaskPlugin=task/affinity`, no cgroups, so memory is consumed for
  *scheduling* but nothing kills a job that exceeds it: job 48169 booked 128 GB, used 185 GB, and
  completed. Under-requesting therefore looks like it works while over-booking the node for
  everyone else. The same absence of cgroups means **GPUs are not isolated either** — see below.

## Why these are not in `hepattn_colliderml/slurm/`

Everything under `hepattn_colliderml/` is a verbatim mirror of the hepattn checkout, and
`verify_sync.sh` checks `slurm/calo_clustering.sh` and `slurm/calo_dump_eventstore.sh`
byte-for-byte against it. Editing them here would turn the mirror into a fork — the same reason
`ce_ai_1/env.sh` gives for living outside it. These are mine, so they live here.

The mirrored `slurm/calo_clustering.sh` still works and is the provenance of the reported
checkpoint (job 48169). It is not what to submit now.

```
env.sh                        paths, GPU selection, mirror sync, Comet key, container wrapper
benchmark_calo_clustering.sh  measure throughput; prints the walltime and allocation to ask for
smoke_calo_clustering.sh      ~20 min: does the new objective run and does the loss respond
train_calo_clustering.sh      the real run
```

## What these do that the mirrored script does not

**They run this repository's config, by absolute path.** `main.py` runs from the hepattn checkout,
whose `configs/calo_clustering.yaml` is still the *old Lion configuration* — a relative `--config`
would silently have trained that. `data.py` and `model.py` are imported as
`hepattn.experiments.colliderml.*` and can only come from the checkout, so the scripts re-copy them
from here first (`SYNC=0` opts out). That is the trap `ce_ai_1/env.sh` documents, in its Slurm
form; the copy here is unconditional rather than `cp -u`, because a freshly checked-out file can be
*older* than the stale copy it has to replace.

**They pick a GPU from the allocation, not from what is visible.** There are no cgroups on this
cluster, so `nvidia-smi` inside a job lists all three cards regardless of what Slurm gave it —
verified in job 48244, which asked for two and saw indices 0, 1 and 2. Choosing from the visible
list would cheerfully land the run on another job's card. `env.sh:select_gpu` intersects Slurm's
own allocation variables with the healthy set, and aborts if that intersection is empty.
(`calo_dump_eventstore.sh` chooses from the visible list, on the assumption that a cgroup
renumbers the allocation from 0. That is right on a cluster that isolates devices; not this one.)

**They ask for the resources the job actually uses**, and a walltime the schedule fits — below.

Unchanged: two GPUs requested so one healthy card is guaranteed, one used; apptainer + the pixi
env; `torch.compile` disabled because the bare container has no C compiler for Triton.

## Measured, 2026-08-05 — and the one that matters is the CPU count

Two probes, ten to twelve minutes each of the real configuration at the real `num_train=20000`:

| allocation | rate | GPU util | peak GPU mem | host RSS | epoch | 7 epochs | job |
|---|---|---|---|---|---|---|---|
| 12 CPU / 12 workers | 1.09 ev/s | 16% | 49.2 / 81.9 GB | 131 GB | 5.19 h | 36.3 h | 48244 |
| 32 CPU / 24 workers | **1.94 ev/s** | 30% | 49.2 / 81.9 GB | 255 GB | 2.91 h | **20.4 h** | 48245 |

**This run is input-bound, not GPU-bound.** At the mirrored script's 12 CPUs the A100 sits idle 84%
of the time waiting for the dataloader, and the extra CPUs — which cost nothing on a 128-core node —
take **nine hours** off the run. That is the answer to "are we asking for the right thing": the
walltime was never the lever, the CPU count was.

It is *still* input-bound at 30% utilisation, so 32 CPUs is not the ceiling — it is where the
ceiling stops being CPUs and becomes host RAM. `data.py` caches 8 decoded row groups **per worker**
(`_row_group_cache_size`, hardcoded), one row group per shard, ~1.4 GB decoded at pu0, so resident
memory is roughly `8 × workers × 1.4 GB` — the 255 GB measured at 24 workers, on a 515 GB shared
node. Going further means lowering that cache size in `data.py` first, not raising
`--cpus-per-task`. This is also why `WORKERS` defaults to a literal 24 rather than to
`SLURM_CPUS_PER_TASK`: tying workers to cores would quietly make `--cpus-per-task` a memory knob.

**The speedup was spent on epochs, not on a shorter run.** `calo_clustering.yaml` now sets
`max_epochs: 7`, so ~20 h of walltime buys **140,000 optimiser steps** where 4 epochs at the old
rate would have taken 20.8 h to reach 80,000. That is the trade the model wants:
`overlay_long_schedule.yaml` sets out the evidence that it is step-starved rather than
data-starved, and run 48169's validation loss was still falling at its final checkpoint.

The walltime follows: **26 h requested against a measured 20.4 h**, with `MAX_TIME` stopping
Lightning cleanly at 25 h. The 27% margin is for throughput drift, not slack — a 20% slowdown still
completes. For context, job 48169 trained the *previous* configuration — same events, 4 epochs,
same card, 12 CPUs — in 19:37:05 against a 20:00:00 wall, with 23 minutes to spare. This
configuration is slightly slower per event (AdamW carries two momentum buffers where Lion carried
one; `accumulate_grad_batches` 4 → 1 pays the optimiser step four times as often), so **at the old
allocation even 4 epochs would have been truncated by that wall** — and a OneCycle schedule that
never reaches its decay phase leaves a final checkpoint at a high learning rate.

To go further than 7 epochs, `overlay_long_schedule.yaml` sets 12 (~34.9 h, single job — the
partition cap is 4 days). It no longer restates `accumulate_grad_batches`, which the previous
version pinned at 4 and which would now quarter the optimiser steps it exists to add.

Peak GPU memory is 49.2 GB of 81.9, so `batch_size: 2` would fit and would also raise utilisation.
It is deliberately left at 1: the effective batch is coupled to `lrs_config.max` and
`accumulate_grad_batches`, and `calo_clustering.yaml` is explicit that those were set together.
That is a training decision, not a resource one.

## Order to run things

```bash
mkdir -p external/slurm_logs

# 1. ~20 min. Ten events, twenty epochs. Does the new objective run at all?
sbatch src/maskformer/dias/smoke_calo_clustering.sh

# 2. ~15 min. Re-measure if anything about the config's per-step cost changed.
sbatch src/maskformer/dias/benchmark_calo_clustering.sh

# 3. the real run: 7 epochs, ~20.4 h expected, 26 h allocation
sbatch src/maskformer/dias/train_calo_clustering.sh

# 4. the event store, from the trained checkpoint (mirrored script, unchanged, LIGHTGPU)
CKPT=external/logs/<run>/ckpts/<best>.ckpt START=20250 NUM=500 OUT=~/eventstore \
  sbatch src/maskformer/hepattn_colliderml/slurm/calo_dump_eventstore.sh
```

`COMET_API_KEY` does not need exporting first: every script here uses a `#!/bin/bash -l` shebang,
so a login shell sources `~/.bashrc`, which exports it. Dropping the `-l` would break Comet logging
in a way that presents as a Comet problem. (`~/.bashrc` is mode 0644 on a shared cluster, so any
account on it can read the key — `../ce_ai_1/README.md`'s "rotate it if it has been shared
anywhere" applies.)

**Step 1 is not optional.** The configuration being submitted is not one any previous DIAS job ran:
different optimiser, different accumulation, different gradient clip, no incidence head, and a mask
objective of equal-weighted dice + bce in place of dice 5 + focal 20.

Logs and checkpoints go to `external/logs/` (gitignored), not into the hepattn checkout, so a run
leaves nothing outside this repository except the synced `data.py`/`model.py`.

`train_calo_clustering.sh` refuses to start if `NUM_TRAIN` would run past 20,000 into the
`[20250, 20750)` window the evaluation store is dumped from. Training on the events the CLUE
comparison is scored over is the one error that makes every downstream number wrong while looking
fine — `src/io/event_store.py` asserts it too, but hours later.

Afterwards, re-sweep `flow_valid`'s `eval_threshold` with
`hepattn_colliderml/scripts/sweep_pred_threshold.py`. The 0.2 in the config was measured against
the *old* objective, with the focal term and the incidence head, and the mask head's calibration is
exactly what a focal term changes.

## Do not move this to LIGHTGPU

It is usually idle and `calo_dump_eventstore.sh` defaults to it, which makes it a tempting way to
skip the queue. It is `compute-gpu-0-0`, carved into six MIG instances of ~20 GB. Dumping a store
is a forward pass and fits; training peaks at 49 GB and does not. `env.sh:select_gpu` warns if it
is tried rather than letting it fail an hour in.

## Pileup-200 does not run here

`configs/overlay_pu200_barrel.yaml` points at `/mnt/ai-datastore/finnbar/ColliderML_data/`, which
is ce-ai-1's datastore. DIAS has `ttbar_pu0` and nothing else, so there is no Slurm script for the
barrel run and one would abort on the missing dataset if there were. Run it on ce-ai-1 with
`../ce_ai_1/train_pu200.sh`. If it ever needs to move here, stage the 297 GB of shards first
(`python setup/download_data.py`; `/home` has room) and give the overlay a DIAS `train_dir` — and
note that the barrel cut needs *this repository's* `data.py`, since the checkout's copy predates
`calohit_max_abs_eta`, so such a run would depend on the sync above rather than merely preferring
it.
