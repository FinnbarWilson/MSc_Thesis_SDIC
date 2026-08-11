# Pileup-200: where it stands, and the decision that is open

Written 2026-08-05, after one failed 19.5 h training run and two benchmarks. Everything here is
measured on `ttbar_pu200` shard 0 (100 events) or on ce-ai-1's A100 80GB unless stated.

The open question is at the bottom. Everything above it is the evidence for it.

---

## 1. The problem pileup-200 poses

MaskFormer's memory goes as `num_queries x num_hits`. Per event:

| | pu0 | pu200 | ratio |
|---|---|---|---|
| calo cells (>2e-4 GeV) | ~22,000 | 532,507 | 24x |
| target particles (pt>0.5, \|eta\|<4) | ~600 | 8,182 | 13.6x |
| `num_queries x num_hits` | 2.2e7 | 4.4e9 | **~200x** |

`calo_clustering.yaml` records that pu0 already OOMs at 4x (`batch_size: 4` on an 80 GB A100). So
pu200 at pu0 settings is roughly two orders of magnitude past the card. Something must shrink.

## 2. Attempt 1 shrank the wrong thing, and failed

Cut targets hard (`particle_min_pt` 0.5 -> 2.0) to afford queries; cut cells only mildly
(`calohit_min_energy` 2e-4 -> 1e-3). Cells and targets then moved independently, and the fraction
of cells belonging to *any* target collapsed:

| | cells | targets | **cells owned by a target** |
|---|---|---|---|
| pu0 | ~22,000 | ~600 | **~37%** |
| attempt 1 | 102,010 | 231 | **2.5%** |

At 2.5% owned, the mask loss is minimised by predicting that no query owns anything — and that is
what the model found. Over 7 epochs / 19.5 h:

- train loss 30.35 -> 23.60 (fell early, then flat)
- **val loss 23.89 first validation, 23.90 last**, best 23.49 at epoch 1 and worse after

This was not a learning-rate problem, and lowering the LR would not have fixed it. The degenerate
solution was genuinely near-optimal for the data as presented.

Comet: `colliderml-calo-clustering-pu200`, run `68cd388c30e24761bc46e2ee60f2da36`.

## 3. The barrel cut fixes the physics

Supervisor's suggestion: restrict to the barrel so the reconstruction task is possible, and keep
pu0's cuts otherwise. Cutting in **eta removes cells and targets together**, so the ratio is
preserved — which is precisely what attempt 1 destroyed.

**Where the barrel ends.** Measured from cell positions: HCAL barrel reaches r = 3441 with
|z| <= 3450, so a track steeper than **eta = 0.883** exits the barrel end before the outer radius
and deposits the rest of its shower in the endcap. The endcaps start well outside that (`ece` at
eta ~ 1.55, `hce` at ~ 1.24), so |eta| < 0.88 selects `ecb` + `hcb` exactly, with nothing
straddling the boundary and no partially-contained showers.

With `calohit_min_energy` and `particle_min_pt` back at their pu0 values:

| | cells | targets | queries | **owned** | **hits/target** |
|---|---|---|---|---|---|
| pu0 | ~22,000 | ~600 | 1,000 | ~37% | 13 |
| **barrel, \|eta\|<0.88** | 52,126 | 1,342 | 2,100 | **38.1%** | **15.4** |
| attempt 1 | 102,010 | 231 | 350 | 2.5% | 10.3 |

It lands on the pu0 regime on both axes, and needs **no energy cut at all** — 93% of calorimeter
energy retained, showers intact. (Attempt 1 kept 59%, leaving hard particles 10.3 cells instead of
27.7.) The target definition is pu0's, so results are comparable to `results/pu0/` in a way
attempt 1's never could have been.

If containment through the **ECAL barrel only** were acceptable (r = 1519, |z| = 3050), the cut
would be eta < 1.44: 105,148 cells, 2,546 targets — more statistics, but each shower's HCAL
component leaks into `hce`.

## 4. It fits, but it is slow

Benchmark, 200 events, `|eta| < 0.88`, 2,100 queries:

- **peak GPU 56,146 MiB of 81,037** — comfortable, 25 GB spare (attempt 1 peaked at 68,456)
- **0.36 events/s** (attempt 1: 0.98 — the 2,100 queries against 350 cost ~3x throughput)

Applying the cache de-rating (see §6) gives ~0.27 ev/s, so ~6.5 h per 6,000-event epoch and about
3 epochs in a 22 h run.

## 5. THE OPEN DECISION: optimiser steps

`calo_clustering.yaml` concluded from two pu0 runs that **performance tracks optimiser steps**, not
data volume — 3,000x28 and 20,000x4 reached nearly the same loss at nearly the same step count
despite 6.7x the data — and calls its own 20,000-step run undertrained.

Steps available are fixed by walltime, throughput and accumulation. The epoch/`num_train` split
does not enter it:

```
steps = walltime x rate / accumulate_grad_batches
      = 22 h x 3600 x 0.27 / accumulation
```

| `accumulate_grad_batches` | steps in 22 h | vs pu0's undertrained 20,000 |
|---|---|---|
| **4 (current)** | **~4,200** | 0.21x |
| 2 | ~8,400 | 0.42x |
| 1 | ~16,600 | 0.83x |

At the current accumulation of 4 the run gets a fifth of an already-undertrained step count. The
data is now correct, but the model would still not learn — for a different reason than attempt 1.

**Accumulation is the only meaningful lever**, and it is coupled to the learning rate.
`calo_clustering.yaml` sets `lrs_config.max: 1e-4` and states it is "only valid TOGETHER WITH
`accumulate_grad_batches: 4`", because Lion uses only the gradient *sign*, so averaging 4 events
before taking it is what makes the larger step safe. It records 1e-4 thrashing and 1.5e-4 diverging
at batch 1, with **5e-5 the value that worked there**.

So the options are:

| option | steps | change |
|---|---|---|
| A | ~16,600 | `accumulation: 1`, `lrs_config.max: 5e-5` |
| B | ~8,400 | `accumulation: 2`, `lrs_config.max: 7e-5` (sqrt(2) scaling) |
| C | ~4,200 | leave as is — not expected to train |

A is closest to pu0's step count; B keeps some gradient averaging so Lion's sign is less noisy.
Neither changes memory: accumulation affects how often we step, not the per-step footprint.

Tightening eta to ~0.7 would buy perhaps 25% throughput (fewer cells *and* fewer queries), but does
not close a 4x step gap on its own and narrows the physics scope further.

## 6. Two methodology notes worth keeping

**Benchmarks over-read throughput.** `data.py` caches 8 decoded row groups per worker and there is
one row group per shard, so a 200-event benchmark touches 2 shards and stays fully cached. Measured
at pu0, the penalty going to many shards was -25%; **measured here it was -41%** (benchmark 0.98 ->
real 0.58), because a decoded pu200 shard is 4.0 GB against ~1.5 GB at pu0, so the same 8 slots
hold far less. Treat any short-benchmark rate as an upper bound, or benchmark with
`EVENTS >= num_train/10`.

**Sizing the schedule matters more than usual.** OneCycleLR is sized from total steps, so a run
that overruns its walltime never reaches its decay phase and its final checkpoint is taken at a
high learning rate. The first barrel-era run was caught at 21 minutes for exactly this (9 epochs
would have needed 27.4 h against a 23:30 wall).

## 7. Secondary question: removing BCE

The suggestion to zero the BCE term may not match what the config contains. **There is no
`mask_bce`** — the mask task uses `mask_dice: 5.0` + `mask_focal: 20.0`, and focal already
down-weights easy negatives. The only BCE is `object_bce: 1.0` on the `flow_valid` head, which
predicts whether a query corresponds to a real particle.

Zeroing that is self-consistent — invalid queries would learn to emit empty masks through
dice/focal, and clusters would be selected on mask occupancy instead — but it makes the
`object_threshold: 0.2` working point meaningless, and `scan_working_points` would have to scan
`mask_threshold` alone. Since the barrel cut already restores the 38% owned fraction that caused
the failure, changing both at once would make the result unattributable.

## 8. State of the machine

Nothing is running; the GPU is idle. Ready to launch as soon as §5 is decided.

- data: 10,000 pu200 events (297 GB) downloaded and verified, on `/mnt/ai-datastore/finnbar` only
- environments: both built under `external/`, all imports verified
- `configs/overlay_pu200_barrel.yaml`: written, benchmarked, `max_epochs` still a placeholder
- `data.py`: gained `calohit_max_abs_eta` (default 0 = disabled, so pu0 behaviour is unchanged)
- `configs/overlay_pu200.yaml`: attempt 1; since DELETED as superseded (see git history)

Unrelated but outstanding: `src/maskformer/README.md` pins hepattn at `30ccb9f`, which no longer
exists upstream — `setup/install_training_env.sh` uses `cb4fb10` instead.
