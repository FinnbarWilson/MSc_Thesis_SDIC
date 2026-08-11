# `ce_ai_1/` — running this on ce-ai-1, and the pileup-200 run

Machine-specific launchers for **ce-ai-1**, and the record of what pileup-200 forced to change.
The pu0 results were produced on DIAS under Slurm; nothing about that setup survives the move, and
this directory is where the differences live rather than being edited into the pu0 scripts.

These are **not** in `hepattn_colliderml/`. Everything under that directory is a verbatim mirror of
the hepattn checkout, checked by `verify_sync.sh`; these launchers are mine and would make that
check fail against a clean upstream. `configs/overlay_pu200_barrel.yaml` *is* in the mirror, because it is
an experiment config that `main.py` loads by path, exactly like the other overlays.

## What changed moving off DIAS

| | DIAS (pu0) | ce-ai-1 (here) |
|---|---|---|
| scheduling | Slurm, 20 h walltime cap, queueing | run it directly, `nohup` for long runs |
| OS | RHEL7, glibc 2.17 → apptainer container | Ubuntu 24.04, glibc 2.39 → no container |
| python env | pixi env inside the container | plain venv on system python 3.12 |
| GPU | 3 × A100, one with 818 uncorrected ECC errors | 1 × A100 80 GB, healthy |
| host RAM | `--mem=128G` | 1.5 TB |
| storage | `$HOME` had room | `/` has ~100 GB free → everything on `/mnt/ai-datastore/finnbar` |

Two consequences worth stating. The **ECC preflight is gone** — it existed because Slurm kept
handing out one specific faulty card, and there is nothing to select between here. And the run no
longer has to be **split into resumable jobs**: `overlay_long_schedule.yaml` exists because a
12-epoch schedule could not fit a 20 h cap, whereas `overlay_pu200_barrel.yaml` sizes a single ~21 h
schedule that reaches its OneCycle decay in one go.

The GPU reports MIG enabled, but as a single `7g.80gb` instance — that is the whole card, not a
slice, so there is no capacity loss and no device selection to do.

## The environments — there are two, and they are separate on purpose

| | built by | used for |
|---|---|---|
| `external/venv-hepattn` | `setup/install_training_env.sh` | training, dumping stores. torch + hepattn + GPU |
| `external/conda-envs/calo-clustering` | `setup/install_analysis_env.sh` | CLUE, scoring, figures. numpy-only, no GPU |

Both live under `external/`, which is gitignored — see [`setup/README.md`](../../../setup/README.md).
**Only the dataset lives outside the repository**, on `/mnt/ai-datastore/finnbar/ColliderML_data`,
because that is a shared datastore and 300 GB of parquet is the only thing that belongs on it.

The split between the two environments is the repository's central design decision, not an
accident of installation — see the root README. The analysis env is conda because
`environment.yml` documents why pip cannot build CLUEstering's dependencies here.

`install_training_env.sh` clones hepattn, applies `hepattn-changes.patch`, and copies
`hepattn_colliderml/` over `src/hepattn/experiments/colliderml/` — the same three steps
`../README.md` gives by hand. Note it pins **`cb4fb10`, not the `30ccb9f` that `../README.md`
names**: that commit no longer exists upstream. See `setup/README.md`.

## Why pu200 is not pu0 with more hits

Measured on `ttbar_pu200` shard 0, against the cuts `calo_clustering.yaml` applies:

| | pu0 (config built for) | pu200 (measured) | ratio |
|---|---|---|---|
| calo hits/event (>2e-4 GeV) | ~22,000 | 532,507 | 24× |
| target particles/event | ~600 | 8,182 | 13.6× |

MaskFormer's memory is driven by `num_queries × num_hits`. pu0 sits at 2.2e7 and *already OOMs at
4×* — `calo_clustering.yaml` records `batch_size: 4` OOMing an 80 GB A100. A faithful pu200 event
is 4.4e9, **about 200× the pu0 footprint**: two orders of magnitude past the card, not a tuning
problem.

`configs/overlay_pu200_barrel.yaml` buys that back with two measured cuts — `calohit_min_energy`
2e-4 → 1e-3 and `particle_min_pt` 0.5 → 2.0 — landing at 117k hits and 277 targets per event, a
2.66× footprint. Its header carries the full arithmetic, the energy cost, and the OOM ladder.

**These numbers are not comparable to `results/pu0/`.** The target definition changed and 41% of
the calorimeter energy is suppressed. What is *not* damaged is the pu200 head-to-head itself: CLUE
reads the store dumped from the model's own dataloader, so both methods see exactly these cells.
The comparison stays controlled; only its comparability to pu0 is lost. Say so when quoting both.

## Event budget

100 shards × 100 events = **10,000 events** downloaded (of 1000 shards available). Windows, all
disjoint, split between `overlay_pu200_barrel.yaml` and `config/experiment.yaml`:

```
train [0, 6000)   val [6000, 6250)   test [6250, 6750)
CLUE tune store [7000, 7050)    CLUE eval store [7500, 8000)    spare [8000, 10000)
```

`train_pu200.sh` refuses to start if `NUM_TRAIN` would run into the store windows, because
training on the events the comparison is scored over is the one error that makes every downstream
number wrong while looking fine. To go beyond 10,000 events:
`python setup/download_data.py --shards 200` — it resumes and skips what is already there.

## The order to run things

```bash
# 0. once, from the repository root
./setup/install_training_env.sh
./setup/install_analysis_env.sh
python setup/download_data.py                 # ~297 GB, resumable
python setup/verify_data.py                   # 100 matched shards per collection, 10,000 events

# 1. does it fit, and how fast? ~15 min. Prints the max_epochs for a 22 h run.
./benchmark_pu200.sh

# 2. set trainer.max_epochs in configs/overlay_pu200_barrel.yaml from what step 1 printed, then:
nohup ./train_pu200.sh > ~/train_pu200.log 2>&1 &

# 3. both stores, from the trained checkpoint
CKPT=<logs/.../ckpts/....ckpt> ./dump_store_pu200.sh tune
CKPT=<logs/.../ckpts/....ckpt> ./dump_store_pu200.sh eval

# 4. point config/experiment.yaml at them, then everything below is numpy
#    dataset.active: pu200, dataset.pu200.store / .tune_store / .overrides.maskformer.checkpoint
python -m scripts.show_config
python -m scripts.tune_clue            # READ THE EDGE WARNINGS -- see below
python -m scripts.scan_working_points  # re-derive the mask/object thresholds; 0.5/0.2 is a pu0 result
python -m scripts.score --algo clue
python -m scripts.score --algo maskformer
python -m scripts.make_figures
```

**Step 1 is not optional.** `overlay_pu200_barrel.yaml` sizes `max_epochs` from an *estimate* of
0.25 events/s extrapolated from pu0's measured 1.13. OneCycleLR is sized from total steps, so a
wrong estimate does not just give a run of the wrong length — it gives a run whose final
checkpoint sits at a high learning rate. That is exactly how the hit-filter run was wasted.

**On `tune_clue`:** `config/experiment.yaml` deliberately does not override the CLUE search ranges
for pu200, and explains why inventing them would be the same failure the pu0 ranges were widened to
avoid. Run it with the pu0 ranges, read the edge warnings `tune_subsystem` prints, widen in the
direction it names, and re-run until nothing presses a bound. A baseline tuned in the wrong box is
under-tuned, which is the one way this comparison can be unfair to CLUE.

## Keeping pu0 reproducible

Nothing here touches it. `dataset.active` in `config/experiment.yaml` is the only switch, and it
scopes the stores, `results/<dataset>/`, `figures/<dataset>/` and the Optuna study names together —
so a pu200 run cannot land on a pu0 table. `overlay_pu200_barrel.yaml` additionally logs to a separate
Comet project, and every pu200 value it changes lives in the overlay rather than being edited into
`calo_clustering.yaml`, so the pu0 configuration is still exactly what produced the checkpoint.

To regenerate the pu0 figures at any point, from the analysis env alone:

```bash
# config/experiment.yaml: dataset.active: pu0
python -m scripts.make_figures
```

## Secrets

`COMET_API_KEY` is read from `~/.config/colliderml/comet.env` (mode 0600) — outside the git
worktree so it cannot be committed, and not on the shared datastore either.
`calo_clustering.yaml` already took the key from the environment rather than inline; `env.sh` just
supplies it. **Rotate the key at comet.com if it has been shared anywhere** — a Comet key allows
writing to the workspace.
