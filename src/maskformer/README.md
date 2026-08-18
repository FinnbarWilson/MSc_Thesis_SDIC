# `src/maskformer/` — the learned model: training it, and producing the event store

The MaskFormer half of the comparison: the dataset, the model configuration, the launchers for
both machines, and the evaluation dump that turns a trained checkpoint into an event store.
Everything else under `src/` is deliberately numpy-only and runs on a laptop; this part is not.

Read [`../../README.md`](../../README.md) first for what the comparison is and how the two
methods are scored. **Nothing here is needed to reproduce the thesis figures** — that is
`python -m scripts.make_thesis_figures --from-summary` from the repository root.

## The dependency boundary

`src/clue/`, `src/evaluation/`, `src/plotting/` and `src/io/` import nothing but
numpy, scipy, pandas and matplotlib. The code here needs torch, `hepattn` and a GPU. Living in a
sibling directory is not what keeps the two apart — **nothing importing across the line** is:

```bash
grep -rn "src\.maskformer" --include="*.py" src scripts tests   # returns nothing
```

```
  src/maskformer/  ──(writes)──>  event store (plain .npz)  ──(reads)──>  the rest of src/
  GPU, hepattn, dataset                                          numpy and nothing else
```

`src/io/event_store.py` is a hand-written *mirror* of `hepattn_colliderml/eval/format.py`, not an
import of it. Two readers of the same format, maintained side by side, is what keeps the boundary
honest: if one side changes the format, the other fails loudly on a version check rather than
silently misreading.

There is no `__init__.py` at this level and there is not meant to be. This directory is a *source
of record*, not an importable subpackage — `import src.maskformer` should not be a thing anyone
can do, and leaving the `__init__.py` out is what enforces it.

## What is mine and what is not

The model is built from **`hepattn`** (<https://github.com/samvanstroud/hepattn>), an existing
library for transformer-based reconstruction in HEP. I did not write it, and it is not reproduced
here — vendoring someone else's framework into a thesis repository would misstate authorship in
the one direction that matters.

`hepattn` gained a ColliderML experiment upstream (commit `cb4fb10`, "Add ColliderML
Experiment"), so this directory is a mixture. Checked file by file against that commit:

| file | status |
|---|---|
| `configs/pu0.yaml`, `configs/pu200.yaml` | **mine** — not upstream |
| `eval/dump.py`, `eval/format.py`, `eval/geometry.py` | **mine** — not upstream |
| `data.py` | upstream file, **heavily modified** (889 lines differ): the shower-level truth collapse, the cell and particle selections, the calo association builders |
| `model.py`, `main.py` | **upstream, unmodified** — byte-identical to `cb4fb10` |
| `hepattn-changes.patch` | **mine** — three modifications to the library proper |

Not mine either way: `hepattn` itself — the transformer encoder, the MaskFormer decoder, the task
heads, the loss functions and the Hungarian matcher.

Everything under `hepattn_colliderml/` is a **verbatim copy** of
`hepattn/src/hepattn/experiments/colliderml/`, which is where it actually runs. Nothing else in
this directory is. That is the reason for the extra nesting: the answer to "which files must stay
byte-identical to upstream" is `ls hepattn_colliderml/` rather than a list someone has to keep in
their head. Its filenames are upstream's and cannot be made more descriptive — `data.py` is
imported as `hepattn.experiments.colliderml.data`. Check the copies against a checkout with:

```bash
HEPATTN=/path/to/hepattn ./verify_sync.sh
```

`hepattn-changes.patch` holds three small modifications, kept as a patch rather than a copy so
what I changed is separable from what was already there:

- `models/loss.py` — adds `mask_dice_weighted_loss`. The plain DICE loss documents that it ignores
  its `sample_weight` argument and every existing caller passes one, so honouring it there would
  have silently changed the objective of every configuration in the library. Opting in under a new
  name cannot. Under a uniform weight it reduces exactly to the original.
- `models/task.py` — adds `constituent_weight_field` to `ObjectHitMaskTask`, so the mask loss can
  be weighted per cell.
- `callbacks/prediction_writer.py` — an `output_name` argument. The writer named its output after
  the data directory, and all three splits live in one directory here, so every split overwrote
  the same file.

**A known defect in the upstream `model.py`, left unfixed on purpose.** Its `log_custom_metrics`
does `pred_hit_masks &= ...` and `true_hit_masks &= ...` on tensors it holds by reference, so it
mutates the `preds` and `targets` dicts in place. It is harmless as the wrapper currently calls it
— logging runs after `model.loss()` and the batch is discarded afterwards — but it is one
reordering away from corrupting the truth masks during training. It is upstream's code, so the fix
belongs upstream or in the patch, not as a silent edit to a file claimed to be a verbatim mirror.

**The hepattn pin.** `install_training_env.sh` pins `cb4fb10`. An earlier reference version,
`30ccb9f`, no longer exists upstream — a fresh clone reports `fatal: Not a valid object name`, so
it was rebased or squashed away. `hepattn-changes.patch` applies cleanly to `cb4fb10`, and also to
`main`, `1df05cc` and `93b2842`, so this is the most specific surviving choice rather than the only
one that works. Override with `HEPATTN_COMMIT=<sha> ./setup/install_training_env.sh`. Note the
risk this leaves: upstream has deleted a pinned commit once and can again, at which point the
checkpoint's provenance is no longer reconstructible from this repository alone.

## Layout

```
hepattn_colliderml/          verbatim copy of hepattn/src/hepattn/experiments/colliderml/
  data.py                    ColliderMLDataset / ColliderMLDataModule: reads the raw parquet
                             shards, applies the cell and particle selections, and builds the
                             truth targets -- including the shower-level collapse
  model.py                   ColliderMLModel, a thin LightningModule wrapper
  main.py                    CLI entry point (Lightning's LightningCLI)
  configs/pu0.yaml           pileup 0, full detector. Self-contained
  configs/pu200.yaml         pileup 200, barrel only. Self-contained
  eval/dump.py               runs the model over an event window and writes an event store.
                             The only GPU-dependent step of the analysis
  eval/format.py             the on-disk store format. Mirrored by src/io/event_store.py
  eval/geometry.py           recovers 48 ECAL and 36 HCAL layers by projecting barrel cells
                             onto the stave normal

dias/                        Slurm launchers for DIAS, where pu0 was trained and scored
ce_ai_1/                     launchers for ce-ai-1, where pu200 was trained and scored
hepattn-changes.patch        my modifications to the upstream library
verify_sync.sh               checks hepattn_colliderml/ against a hepattn checkout
```

The launchers are **not** in `hepattn_colliderml/`, because they are mine and would make
`verify_sync.sh` fail against a clean upstream. `configs/pu200.yaml` *is* in the mirror, because
it is an experiment config `main.py` loads by path, exactly like the other overlays.

**There are exactly two configs, one per pileup condition, and neither is an overlay on the
other.** That replaced a stack of nineteen overlay files. Overlays meant the objective a run
actually used depended on the *order* of several `--config` flags, and because `tasks` is a YAML
list, any overlay touching it replaced the whole list rather than merging into it — which is how
one pu0 run ended up on a different mask objective from the one its base config documented. The
two files are identical except where the data forces a difference, and those places are marked
`PU200 DIFFERS` / `PU0 DIFFERS`; `diff configs/pu0.yaml configs/pu200.yaml` is the intended way to
see what pileup changes.

## Why pileup 200 is not pileup 0 with more hits

Measured on `ttbar_pu200` shard 0:

| | pu0 | pu200 | ratio |
|---|---|---|---|
| calo hits/event (> 2e-4 GeV) | ~22,000 | 532,507 | 24x |
| target particles/event | ~600 | 8,182 | 13.6x |

MaskFormer's memory is driven by `num_queries x num_hits`. pu0 sits at 2.2e7 and already OOMs at
batch 4 on an 80 GB A100. A faithful pu200 event is 4.4e9 — about **200x the pu0 footprint**, two
orders of magnitude past the card rather than a tuning problem. `configs/pu200.yaml` buys that
back by restricting to the barrel (|eta| <= 0.88); its header carries the full arithmetic, the
energy cost, and the OOM ladder.

The pu200 model is trained as **its own model**, not the pu0 checkpoint evaluated on pu200 — a
pu0-trained model run on pu200 measures domain shift, not the architecture. It got a 5x smaller
training budget (24,000 steps against 120,000), which the thesis states wherever the two columns
are compared.

## Event budget

100 shards x 100 events = 10,000 pu200 events downloaded, of 1000 shards available. All windows
disjoint, split between `configs/pu200.yaml` and `config/experiment.yaml`:

```
train [0, 6000)   val [6000, 6250)   test [6250, 6750)
CLUE tune store [7000, 7050)    CLUE eval store [7500, 8000)    spare [8000, 10000)
```

`train.sh pu200` refuses to start if `NUM_TRAIN` would run into the store windows, because
training on the events the comparison is scored over is the one error that makes every downstream
number wrong while looking fine. For more events: `python setup/download_data.py --shards 200`.

## Running it

Everything below assumes the training env from `setup/install_training_env.sh`. `ce_ai_1/env.sh`
re-syncs this directory into the hepattn checkout on every launch, because `main.py` runs from the
checkout and editing a config here without copying it across silently uses the stale copy.

```bash
cd src/maskformer/ce_ai_1

# 1. does it fit, and how fast? ~15 min. Prints the max_epochs for a 22 h run.
NUM_TRAIN=600 MAX_EPOCHS=1 ./train.sh pu200

# 2. set trainer.max_epochs in configs/pu200.yaml from what step 1 printed, then:
nohup ./train.sh pu200 > ../../../external/train_pu200.log 2>&1 &

# 3. both stores, from the trained checkpoint
CKPT=<logs/.../ckpts/....ckpt> ./dump_store.sh pu200 tune
CKPT=<logs/.../ckpts/....ckpt> ./dump_store.sh pu200 eval
```

**Step 1 is not optional.** OneCycleLR is sized from total steps, so a wrong throughput estimate
does not just give a run of the wrong length — it gives a run whose final checkpoint sits at a
high learning rate.

Then point `config/experiment.yaml` at the new stores (`dataset.pu200.store`, `.tune_store`,
`.overrides.maskformer.checkpoint`), confirm with `python -m scripts.show_config`, and continue
with the numpy-only pipeline in the root README.

### What differs between the two machines

| | DIAS (pu0) | ce-ai-1 (pu200) |
|---|---|---|
| scheduling | Slurm, 20 h walltime cap | run directly, `nohup` for long runs |
| OS | RHEL7, glibc 2.17 → apptainer container | Ubuntu 24.04 → no container |
| python env | pixi env inside the container | plain venv on system python 3.12 |
| GPU | 3 x A100, one with 818 uncorrected ECC errors | 1 x A100 80 GB, healthy |
| storage | `$HOME` had room | everything on `/mnt/ai-datastore/finnbar` |

On DIAS both environments are built *and* run inside `~/ubuntu22.sif`, because hepattn and the
pinned conda-forge builds need glibc 2.28+:

```bash
apptainer build ~/ubuntu22.sif docker://ubuntu:22.04
apptainer exec --bind $HOME ~/ubuntu22.sif bash setup/install_analysis_env.sh
```

The image is the bare `ubuntu:22.04` base and ships no compiler and no `curl`. `dias/env.sh`
points `CC`/`CXX` at the pixi env's conda-forge GCC — without it Triton cannot initialise its
driver and training dies at the first kernel — and `install_analysis_env.sh` falls back to wget or
python for the miniforge download.

`config/experiment.yaml` holds ce-ai-1's `/mnt/ai-datastore` paths. `CALO_STORE_ROOT` relocates
them without editing the config, so one config stays valid on both machines; only the directory
moves, since the store *name* encodes the window and format version `EventStore` checks.

The `--mem` values in the Slurm scripts look absurd for the work being done and are not
negotiable: Slurm here applies `VSizeFactor`, so `--mem` is a hard *virtual* memory cap of 1.1x
the request, and `expandable_segments` reserves a large virtual range at start-up. An
under-request surfaces as `CUDA driver error: out of memory` on an idle 80 GB card.

## The checkpoint — and why there is none here

**No checkpoint is tracked in this repository.** One was, and it was deleted: it had been trained
on an objective the thesis no longer uses, so it could not produce any reported number while
looking as though it could, and it was 90% of the repository's tracked bytes.
`git log -- src/maskformer/checkpoint/` recovers it with its provenance intact.

Training writes checkpoints to `external/hepattn/.../logs/<run>/ckpts/`, which is gitignored and
machine-local. Each run also writes its own fully-resolved `config.yaml` beside them — that file,
not `configs/pu0.yaml`, is the authoritative record of what was trained, because the configs move
on. A checkpoint is needed only to dump an event store; nothing in `scripts/` touches one.

`COMET_API_KEY` is read from `~/.config/colliderml/comet.env` (mode 0600), outside the git
worktree so it cannot be committed.
