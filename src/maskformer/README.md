# `src/maskformer/` — the learned model: training it, and producing the event store

This directory holds the MaskFormer half of the comparison: the dataset, the model
configuration, and the evaluation dump that turns a trained checkpoint into an event
store. Everything else under `src/` is deliberately numpy-only and runs on a laptop; this
part is not.

Read `../../README.md` first for what the comparison is and how the two methods are scored.
This file covers only how the model is defined, trained, and turned into an event store.

## The dependency boundary, and how it survives being under `src/`

`src/clue/`, `src/evaluation/`, `src/plotting/` and `src/io/` import nothing but numpy,
scipy, pandas and matplotlib. That is what lets an assessor regenerate every figure in the
thesis from the tables in `results/` with no GPU, no `hepattn`, and no access to the 991 GB
ColliderML dataset. The code here needs all three.

Living in a sibling directory is not what keeps those two apart — **nothing importing across
the line** is. That is the invariant, and it is unchanged and checkable:

```bash
grep -rn "src\.maskformer" --include="*.py" src scripts tests
```

returns nothing, and the files here import `hepattn.experiments.colliderml.*`, never `src.*`.
Neither half can reach the other even by accident, because neither names the other.

```
  src/maskformer/  ──(writes)──>  event store (plain .npz)  ──(reads)──>  the rest of src/
  GPU, hepattn, dataset                                          numpy and nothing else
```

`src/io/event_store.py` is a hand-written *mirror* of `hepattn_colliderml/eval/format.py`,
not an import of it. Two readers of the same format, maintained side by side, is the
mechanism that keeps the boundary honest: if this side changes the format, the other side
fails loudly on a version check rather than silently misreading the file.

There is no `__init__.py` at this level and there is not meant to be. This directory is a
*source of record*, not an importable subpackage of `src` — `import src.maskformer` should
not be a thing anyone can do, and leaving the `__init__.py` out is what enforces it.

## `hepattn_colliderml/` — the mirrored subtree

Everything under `hepattn_colliderml/` is a **verbatim copy** of
`hepattn/src/hepattn/experiments/colliderml/`, which is where it actually runs. Nothing else
in this directory is. That is the whole reason for the extra level of nesting: the answer to
"which of these files must stay byte-identical to upstream" is `ls hepattn_colliderml/`
rather than a list someone has to keep in their head.

The filenames inside it are upstream's and cannot be made more descriptive — `data.py` is
imported as `hepattn.experiments.colliderml.data`, and `verify_sync.sh` diffs them by name.
Renaming any of them would break both.

`verify_sync.sh` checks the copies against a `hepattn` checkout, so "these are the files that
produced the reported results" is checkable rather than asserted:

```bash
HEPATTN=/path/to/hepattn ./verify_sync.sh
```

## What is mine and what is not

The model is built from **`hepattn`** (<https://github.com/samvanstroud/hepattn>), an existing
library for transformer-based reconstruction in HEP. I did not write it, and it is not
reproduced here — vendoring someone else's framework into a thesis repository would misstate
authorship in the one direction that matters.

`hepattn` gained a ColliderML experiment upstream (commit `cb4fb10`, "Add ColliderML
Experiment"), so this directory is a mixture rather than wholly mine. Checked file by file
against that commit on 2026-08-13:

| file | status |
|---|---|
| `configs/pu0.yaml`, `configs/pu200.yaml` | **mine** — not upstream |
| `eval/dump.py`, `eval/format.py`, `eval/geometry.py` | **mine** — not upstream |
| `data.py` | upstream file, **heavily modified by me** (889 lines differ): the shower-level truth collapse, the cell and particle selections, the calo association builders |
| `model.py` | **upstream, unmodified** — byte-identical to `cb4fb10` |
| `main.py` | **upstream, unmodified** — byte-identical to `cb4fb10` |
| `hepattn-changes.patch` | **mine** — three modifications to the library proper |

Not mine either way: `hepattn` itself — the transformer encoder, the MaskFormer decoder, the
task heads, the loss functions and the Hungarian matcher.

Verify any of the above with:

```bash
diff <(git -C <hepattn> show cb4fb10:src/hepattn/experiments/colliderml/model.py) \
     hepattn_colliderml/model.py
```

**A known defect in the upstream `model.py`, left unfixed on purpose.** Its
`log_custom_metrics` does `pred_hit_masks &= ...` and `true_hit_masks &= ...` on tensors it
holds by reference, so it mutates the `preds` and `targets` dicts in place. It is harmless as
the wrapper currently calls it — logging runs after `model.loss()` in both the training and
validation steps, and the batch is discarded afterwards — but it is one reordering away from
corrupting the truth masks during training. It is upstream's code, so the fix belongs upstream
or in `hepattn-changes.patch`, not as a silent edit to a file this repository claims is a
verbatim mirror.

`hepattn-changes.patch` is the exception worth reading: three small modifications I made to
the upstream library, kept as a patch rather than a copy so that what I changed is separable
from what was already there.

- `models/loss.py` — adds `mask_dice_weighted_loss`. The plain DICE loss documents that it
  ignores its `sample_weight` argument, and every existing caller passes one, so honouring it
  there would have silently changed the objective of every configuration in the library. Opting
  in under a new name cannot. Under a uniform weight it reduces exactly to the original, which
  the tests assert.
- `models/task.py` — adds `constituent_weight_field` to `ObjectHitMaskTask`, so the mask loss
  can be weighted per cell. It is not used by either current config; `git log` has the overlay that did.
- `callbacks/prediction_writer.py` — an `output_name` argument. The writer named its output
  after the data directory, and all three splits live in one directory here, so every split
  overwrote the same file.

The reference version is `hepattn` at commit `30ccb9f`. To reproduce the environment:

```bash
git clone https://github.com/samvanstroud/hepattn && cd hepattn && git checkout 30ccb9f
git apply /path/to/MSc_Thesis_SDIC/src/maskformer/hepattn-changes.patch
```

then copy the contents of `hepattn_colliderml/` over `src/hepattn/experiments/colliderml/`.

## Layout

```
hepattn_colliderml/          verbatim copy of hepattn/src/hepattn/experiments/colliderml/
  data.py                    ColliderMLDataset / ColliderMLDataModule: reads the raw parquet
                             shards, applies the cell and particle selections, and builds the
                             truth targets -- including the shower-level collapse. The only
                             file here with no hepattn import.
  model.py                   ColliderMLModel, a thin LightningModule wrapper. The architecture
                             itself is assembled from the config, not from this file.
  main.py                    CLI entry point (Lightning's LightningCLI).
  configs/
    pu0.yaml                 pileup 0, full detector. Self-contained.
    pu200.yaml               pileup 200, barrel only. Self-contained.
  eval/
    dump.py                  Runs the model over an event window and writes an event store.
                             The only GPU-dependent step of the analysis.
    format.py                The on-disk store format. Mirrored by src/io/event_store.py.
    geometry.py              Calorimeter layer geometry: recovers 48 ECAL and 36 HCAL layers by
                             projecting barrel cells onto the stave normal.

ce_ai_1/                     Launchers for ce-ai-1, the machine everything now runs on, and the
                             record of what pileup-200 forced to change. NOT part of the mirror --
                             these are mine, so keeping them out of hepattn_colliderml/ is what
                             lets verify_sync.sh pass. Start at ce_ai_1/README.md.
hepattn-changes.patch        My modifications to the upstream library.
verify_sync.sh               Checks hepattn_colliderml/ against a hepattn checkout.
```

**There are exactly two configs, one per pileup condition, and neither is an overlay on the
other.** That replaced a stack of nineteen overlay files on 2026-08-12. Overlays meant the
objective a run actually used depended on the ORDER of several `--config` flags, and because
`tasks` is a YAML list, any overlay touching it replaced the whole list rather than merging into
it — which is how one pu0 run ended up on a different mask objective from the one its base config
documented. The two files are identical except where the data forces a difference, and those
places are marked `PU200 DIFFERS` / `PU0 DIFFERS`; `diff configs/pu0.yaml configs/pu200.yaml` is
the intended way to see what pileup changes.

The sweep, variant and probe overlays that used to live here, the DIAS launchers, and the
`mask_variants.py` / `affinity.py` modules they named were deleted in the same pass.
`git log -- src/maskformer/` recovers all of them with the measurements recorded alongside.

## The checkpoint — and why there is no longer one here

**No checkpoint is tracked in this repository.** One was: a 41.7 MB
`epoch=003-val_loss=18.06191-weights.ckpt`, stripped of optimiser state, alongside its resolved
`config.yaml` and a `metadata.yaml` of git commit, job id, GPU and Comet URL. It was deleted on
2026-08-11 for two reasons that compounded:

- **it was trained on an objective the thesis no longer uses** — dice 5 + focal 20 with an
  incidence head at `kl_div` weight 100 — so it could not produce any reported number, and an
  assessor finding it beside the current configs would reasonably assume it could;
- **it was 90% of the repository's tracked bytes**, which is a poor trade for a file that
  reproduces nothing.

`git log -- src/maskformer/checkpoint/` recovers all three files with their provenance intact.

**Where checkpoints actually live.** Training writes them to
`external/hepattn/src/hepattn/experiments/colliderml/logs/<run>/ckpts/`, which is gitignored and
machine-local. Each run also writes its own fully-resolved `config.yaml` beside them — that file,
not `configs/pu0.yaml`, is the authoritative record of what was trained, because the
configs move on. Comet holds the training curves; the run URL is printed at launch and appears in
the log.

A checkpoint is needed only to dump an event store. Nothing in `scripts/` touches one, so every
table and figure regenerates from the stores alone.

## Running it

Nothing below is needed to reproduce the thesis figures. That is `python -m scripts.score` then
`python -m scripts.make_thesis_figures` from the repository root, and neither touches any of this.

Everything runs on **ce-ai-1** now: no Slurm, no container, no pixi. `ce_ai_1/env.sh` re-syncs
this directory into the hepattn checkout on every launch, because `main.py` runs from the
checkout and editing a config here without copying it across silently uses the stale copy.

**Train.**

```bash
cd src/maskformer/ce_ai_1
nohup ./train.sh pu0 > ../../../external/train_pu0.log 2>&1 &
```

`NUM_TRAIN` and `MAX_EPOCHS` override the config for smoke tests; `CKPT=<path>` resumes. For
pu200, size `max_epochs` from ~300 steps of a real run before committing — OneCycleLR is sized
from total optimiser steps and cannot be resized mid-run, so a schedule that overruns its
walltime never reaches its decay phase and its final checkpoint sits at a high learning rate.

**Produce an event store.** ~2 minutes for the 50-event tuning window at pu0, ~25 for the
500-event evaluation window, and about 360 kB per event.

```bash
CKPT=<a checkpoint under external/hepattn/.../logs/*/ckpts/> ./dump_store.sh pu0 tune
CKPT=<same> ./dump_store.sh pu0 eval
```

Then point `dataset.pu0.store` and `.tune_store` in `config/experiment.yaml` at the results, and
everything from `scripts/` works with no GPU.

## What is deliberately not here

- **The raw dataset** (991 GB) and the **training logs** (9.9 GB). The store in `results/` and
  the figures are the reproducible artefacts.
- **The hit-filter stage** (`calo_hit_filter.yaml` and friends). A two-stage variant that is
  not used by any reported result — the reported checkpoint runs with filtering off. It was
  measured at AUC 0.80, too weak to threshold on, because the hits it rejects are real deposits
  from particles just below the pT cut rather than noise. Worth revisiting at pu200, where the
  population it would reject is a different thing entirely.
- **Dynamic query initialisation**, likewise not used by the reported checkpoint
  (`dynamic_queries: false`).
- **The sweep, variant and probe configurations.** Nineteen overlay files, the `mask_variants.py`
  and `affinity.py` modules two of them named, and the DIAS launchers that ran them, all deleted
  on 2026-08-12 in favour of one config per pileup condition. They are the record of what was
  tried and did not work, so they are worth finding rather than reconstructing:
  `git log -- src/maskformer/` recovers every one with the measurements written beside it.
- **`hepattn` itself**, for the authorship reason above.
