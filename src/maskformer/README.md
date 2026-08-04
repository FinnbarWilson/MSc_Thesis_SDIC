# `src/maskformer/` — the learned model: training it, and producing the event store

This directory holds the MaskFormer half of the comparison: the dataset, the model
configuration, the trained weights, and the evaluation dump that turns them into an event
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

| | |
|---|---|
| Mine | everything in this directory: the dataset, the model configuration, the evaluation dump, the store format, and `hepattn-changes.patch` |
| Not mine | `hepattn` itself — the transformer encoder, the MaskFormer decoder, the task heads, the loss functions, the Hungarian matcher |

`hepattn-changes.patch` is the exception worth reading: three small modifications I made to
the upstream library, kept as a patch rather than a copy so that what I changed is separable
from what was already there.

- `models/loss.py` — adds `mask_dice_weighted_loss`. The plain DICE loss documents that it
  ignores its `sample_weight` argument, and every existing caller passes one, so honouring it
  there would have silently changed the objective of every configuration in the library. Opting
  in under a new name cannot. Under a uniform weight it reduces exactly to the original, which
  the tests assert.
- `models/task.py` — adds `constituent_weight_field` to `ObjectHitMaskTask`, so the mask loss
  can be weighted per cell. See `hepattn_colliderml/configs/overlay_metric_aligned.yaml` for why.
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
                             truth targets. The only file here with no hepattn import.
  model.py                   ColliderMLModel, a thin LightningModule wrapper. The architecture
                             itself is assembled from the config, not from this file.
  main.py                    CLI entry point (Lightning's LightningCLI).
  configs/
    calo_clustering.yaml     The training configuration, and the primary document of what the
                             model IS: input features, encoder, decoder, all three task heads,
                             losses and their weights, the optimiser and schedule. Heavily
                             commented with the measurements behind each choice.
    overlay_metric_aligned.yaml   Objective changes: energy-weighted mask loss, exclusive mask
                             target. Applied on top of the above.
    overlay_long_schedule.yaml    A longer training schedule, and the step arithmetic behind it.
  eval/
    dump.py                  Runs the model over an event window and writes an event store.
                             The only GPU-dependent step of the analysis.
    format.py                The on-disk store format. Mirrored by src/io/event_store.py.
    geometry.py              Calorimeter layer geometry: recovers 48 ECAL and 36 HCAL layers by
                             projecting barrel cells onto the stave normal.
  scripts/
    sweep_pred_threshold.py  The post-hoc threshold sweep that chose the object threshold of
                             0.2 over the implicit argmax default.
  slurm/                     Job scripts for the cluster this ran on. Both check GPU ECC health
                             first: one card on the node has ~818 uncorrected errors and kills
                             long jobs minutes in.

ce_ai_1/                     Launchers for ce-ai-1, the machine the pu200 work runs on, and the
                             record of what pileup-200 forced to change. NOT part of the mirror --
                             these are mine, so keeping them out of hepattn_colliderml/ is what
                             lets verify_sync.sh still pass. Start at ce_ai_1/README.md.
checkpoint/                  The trained weights the reported results come from, plus provenance.
hepattn-changes.patch        My modifications to the upstream library.
verify_sync.sh               Checks hepattn_colliderml/ against a hepattn checkout.
```

The slurm scripts hardcode `/home/xucapfwi/hepattn/src/hepattn/experiments/colliderml` as
their working directory, because that is where they run — Slurm copies a submitted script
into a spool directory, so deriving the repository root from `$BASH_SOURCE` resolves to `/`.
Moving this directory does not affect them, and should not: they address the checkout, not
the mirror.

## The checkpoint

`checkpoint/epoch=003-val_loss=18.06191-weights.ckpt` is the model every MaskFormer number in
the thesis comes from. 11.6 M parameters, trained on events `[0, 20000)`, evaluated on
`[20250, 20750)` — disjoint, and `src/io/event_store.py` asserts the disjointness rather than
trusting it.

It is a **pileup-0** checkpoint, which matters for the pu200 work: running it on pu200 events
measures how far the model transfers across pileup conditions, and that is a different
question from how the architecture compares to CLUE. `config/experiment.yaml` keeps the
checkpoint under `dataset.pu200.overrides.maskformer` for exactly that reason — so a pu200
run has to name its own rather than inheriting this one by default.

The optimiser state has been stripped (83.3 MB to 41.7 MB). Those are Lion's momentum buffers
and Lightning's loop bookkeeping; they are needed only to *resume* training, never to load the
model. The architecture survives, because Lightning stores it in `hyper_parameters`, so:

```python
from hepattn.experiments.colliderml.model import ColliderMLModel
model = ColliderMLModel.load_from_checkpoint("checkpoint/epoch=003-...-weights.ckpt",
                                             map_location="cpu")
```

reconstructs the full network from this file alone. Verified to give weights byte-identical to
the unstripped original. **It cannot be used to resume training** — retrain from scratch, or
take the original from `logs/` on the cluster.

`checkpoint/config.yaml` is the run's own fully-resolved configuration, written by Lightning at
launch, and is the authoritative record of what was trained — prefer it over
`hepattn_colliderml/configs/calo_clustering.yaml`, which has moved on since.
`checkpoint/metadata.yaml` carries the provenance: git commit, slurm job id, GPU, and the Comet
run URL with the training curves.

## Running it

Nothing below is needed to reproduce the thesis figures. That is `python -m scripts.make_figures`
from the repository root, and it touches none of this.

> **On ce-ai-1, none of the commands in this section apply.** That machine has no Slurm, no
> apptainer and no pixi, and the pu200 run needs different cuts to fit at all. See
> [`ce_ai_1/README.md`](ce_ai_1/README.md), which covers both. What follows is the DIAS setup the
> pu0 checkpoint was produced on, kept because it is the provenance of the reported results.

**Environment.** The cluster is RHEL7 (glibc 2.17) and the environment needs glibc 2.28+, so
everything runs inside an Ubuntu 22.04 container:

```bash
apptainer build ~/ubuntu22.sif docker://ubuntu:22.04
```

**Train.** ~4.9 h per epoch at 20000 events on one A100, measured end to end:

```bash
sbatch src/maskformer/hepattn_colliderml/slurm/calo_clustering.sh
```

Add the overlays to change the objective or the schedule:

```bash
python main.py fit --config configs/calo_clustering.yaml \
                   --config configs/overlay_metric_aligned.yaml \
                   --config configs/overlay_long_schedule.yaml
```

The long schedule does not fit one job's walltime and is meant to be resumed with `CKPT=`;
`overlay_long_schedule.yaml` explains the arithmetic and the trap (a OneCycle schedule sized
for twelve epochs but stopped at four never reaches its decay phase, and its final checkpoint
is taken at a high learning rate).

**Produce an event store.** ~12 minutes for 500 events at pu0, and about 320 kB per event:

```bash
CKPT=src/maskformer/checkpoint/epoch=003-val_loss=18.06191-weights.ckpt \
START=20250 NUM=500 OUT=~/eventstore sbatch \
  src/maskformer/hepattn_colliderml/slurm/calo_dump_eventstore.sh
```

Then point `dataset.pu0.store` in `config/experiment.yaml` at the result, and everything from
there on is numpy.

### Dumping a pileup-200 store

On ce-ai-1 this is `ce_ai_1/dump_store_pu200.sh tune|eval`, which has the three decisions below
already made and explained. The rest of this section is the reasoning behind them.

The same command, with three things to decide first, because none of them has a right default:

- **`OUT` must be a different directory.** The store name encodes the event window, not the
  pileup condition, so two dumps of the same event range would otherwise collide.
- **`CHUNK` is the memory knob.** It defaults to 25 events per `.npz`, sized against pu0 cell
  counts, and the dump holds a chunk in memory while writing it. Lower this first if the job
  is killed for memory, and expect the store to be several times larger per event.
- **`INCIDENCE_TOP_K` is left unset on purpose.** `eval/format.py` carries the measured value
  and is the one place the choice is justified. Setting it here silently overrides that — it
  did once, at 4 against format's 16 — so leave it alone unless the multi-owner study says the
  truncation is binding.

Then set `dataset.pu200.store`, `dataset.pu200.tune_store` and `dataset.pu200.windows` in
`config/experiment.yaml` and check the result with `python -m scripts.show_config`.

## What is deliberately not here

- **The raw dataset** (991 GB) and the **training logs** (9.9 GB). The store in `results/` and
  the figures are the reproducible artefacts.
- **The hit-filter stage** (`calo_hit_filter.yaml` and friends). A two-stage variant that is
  not used by any reported result — the reported checkpoint runs with filtering off. It was
  measured at AUC 0.80, too weak to threshold on, because the hits it rejects are real deposits
  from particles just below the pT cut rather than noise. Worth revisiting at pu200, where the
  population it would reject is a different thing entirely.
- **Dynamic query initialisation** (`overlay_dynamic_queries.yaml`), likewise not used by the
  reported checkpoint (`dynamic_queries: false`).
- **`hepattn` itself**, for the authorship reason above.
