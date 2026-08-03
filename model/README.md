# The learned model: training it, and producing the event store

This directory holds the MaskFormer half of the comparison. Everything else in this
repository is deliberately numpy-only and runs on a laptop; this part is not, and the
separation is the point rather than an inconvenience.

Read `../README.md` first for what the comparison is and how the two methods are scored. This
file covers only how the model is defined, trained, and turned into an event store.

## Why this is not under `src/`

`src/` is the scoring and plotting code, and its defining property is that it imports nothing
but numpy, scipy, pandas and matplotlib. That is what lets an assessor regenerate every figure
in the thesis from the tables in `results/` with no GPU, no `hepattn`, and no access to the
991 GB ColliderML dataset.

The code here needs all three. Putting it in `src/` would quietly destroy that property, so it
sits outside, and nothing in `src/` imports anything from this directory. The two halves meet
only at the **event store**: this side writes one, `src/io/event_store.py` reads it.

```
  model/  ──(writes)──>  event store (~590 MB, plain .npz)  ──(reads)──>  src/
  GPU, hepattn, dataset                                            numpy and nothing else
```

`src/io/event_store.py` is a hand-written *mirror* of `eval/format.py`, not an import of it.
Two readers of the same format, maintained side by side, is the mechanism that keeps the
boundary honest: if this side changes the format, the other side fails loudly on a version
check rather than silently misreading the file.

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
  can be weighted per cell. See `configs/overlay_metric_aligned.yaml` for why.
- `callbacks/prediction_writer.py` — an `output_name` argument. The writer named its output
  after the data directory, and all three splits live in one directory here, so every split
  overwrote the same file.

The reference version is `hepattn` at commit `30ccb9f`. To reproduce the environment:

```bash
git clone https://github.com/samvanstroud/hepattn && cd hepattn && git checkout 30ccb9f
git apply /path/to/MSc_Thesis_SDIC/model/hepattn-changes.patch
```

then copy this directory's contents over `src/hepattn/experiments/colliderml/`. The files here
are verbatim copies of the ones that produced the reported results; `verify_sync.sh` diffs them
against a `hepattn` checkout so the claim is checkable rather than asserted.

## Layout

```
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
checkpoint/                The trained weights the reported results come from, plus provenance.
hepattn-changes.patch      My modifications to the upstream library.
verify_sync.sh             Checks these copies against a hepattn checkout.
```

## The checkpoint

`checkpoint/epoch=003-val_loss=18.06191-weights.ckpt` is the model every MaskFormer number in
the thesis comes from. 11.6 M parameters, trained on events `[0, 20000)`, evaluated on
`[20250, 20750)` — disjoint, and `src/io/event_store.py` asserts the disjointness rather than
trusting it.

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
`configs/calo_clustering.yaml`, which has moved on since. `checkpoint/metadata.yaml` carries
the provenance: git commit, slurm job id, GPU, and the Comet run URL with the training curves.

## Running it

Nothing below is needed to reproduce the thesis figures. That is `python -m scripts.make_figures`
from the repository root, and it touches none of this.

**Environment.** The cluster is RHEL7 (glibc 2.17) and the environment needs glibc 2.28+, so
everything runs inside an Ubuntu 22.04 container:

```bash
apptainer build ~/ubuntu22.sif docker://ubuntu:22.04
```

**Train.** ~4.9 h per epoch at 20000 events on one A100, measured end to end:

```bash
sbatch model/slurm/calo_clustering.sh
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

**Produce an event store.** ~12 minutes for 500 events:

```bash
CKPT=model/checkpoint/epoch=003-val_loss=18.06191-weights.ckpt \
START=20250 NUM=500 OUT=~/eventstore sbatch model/slurm/calo_dump_eventstore.sh
```

Then point `dataset.pu0.store` in `config/experiment.yaml` at the result, and everything from
there on is numpy.

## What is deliberately not here

- **The raw dataset** (991 GB) and the **training logs** (9.9 GB). The store in `results/` and
  the figures are the reproducible artefacts.
- **The hit-filter stage** (`calo_hit_filter.yaml` and friends). A two-stage variant that is
  not used by any reported result — the reported checkpoint runs with filtering off. It was
  measured at AUC 0.80, too weak to threshold on, because the hits it rejects are real deposits
  from particles just below the pT cut rather than noise.
- **Dynamic query initialisation** (`overlay_dynamic_queries.yaml`), likewise not used by the
  reported checkpoint (`dynamic_queries: false`).
- **`hepattn` itself**, for the authorship reason above.
