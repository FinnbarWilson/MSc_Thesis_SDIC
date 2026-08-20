# `src/maskformer/`: training the model and dumping the event store

The GPU half of the project: the model configuration, the training launchers, and the dump that runs a trained checkpoint over a window of events and writes the event store the rest of the analysis reads.

You are here for one of two reasons:

- **To dump an event store** from a checkpoint you fetched. That is step 4 of [Path 2](../../README.md#path-2-rerun-the-methods) in the root README. Skip to [step 3](#3-dump-the-event-stores) below.
- **To train the model**, which is [Path 3](../../README.md#path-3-full-run). Steps 1 to 3 in order, once per pileup condition.

Neither is needed to redraw the report's figures, which is Path 1.

## Running it

Assumes `setup/install_training_env.sh` has been run and the dataset downloaded, which are steps 1 and 2 of Path 2. Under Slurm, submit from the repository root: the launchers take the repository location from `SLURM_SUBMIT_DIR`, and the `#SBATCH` log paths are relative to it.

```bash
mkdir -p external/slurm_logs
```

### 1. Measure the throughput

```bash
NUM_TRAIN=600 MAX_EPOCHS=1 sbatch src/maskformer/dias/train.sh          # DATASET=pu200 for the other
```

Takes about 15 minutes and prints the `max_epochs` that fills the walltime. Do not skip it. OneCycleLR is sized from the total step count, so a wrong estimate does not just give a run of the wrong length, it gives a run whose final checkpoint sits at a high learning rate.

### 2. Set the schedule and train

Put the printed number into `trainer.max_epochs` in `hepattn_colliderml/configs/pu0.yaml`, then:

```bash
sbatch src/maskformer/dias/train.sh
```

Checkpoints land in `external/hepattn/src/hepattn/experiments/colliderml/logs/<run>/ckpts/`. A day or more per condition.

### 3. Dump the event stores

The step that runs the model. It needs a GPU whether the checkpoint was trained here or fetched.

```bash
# absolute, because the launchers cd into the hepattn checkout before running
CKPT=$(python -c "from pathlib import Path; from src.config import settings_for; print(Path(settings_for('pu0')['maskformer']['checkpoint']).resolve())")

CKPT=$CKPT sbatch src/maskformer/dias/dump_store.sh pu0 eval
CKPT=$CKPT sbatch src/maskformer/dias/dump_store.sh pu0 tune     # only for Path 3
```

Name the condition explicitly, as `settings_for('pu0')` does here: plain `settings()` returns whichever condition `dataset.active` happens to be, and the two have different checkpoints. If you trained your own, point `CKPT` at that file instead.

The evaluation store is what the results are reported over. The tuning store is a separate 50-event window, needed only to re-derive CLUE's parameters or the MaskFormer working point, both of which are already committed. Both land in `external/eventstores/`, which is where `config/experiment.yaml` already looks. Lower `CHUNK` if the dump is killed for memory.

### 4. Back to the root README

```bash
python -m scripts.show_config          # confirm it found the new store
```

Then Path 2 step 5, to score the store and rebuild the figures. Repeat all of the above for the other pileup condition.

### On a machine without a scheduler

`src/maskformer/ce_ai_1/` runs the same steps in the foreground:

```bash
cd src/maskformer/ce_ai_1
NUM_TRAIN=600 MAX_EPOCHS=1 ./train.sh pu0
nohup ./train.sh pu0 > ../../../external/train_pu0.log 2>&1 &
CKPT=<absolute path> ./dump_store.sh pu0 eval
```

### Overrides

Set these as environment variables at launch. The full list is in the header comment of each script.

The two launcher sets differ in how they take the pileup condition. `dias/train.sh` reads `DATASET` (default `pu0`) because Slurm passes no arguments; every `ce_ai_1/` script and `dias/dump_store.sh` takes it as the first positional argument, as shown above.

| variable | applies to | effect |
|---|---|---|
| `DATASET` | `dias/train.sh` | `pu0` or `pu200` |
| `NUM_TRAIN`, `MAX_EPOCHS` | `train.sh` | override the config, for smoke tests |
| `BATCH_SIZE`, `WORKERS` | `train.sh` | override the config; read the batch-size note in `configs/pu0.yaml` first |
| `CKPT` | both | checkpoint to resume from, or to dump from |
| `DATA_DIR` | `train.sh` | where `ttbar_<dataset>/` lives (default `external/ColliderML_data`) |
| `STORE_ROOT` | `dias/dump_store.sh` | where the dump writes (default `external/eventstores`) |
| `OUT` | `ce_ai_1/dump_store.sh` | the same, under a different name |
| `CHUNK` | `dump_store.sh` | events held in memory per chunk file (25 at pu0, 10 at pu200) |
| `COMET_API_KEY` | `train.sh` | read from the environment; the run warns and continues without it |

## What is here

```
hepattn_colliderml/          copy of hepattn/src/hepattn/experiments/colliderml/
  data.py                    reads the raw parquet, applies the selections, builds the truth
                             targets including the shower-level collapse
  model.py                   a thin LightningModule wrapper
  main.py                    CLI entry point
  configs/pu0.yaml           pileup 0, full detector. Self-contained
  configs/pu200.yaml         pileup 200, barrel only. Self-contained
  eval/dump.py               runs the model over an event window and writes an event store
  eval/format.py             the on-disk store format, mirrored by src/io/event_store.py
  eval/geometry.py           recovers 48 ECAL and 36 HCAL layers from cell positions
  eval/bench_maskformer.py   per-event inference timing, the counterpart to scripts.bench_clue

dias/                        Slurm launchers, where pu0 was trained and scored
ce_ai_1/                     launchers for a single machine, where pu200 was trained and scored
hepattn-changes.patch        my modifications to the upstream library
verify_sync.sh               checks hepattn_colliderml/ against a hepattn checkout
```

Everything under `hepattn_colliderml/` is a copy of the experiment directory inside the `hepattn` checkout, which is where it actually runs. Both `env.sh` files re-copy it on every launch, because editing a config here and launching without that copy would silently use the stale one. The launchers sit outside that directory because they are mine and would make `verify_sync.sh` fail against a clean upstream.

There is deliberately no `__init__.py` at this level: `import src.maskformer` should not be possible. `src/io/event_store.py` is a hand-written mirror of `eval/format.py` rather than an import of it, so if one side changes the format the other fails on a version check instead of silently misreading.

There are exactly two configs, one per condition, and neither is an overlay on the other. `tasks` is a YAML list, so any overlay touching it replaced the whole list rather than merging into it, which made the objective a run used depend on the order of its `--config` flags. The two files are identical except where the data forces a difference, and those places are marked `PU200 DIFFERS` and `PU0 DIFFERS`. `diff configs/pu0.yaml configs/pu200.yaml` shows what pileup changes.

## Adapting it to another cluster

No path needs editing; `setup/paths.sh` derives every location from the repository root. Set `COLLIDERML_DATA` and `CALO_STORE_ROOT` to move the dataset and the stores off it.

What will not carry over is the machine configuration. `dias/` targets RHEL7 with Slurm, so it runs everything inside an Apptainer container built from `docker://ubuntu:22.04`, pins `CC`/`CXX` at the conda-forge GCC because Triton shells out to a compiler during driver initialisation, and asks for two GPUs so that `env.sh`'s `select_gpu` can avoid a card reporting uncorrected ECC errors. `ce_ai_1/` needs none of that. Read whichever is closer and expect to change the `#SBATCH` resource lines, the partition name and the container.

The `--mem` values look far too large for the work being done and are not negotiable on that cluster: Slurm applies `VSizeFactor` there, making `--mem` a hard virtual memory cap of 1.1x the request, while `expandable_segments` reserves a large virtual range at start-up. An under-request surfaces as `CUDA driver error: out of memory` on an idle 80 GB card.

Comet logging in `configs/*.yaml` points at a workspace you will not have access to. Change `workspace` or set `online: false`.

## Authorship

The model is built from [`hepattn`](https://github.com/samvanstroud/hepattn), an existing library for transformer-based reconstruction in HEP. I did not write it and it is not reproduced here. `hepattn` gained a ColliderML experiment upstream at commit `cb4fb10`, so this directory is a mixture. Checked file by file against that commit:

| file | status |
|---|---|
| `configs/pu0.yaml`, `configs/pu200.yaml` | mine, not upstream |
| `eval/dump.py`, `eval/format.py`, `eval/geometry.py`, `eval/bench_maskformer.py` | mine, not upstream |
| `data.py` | upstream file, heavily modified (889 lines differ): the shower-level truth collapse, the cell and particle selections, the calo association builders |
| `model.py`, `main.py` | upstream, unmodified, byte-identical to `cb4fb10` |
| `hepattn-changes.patch` | mine: three modifications to the library proper |

Not mine either way: `hepattn` itself, meaning the transformer encoder, the MaskFormer decoder, the task heads, the loss functions and the Hungarian matcher.

`hepattn-changes.patch` is kept as a patch rather than a copy so that what I changed stays separable:

- `models/loss.py` adds `mask_dice_weighted_loss`. The plain DICE loss documents that it ignores its `sample_weight` argument and every existing caller passes one, so honouring it there would have silently changed the objective of every configuration in the library. Under a uniform weight the new function reduces exactly to the original.
- `models/task.py` adds `constituent_weight_field` to `ObjectHitMaskTask`, so the mask loss can be weighted per cell.
- `callbacks/prediction_writer.py` adds an `output_name` argument. The writer named its output after the data directory, and all three splits live in one directory here, so every split overwrote the same file.

Check the copies against a checkout with `HEPATTN=/path/to/hepattn ./verify_sync.sh`.
