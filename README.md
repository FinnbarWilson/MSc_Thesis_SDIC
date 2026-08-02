# Calorimeter clustering: a classical baseline and a learned model

Code for an MSc thesis comparing the classical CLUE clustering algorithm against
a MaskFormer set-prediction model on high-granularity calorimeter data from the
ColliderML dataset.

The point of the comparison is that it is controlled. Both methods read the same
events, are given the same cells, are asked to reconstruct the same particles,
and are scored by the same code. Every decision the two methods share lives in
`config/experiment.yaml`, so changing a threshold changes it for both or for
neither.

## Layout

```
config/experiment.yaml   every shared decision: splits, cuts, thresholds, seeds
src/config.py            loads and validates the experiment definition
src/data/                reading events, event and particle selection, the truth record
src/clue/                the CLUE baseline and its hyperparameter search
src/evaluation/          matching, metrics and jets - shared by both methods
src/plotting/            figure generation
scripts/                 command line entry points
results/                 JSON and parquet output
figures/                 generated figures
```

`src/evaluation/` never learns which method produced a clustering. It takes a
label per cell plus the truth record and returns numbers, so both pipelines are
scored by identical code rather than by two implementations that are meant to
agree.

## Installation

```bash
conda env create -f environment.yml
conda activate calo-clustering
```

CLUEstering compiles a CPU backend on install. GPU backends need CUDA or ROCm
present at build time; without them `clue.backend` must stay on a CPU setting.

## Data

The ColliderML calorimeter and particle files are not in this repository. Set
their paths in `config/experiment.yaml` under `dataset.pu0` or `dataset.pu200`,
then choose which sample to run on with `dataset.active`.

The dataset is available from
[HuggingFace](https://huggingface.co/datasets/CERN/ColliderML-Release-1).

## Running

All scripts run from the repository root as modules, so there is no installation
step:

```bash
python -m scripts.run_truth_study     # characterise the task before clustering
python -m scripts.tune_clue           # Optuna search, one study per detector
python -m scripts.evaluate_clue       # score the tuned baseline on the test split
```

Run them in that order the first time: `evaluate_clue` reads the parameter file
that `tune_clue` writes.

## The experiment definition

`config/experiment.yaml` is the file to read first. It sets:

- **which dataset** is active, pileup-free or pileup-200
- **three disjoint event splits**, with the test split reserved for reported
  numbers and seen by neither method during tuning or training
- **event selection**, which drops the near-empty events present in the pu0
  release
- **hit selection**, the readout threshold applied before either method sees a
  cell
- **the reconstructable particle definition**, used simultaneously as the
  MaskFormer's targets and as the denominator of CLUE's efficiency
- **the metric definitions**, including how overlapping assignments are handled
- **the jet definition**, radius, match cone and acceptance

Settings below the marked line belong to one method only, and are properties of
that method rather than of the experiment.

## Metrics

Purity and efficiency are energy-weighted rather than cell-counted, because a
calorimeter measures energy and not occupancy: an algorithm can discard most of
a particle's energy while still recovering most of its cells.

Efficiency is taken per truth particle rather than per cluster. Scoring from the
cluster side hides failures, since a particle merged into a neighbour dominates
no cluster and disappears from the denominator instead of counting as a miss.

Both are pooled across events rather than averaged per event, so every particle
carries equal weight regardless of how busy its event was.

The same definitions serve tuning and reporting, so the reported numbers are the
ones the baseline was optimised toward.

## Choice of CLUE configuration

The pipeline runs CLUE twice, first within each detector layer and then over the
resulting layer centroids, which is the established strategy for high-granularity
calorimeters. Neither pass modifies the algorithm: the parameters are CLUE's own
density radius, density threshold and linking radius, and the periodic distance
metric used in angular coordinates is a CLUEstering feature.

`z_scale` divides the layer depth before the three-dimensional pass. It is a unit
conversion rather than a tuning device: without it the depth axis is in
millimetres while the two angular axes are of order one, and no single density
radius can be meaningful in both.
