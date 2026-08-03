# Calorimeter clustering: a classical baseline and a learned model

Code for an MSc thesis comparing the classical CLUE clustering algorithm against a
MaskFormer set-prediction model on high-granularity calorimeter data from the ColliderML
dataset.

The point of the comparison is that it is controlled. Both methods read the same events, are
given the same cells, are asked to reconstruct the same particles, and are scored by the same
code.

The mechanism that makes that true is worth stating up front, because it is the main design
decision in the repository. The two methods do not each read the dataset and apply matching
cuts. Instead a single **event store** is dumped once, from the model's own dataloader, and
CLUE clusters the cells in that store. "Both algorithms saw identical input" is therefore
structural rather than a promise two config files make to each other.

A useful side effect: the scoring and plotting code depends on nothing but numpy, scipy,
pandas and matplotlib. No GPU, no hepattn, no access to the 991 GB dataset. The pooled
result tables are small enough to sit beside the thesis, so every figure can be regenerated
from them.

## Layout

```
config/experiment.yaml   every shared decision, and the expectations checked against the store
src/config.py            loads and validates the experiment definition
src/io/event_store.py    reads the event store; validates it against the config
src/clue/                the CLUE baseline and its hyperparameter search
src/evaluation/          matching, metrics and differential binning - shared by both methods
src/plotting/            figure generation
scripts/                 command line entry points
tests/                   scorer identity tests and the CLUE periodic-metric regression
results/                 pooled parquet tables
figures/                 generated figures
```

`src/evaluation/` never learns which method produced a clustering. It takes a label per cell
plus the truth partition and returns numbers, so both pipelines are scored by identical code
rather than by two implementations that are meant to agree.

The producer of the event store lives in the `hepattn` repository, at
`src/hepattn/experiments/colliderml/eval/`. It is the only GPU-dependent part of the
analysis. `src/io/event_store.py` is a deliberate hand-written *mirror* of its format module
rather than an import of it, which is what keeps the dependency boundary where it is.

## Installation

```bash
conda env create -f environment.yml
conda activate calo-clustering
```

Two dependency notes, both learned the hard way on this platform. Boost is pinned as
`libboost-headers`, not `boost`: the CLUE kernels only include headers, while the `boost`
metapackage drags in `libboost-python` and an icu that no longer co-solves with pyarrow 24.
And `scikit-learn` is taken from conda rather than left to pip, because pip has no wheel for
this python/numpy combination and falls back to a source build that fails.

CLUEstering ships CPU backends only here (`cpu serial`, `cpu openmp`); `clue.backend` must
stay on one of those.

## Data

Nothing in this repository opens a raw ColliderML file. It reads an event store produced by:

```bash
apptainer exec --nv --bind /home/xucapfwi/ColliderML_data ~/ubuntu22.sif \
  ~/hepattn/.pixi/envs/default/bin/python -m hepattn.experiments.colliderml.eval.dump \
  <checkpoint> --start-event 20250 --num-events 500 --out ~/eventstore
```

or, on the cluster, via `slurm/calo_dump_eventstore.sh` in the hepattn experiment directory.
A store costs roughly 320 kB per event: about 160 MB for the 500-event evaluation window.

Point `dataset.pu0.store` and `dataset.pu0.tune_store` at the results.

## Running

All scripts run from the repository root as modules:

```bash
python -m scripts.tune_clue                          # Optuna search, one study per subsystem
python -m scripts.score --algo maskformer            # score the model
python -m scripts.score --algo clue --params results/clue_parameters.json
python -m scripts.score --algo oracle_geometric      # reference clusterings; see Metrics
python -m scripts.score --algo oracle_resolution
python -m scripts.score_soft                         # multi-owner capability study
python -m scripts.scan_soft_threshold                # soft metric vs mask threshold
python -m scripts.make_figures                       # every figure, from the tables alone
```

`make_figures` is the only one an assessor needs; it touches no checkpoint and no dataset.

## The experiment definition

`config/experiment.yaml` is the file to read first. Note the split in what it does. The cuts
that decide which cells and which particles exist are **not applied** here -- they were
applied once, by the dump, and travel inside the store as metadata. What the config holds are
the *expectations*, and `EventStore` refuses to open a store that disagrees with them. A
config that has drifted fails loudly at load rather than quietly producing numbers for a
different experiment.

## Metrics

Purity and efficiency are energy-weighted rather than cell-counted, because a calorimeter
measures energy and not occupancy: an algorithm can discard most of a particle's energy while
still recovering most of its cells. Energies are calibrated per cell by subsystem first --
ECAL and HCAL are calibrated differently, so the factor does not cancel in a ratio.

Efficiency is taken per truth particle rather than per cluster. Scoring from the cluster side
hides failures, since a particle merged into a neighbour dominates no cluster and disappears
from the denominator instead of counting as a miss. Both are pooled across events rather than
averaged per event, so every particle carries equal weight regardless of how busy its event
was.

Truth particles and predicted clusters are matched by a **global one-to-one assignment**
(`scipy.optimize.linear_sum_assignment`) on shared calibrated energy. The metric the model
logs during training cannot be reused: it compares query *i* with target *i*, which is only
meaningful because the training loss has already permuted them onto each other, and CLUE has
no such permutation.

Truth is the **exclusive** partition -- each cell belongs to the particle that deposited the
most energy in it. CLUE produces a partition and cannot do otherwise, so scoring it against a
multi-owner truth would be structurally unfair. The exclusive partition is nearly lossless:
it keeps ~83% of (particle, cell) associations, and those carry ~94% of each particle's own
energy.

### What the numbers are measured against

An efficiency of 0.31 cannot be read against 1.0, because 1.0 is not available. Feeding the
truth partition back in as a prediction scores exactly 1 by construction, so "perfect" says
nothing; a meaningful reference has to constrain what an algorithm *knows*. Two are built in
`src/evaluation/oracle.py` and scored through the same code path as the real methods:

- **`oracle_geometric`** -- an idealised method handed the true particle count and the true
  shower axis of each, assigning every cell to its nearest axis in **angle and depth**. Its
  one free parameter, the relative weight of depth, is scanned and maximised rather than
  guessed: a ceiling left at an arbitrary setting is one arbitrary algorithm, not a bound.
  Depth is worth more than expected, lifting it from 0.530 to 0.591 mean efficiency, which is
  why an angle-only version was an unfair reference for a depth-aware baseline like CLUE.

  This is the ceiling for **spatial clustering as a class** -- the class CLUE belongs to and
  the MaskFormer does not -- which gives it a use no aggregate has: whether a learned model
  exceeds what perfect-seeded geometry can achieve. It remains optimistic in knowing the
  seeds, and that caveat belongs wherever it is quoted.
- **`oracle_resolution`** -- target particles that share too much of each other's energy to
  be separable are merged, then clustered perfectly. Efficiency is ~0.97 by construction, so
  the number to read is its **purity of ~0.50**, which is a genuine ceiling. Only ~5% of
  target particles fall into a multi-particle group, so that ceiling is set overwhelmingly by
  sub-threshold contamination -- 46% of the calorimeter energy comes from particles below the
  pT cut and has to land in some cluster -- rather than by target particles overlapping.

Two things the references settle that the raw numbers could not:

- **Efficiency falling with particle energy is a property of the task, not a defect.** The
  geometric ceiling falls the same way, and above about 20 GeV CLUE meets it. Energetic
  particles here sit in dense jet cores, where even an idealised assignment given the true
  axes cannot keep them apart. The same applies to purity collapsing above ~10 GeV of cluster
  energy: the ceiling collapses with it.
- **The headroom is in the isolated regime, not the crowded one.** For particles with no
  neighbour within dR = 0.2 the ceiling is far above both methods; inside a jet core they are
  much closer to it. So the two methods are furthest from what is achievable exactly where
  the problem is easiest, which is where effort is best spent.

## The multi-owner capability study

`scripts/score_soft.py` scores the same events with ownership left **fractional on both
sides** -- the truth's real multi-owner incidence, and MaskFormer's overlapping masks divided
between queries in proportion to their mask probabilities. CLUE passes through the identical
code path with every weight equal to 1, since a partition is just the degenerate case.

This exists because the head-to-head collapses MaskFormer's overlapping masks to one winner
per cell so CLUE has something it can express, which means the main comparison measures the
model with its distinguishing capability switched off. That is right for the head-to-head and
wrong as the last word.

Read it with two things in mind:

- **CLUE's efficiency is lower here, and that is not a penalty.** The denominator changes from
  "energy in cells this particle dominates" -- a target defined by what a partition can
  express -- to the particle's actual deposited energy, which neither method's output can
  move. Scoring the truth partition itself under this metric gives exactly `exclusive_share`
  rather than 1, so the shortfall is a property of the algorithm class.
- **The headline is the tail, not the aggregate.** About 0.4% of target particles own no cell
  exclusively; no partitioning method can reach them at all, and the bar for CLUE is zero by
  construction rather than by tuning. What a mask-based method recovers there is capability
  the baseline does not have.
- **A mask probability is not a trained energy fraction, and that is measurable.** The model
  divides each cell 2.08 ways against the truth's 1.22, so normalisation splits its claims
  further than the energy really splits and its soft efficiency falls below its exclusive one.
  `scripts/scan_soft_threshold.py` establishes this is not a working-point problem: efficiency
  falls monotonically as the mask threshold rises (0.350 at the store's 0.02 floor, 0.322 at
  the nominal 0.5, 0.300 at 0.95), and even at 0.95 the model still divides 1.85 ways. The
  overlap is confident, not marginal.

  This follows from the architecture rather than from tuning: the mask head applies an
  element-wise sigmoid per (query, cell) rather than a softmax over queries, so nothing in the
  loss constrains one cell's claims to sum to anything and the model is never taught what a
  cell's division should add up to. Read the shortfall as a statement about mask *calibration*,
  not about whether the architecture can represent shared cells -- it plainly can, which is
  what the recovery of otherwise-unreachable particles shows. Quote `sharing_diagnostics`
  alongside any soft efficiency.

### Caveats worth knowing before reading any number

- About **0.4% of target particles own no cell exclusively** -- every cell they touched was
  dominated by someone else. No exclusive-partition algorithm can recover them, so they are a
  ceiling on efficiency. They are reported as unmatched rather than quietly dropped.
- The split-rate definition (more than one cluster holding at least 10% of a particle) is
  **blind to severe fragmentation**: a particle spread over more than ten clusters gives none
  of them 10%, so it registers as unsplit, and 57% of particles here have more than ten
  cells. `frag_frac`, the share of a particle outside its largest piece, has no such blind
  spot and is the one to read.
- Splitting and merging are computed in **both weightings** and the energy one is primary,
  matching every other metric here. The difference is asymmetric: merging is barely affected,
  but above ~8 GeV the two definitions of *splitting* disagree on the sign of MaskFormer's
  trend -- hit-counted falls with particle energy (0.36 to 0.16) while energy-weighted rises
  (to 0.53). The hit-counted fall is the `n_frag` blind spot biting hardest on the most
  fragmented particles, so reporting it would have supported the opposite conclusion.
  `figures/weighting_comparison.pdf` shows both definitions side by side.
- Matching applies a **relative floor** (`metrics.min_overlap_frac`), so a pair must share at
  least 5% of the smaller of the two totals. Without it a single shared 1 MeV cell counted as
  a match, which made the fake rate close to vacuous; applying it roughly triples the measured
  fake rate.
- Error bars on fractions come from resampling **events**, not particles. The ~620 particles
  in an event share cells and occupancy and are not independent trials, so binomial intervals
  over the pooled table are too narrow. Clopper-Pearson is kept as a fallback for bins holding
  fewer than three events.

## Choice of CLUE configuration

The pipeline runs CLUE twice, first within each detector layer and then over the resulting
layer centroids, which is the established strategy for high-granularity calorimeters. Neither
pass modifies the algorithm: the parameters are CLUE's own density radius, density threshold
and linking radius, and the periodic distance metric used in angular coordinates is a
CLUEstering feature.

**What a layer is.** The endcaps are planes of constant |z|. The barrels are 16-fold
polygonal, built from flat staves, so one layer spans a range of radii; projecting onto the
stave normal collapses each plate back to a single depth. Pooling many events and splitting
where consecutive depths differ by more than 1 mm recovers 48 ECAL layers at 5.05 mm pitch
and 36 HCAL layers at 51.0 mm, identically in barrel and endcap. The uniform pitch is the
check that the projection is right: a wrong stave count gives ragged, unphysical spacings.
The geometry is calibrated once and frozen into the store, because a sparse subsystem leaves
most of its layers unlit in any single event.

**`depth_scale`** divides the layer *index*, not a physical depth. Using the index keeps the
depth axis meaning the same thing in every subsystem, which a length cannot: ECAL layers are
5.05 mm apart and HCAL layers 51 mm, so one linking radius in metres would be ten times more
permissive in one than the other.

**Periodic phi needs two things, not one.** `choose_metric("periodic_euclidean")` changes the
distance function, but CLUE first bins points into a tile grid to decide which pairs to
compare at all, and that grid wraps only when `wrapped_coords` says so. With the metric alone
a shower straddling +/-pi still comes back as two clusters. `tests/test_clue_periodic.py`
pins both halves, including the negative case, so the call cannot be "simplified" back.

## Jets

Deferred. 46% of the calorimeter energy comes from particles below the 0.5 GeV target cut, so
MaskFormer jets would be missing that energy by construction while CLUE's would not, and the
resulting plot would measure the target definition rather than either algorithm. The
decisions already taken are recorded under `jets:` in the config so they are not relitigated.
