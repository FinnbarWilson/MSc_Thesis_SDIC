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
src/config.py            loads, merges and validates the experiment definition
src/io/event_store.py    reads the event store; validates it against the config
src/clue/                the CLUE baseline and its hyperparameter search
src/evaluation/          matching, metrics and differential binning - shared by both methods
src/plotting/            figure generation
src/maskformer/          the learned model: training config, the trained weights, and the dump
                         that writes an event store. Needs hepattn and a GPU; see below
scripts/                 command line entry points
tests/                   scorer identity tests and the CLUE periodic-metric regression
results/<dataset>/       pooled parquet tables, one directory per pileup condition
figures/<dataset>/       generated figures, likewise
attic/                   superseded code, kept for its decisions and imported by nothing
```

Two things about that layout are load-bearing.

**`src/maskformer/` is the only part that needs a GPU**, and the rest of `src/` imports
nothing but numpy, scipy, pandas and matplotlib. Those two halves are kept apart by nothing
importing across the line, not by which directory they sit in:

```bash
grep -rn "src\.maskformer" --include="*.py" src scripts tests   # returns nothing
```

The model's files import `hepattn.experiments.colliderml.*` and never `src.*`, so neither
side can reach the other even by accident. They meet only at the **event store**, which one
writes and the other reads — and `src/io/event_store.py` is a hand-written *mirror* of the
format module rather than an import of it, so a format change fails loudly on a version check
instead of being silently misread. `src/maskformer/README.md` covers what the model is, how it
was trained, and which parts of it are mine.

**Everything a run writes is scoped by dataset.** `results/pu0/` and `results/pu200/` are
separate directories chosen by `dataset.active`, because the tables are read back by name and
a second pileup condition writing beside the first would replace it — and then `make_figures`
would draw one panel from two experiments with nothing looking wrong.

`src/evaluation/` never learns which method produced a clustering. It takes a label per cell
plus the truth partition and returns numbers, so both pipelines are scored by identical code
rather than by two implementations that are meant to agree.

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

or, on the cluster, via `src/maskformer/hepattn_colliderml/slurm/calo_dump_eventstore.sh`.
A pu0 store costs roughly 320 kB per event: about 160 MB for the 500-event evaluation window.

Point `dataset.pu0.store` and `dataset.pu0.tune_store` at the results.

## Running

All scripts run from the repository root as modules. Every one of them prints which dataset
it resolved to before doing anything, and reads and writes that dataset's directories:

```bash
python -m scripts.show_config                        # what the active dataset resolved to
python -m scripts.tune_clue                          # Optuna search, one study per subsystem
python -m scripts.score --algo maskformer            # score the model, mask head
python -m scripts.score --algo maskformer_incidence  # same model, incidence head; see Two heads
python -m scripts.score --algo clue                  # picks up this dataset's tuned parameters
python -m scripts.score --algo oracle_geometric      # reference clusterings; see Metrics
python -m scripts.score --algo oracle_resolution
python -m scripts.score_soft                         # multi-owner capability study
python -m scripts.scan_working_points                # the efficiency/purity trade-off curve
python -m scripts.scan_soft_threshold                # soft metric vs mask threshold
python -m scripts.make_figures                       # every figure, from the tables alone
```

`make_figures` is the only one an assessor needs; it touches no checkpoint and no dataset.

## Running the other pileup condition

`dataset.active` in `config/experiment.yaml` is the only switch. Setting it to `pu200`
changes four things at once, which is the point — the alternative is remembering four:

- **the stores**, from `dataset.pu200.store` / `.tune_store`
- **the output directories**, to `results/pu200/` and `figures/pu200/`, so nothing can land
  on top of the pu0 tables and no figure can mix the two
- **anything under that dataset's `overrides:` block**, deep-merged over the shared settings
  before any consumer sees them. Everything outside `dataset:` is the value *measured at pu0*;
  a value that has to differ at pu200 goes in the override block rather than being edited in
  place, or the other condition silently changes with it
- **the Optuna study names**, which carry the dataset, so a `--storage` run cannot resume the
  wrong condition's trials and report them as its answer

The order to work in, and what each step is waiting on:

1. **Dump the two stores.** `src/maskformer/README.md` has the command and the three settings
   that need deciding for pu200 (`OUT`, `CHUNK`, and leaving `INCIDENCE_TOP_K` alone).
2. **Fill in `dataset.pu200`** — the two store paths and the four window numbers — then run
   `python -m scripts.show_config` and read it back. The store validates its own metadata
   against the config on open, so a mismatch stops the run rather than shifting the numbers.
3. **Tune CLUE, and read the warnings before the result.** `rho_c` is a local energy density
   and pileup moves the density it is measured over, so the pu0 search ranges are the wrong
   box almost by definition. `tune_subsystem` reports any optimum landing in the outer 5% of
   its log range; widen the named range under `dataset.pu200.overrides.clue.search` and re-run
   until nothing presses against a bound. A baseline tuned in the wrong box is under-tuned,
   which is the one way this comparison can be unfair to CLUE.
   `--events` is the memory knob: every trial re-runs over every tuning record and they are
   all held decoded, so the cost scales with total cells rather than with events.
4. **Re-derive the MaskFormer working point** with `scripts.scan_working_points`. The 0.5/0.2
   pair is a pu0 measurement — the object threshold of 0.2 bought ~25% relative efficiency at
   flat purity *there* — and nothing says it transfers.
5. **Score and plot** exactly as above.

One thing to settle before quoting a head-to-head: **the checkpoint is pileup-0**. Running it
on pu200 measures how far the model transfers across pileup conditions, which is a real
result but a different one from how the architecture compares to CLUE. The config keeps the
checkpoint under `dataset.pu200.overrides.maskformer` so the pu200 run has to name its own
rather than inheriting this one by silence.

## The figures

Eight, each making one point. Styled with
[scienceplots](https://github.com/garrettj403/SciencePlots) (`science` + `no-latex`, so no TeX
installation is needed) on top of `src/plotting/style.py`.

| Figure | The one thing it says |
|---|---|
| `eff_pur_vs_energy` | the head-to-head: efficiency and purity against energy |
| `efficiency_decomposition` | *why* — found more particles, recovered less of each, fewer fakes |
| `performance_vs_density` | the headroom is in isolated particles, not in jet cores |
| `split_and_merge` | the failure mode: fragmentation above the geometric ceiling |
| `energy_decomposition` | the two methods lose energy in different ways |
| `incidence_head_comparison` | the two readings of the model, side by side |
| `working_point_curve` | the efficiency/purity trade-off over each method's working points |
| `weighting_comparison` | methods: counting cells reverses the splitting trend |

Three conventions worth knowing.

**"MaskFormer" means the mask head.** The incidence head is a second *reading* of the same
checkpoint, not a second model, so it appears in exactly one figure. Carrying both rows
through every panel made one model look like two competitors and pushed the CLUE comparison,
which is what the thesis is about, into third place.

**Colours come from scienceplots' own cycle**, with blue and yellow deliberately unused for
the methods: they are the loudest members of it, and blue being the default first colour made
the old figures read as "the blue one is the point". Violet and green carry the two methods,
greys carry the references, and every method keeps its colour and its marker in every figure.

**Panels are labelled (a), (b), (c), not titled.** The caption says what each panel shows, so
the panel only has to be identifiable; a descriptive title duplicates the caption and, on a
two-panel figure, is usually longer than the panel is wide.

Four figures were removed rather than restyled:

- `reference_ceiling` drew the same two panels as `eff_pur_vs_energy` against the same references.
- `fake_and_match_rates` drew the cluster match rate — now panel (c) of `efficiency_decomposition` —
  beside a particle match rate that was already panel (a) of it on a different x-axis.
- `multiowner_capability` and `soft_threshold_scan` were each one number and one flat line
  respectively: 0.56 of otherwise-unreachable particles recovered against CLUE's 0.007, and
  "the over-division survives every threshold". Both are sentences, and both are in this
  README and in `results/pu0/capability_summary.csv`. `scripts.score_soft` and
  `scripts.scan_soft_threshold` still write their tables.

## The experiment definition

`config/experiment.yaml` is the file to read first. Note the split in what it does. The cuts
that decide which cells and which particles exist are **not applied** here -- they were
applied once, by the dump, and travel inside the store as metadata. What the config holds are
the *expectations*, and `EventStore` refuses to open a store that disagrees with them. A
config that has drifted fails loudly at load rather than quietly producing numbers for a
different experiment.

## Two heads, and which one owns a cell

The model makes two predictions about every cell, and which one is read decides what the
head-to-head measures. This is the single most consequential choice in the scoring code, so
it is stated here rather than buried in the metrics section.

The **mask head** emits an independent sigmoid per (query, cell). Nothing in its loss relates
one cell's claims to each other, so its output is a detection score and *not* a share of a
cell -- a fact with a measured consequence: normalising mask probabilities divides each cell
2.04 ways against truth's 1.22, and that over-division survives every mask threshold up to
0.95, so it is a calibration property rather than a working point.

The **incidence head** applies a softmax over queries within a cell and is trained by KL
divergence against `I_ia = E_ia / E_i`, the fraction of that cell's energy belonging to each
particle. It is the head that was taught what a cell's division adds up to. Until the store
format grew a version 2 it was computed on every forward pass and discarded, so every number
reported before then resolved ownership with the quantity that was never supervised to do it.

That gives four ways to turn one checkpoint into clusters, and the repository scores all of
them through the same code:

|                | mask head          | incidence head                 |
|----------------|--------------------|--------------------------------|
| **exclusive**  | `maskformer`       | `maskformer_incidence`         |
| **fractional** | `scripts.score_soft` | `scripts.score_soft` (both rows) |

Two things are worth being clear about, because the change invites a misreading.

**Reading the incidence head is not a step towards hard borders.** The head-to-head already
collapsed to an exclusive partition, and always had to: CLUE produces a partition and cannot
express a shared cell, so a like-for-like comparison has to be run on something both methods
can represent. Only the *rule* that does the collapsing changes, from "highest mask
probability" to "largest predicted share". The soft study, where the fractional capability
actually lives, gains a row scored with the calibrated quantity rather than losing anything.

**Detection stays with the mask head in every variant.** The incidence softmax sums to one
over queries for *every* cell, including the ~63% of hits belonging to no target particle, so
on its own it could never decline anything and purity would collapse. The mask head decides
whether a cell is claimed at all; the incidence head decides whose it is. Because detection is
identical, `maskformer` and `maskformer_incidence` claim exactly the same cells -- asserted in
`tests/test_incidence_labels.py` -- and the difference between their rows is the assignment
rule and nothing else. That is what makes the pair a measurement rather than a tuning choice.

### What it measured

The two rules disagree about the owner of **98.5% of claimed cells**, so this is a real
experiment rather than two names for one clustering. Exclusive head-to-head, 500 events:

| | mask head | incidence head | CLUE |
|---|---|---|---|
| efficiency @0.5 | 0.341 | **0.350** | 0.315 |
| purity @0.5 | 0.290 | **0.304** | 0.255 |
| split rate (energy) | 0.420 | **0.364** | 0.300 |
| fragmentation | 0.551 | **0.527** | 0.306 |
| fake rate | 0.398 | **0.372** | 0.559 |
| merge rate (energy) | **0.283** | 0.295 | 0.159 |

The gain concentrates exactly where the mask head was weakest, which is the fragmentation
story from the other direction: efficiency at E > 5 GeV goes 0.239 -> 0.277 against CLUE's
0.297, and at E > 20 GeV 0.147 -> 0.173 against CLUE's 0.224. Roughly half the gap to CLUE in
the energetic bins closes by changing nothing but which head is read. Below ~3 GeV the two are
indistinguishable. It does not overturn the headline: CLUE still wins above 5 GeV.

**The soft study went the other way, and that was not the prediction.** The expectation was
that incidence shares, being trained against real energy fractions, would divide a contested
cell better than mask probabilities. They divide it *worse*:

| | mask head | incidence head | CLUE | truth |
|---|---|---|---|---|
| effective claims/cell | 2.02 | 4.15 | 1.00 | 1.16 |
| soft efficiency @0.5 | 0.227 | 0.175 | 0.309 | |
| soft purity @0.5 | 0.351 | 0.278 | 0.292 | |

The incidence head spreads a cell over 4.15 queries effectively, against the mask head's 2.02
and truth's 1.16. Its softmax is flatter than the mask head's sigmoid, not sharper. So the
head is better at *ranking* which query owns a cell -- which is all the exclusive argmax needs
-- and worse at saying *by how much*. Both are consistent with a model that has learned the
ordering before the sharpness, which is what 20k optimiser steps would predict. Quote the
exclusive gain and the soft loss together; taking either alone misrepresents the head.

Read `eff_claims_per_cell`, not `claims_per_cell`, when comparing these. The raw count depends
on how many claims a method happens to emit: the incidence head is stored as a fixed top-k, so
its raw count is pinned near k (15.4 at k = 16) and reports the truncation rather than the
model. The effective count is a perplexity over the normalised weights and is k-independent.

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

  **This was tested against the incidence head and the answer was no.** The model does have a
  head trained to divide a cell -- see *Two heads* above -- so `scripts.score_soft` scores
  both: `maskformer` with mask probabilities as the sharing weights, `maskformer_incidence`
  with incidence shares. If the over-division were purely a matter of the mask head predicting
  the wrong quantity, the supervised head should have divided better. It divides *worse*: 4.15
  effective ways per cell against the mask head's 2.02 and truth's 1.16.
  
  So the over-division is not explained by which quantity is read. Both heads are too diffuse,
  in the same direction, and the head trained on energy fractions is the more diffuse of the
  two. That points at the model being undertrained rather than at the mask head's objective --
  consistent with a validation loss still falling at the final checkpoint after only ~20k
  optimiser steps. It does not rescue the mask head's calibration; it says the architecture
  argument cannot be settled from this checkpoint, and the run length has to be fixed first.

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
  `figures/pu0/weighting_comparison.pdf` shows both definitions side by side.
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

The implementation is in `attic/jets.py`, where it does not run: it was written against the
raw parquet loader, which the event store replaced. Reviving jets means rewriting it against
`src/io/event_store.py`, not repairing it in place.

## `attic/`

Four modules from the design before the event store, when each method opened the raw
ColliderML parquet and applied the shared cuts itself. None of them imports — they call
`src.config.dataset_paths` and `src.config.split_bounds`, which no longer exist — and nothing
live references them. They are kept because the decisions inside them are still open even
though the code is not; `attic/README.md` says which is which.
