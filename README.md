# Calorimeter clustering: a classical baseline and a learned model

Code for an MSc thesis comparing the classical **CLUE** density-based clustering algorithm
against a **MaskFormer** set-prediction model on high-granularity calorimeter data from the
[ColliderML](https://huggingface.co/datasets/CERN/ColliderML-Release-1) release. Both pileup
conditions are complete: pileup 0 (full detector) and pileup 200 (barrel, |eta| <= 0.88).

The point of the comparison is that it is *controlled*. Both methods read the same events, are
given the same cells, are asked to reconstruct the same particles, and are scored by the same
code that is never told which method produced the labels it is given.

The mechanism that makes that true is the main design decision here. The two methods do not
each read the dataset and apply matching cuts. Instead a single **event store** is dumped once,
from the model's own dataloader, and CLUE clusters the cells in that store. "Both algorithms saw
identical input" is therefore structural rather than a promise two config files make to each
other.

A useful consequence: everything downstream of the store depends on nothing but numpy, scipy,
pandas and matplotlib. No GPU, no `hepattn`, no access to the ~300 GB dataset. The binned result
tables are small enough to commit, so every thesis figure regenerates from the repository alone.

## Layout

```
config/experiment.yaml   every shared decision, and the expectations checked against the store
src/config.py            loads, merges and validates the experiment definition
src/io/event_store.py    reads the event store; validates it against the config
src/io/colliderml.py     reads the RAW dataset parquet - only for the dataset-composition
                         numbers, which have to see the particles before the target cuts
src/clue/                the CLUE baseline and its hyperparameter search
src/evaluation/          matching, metrics, differential binning, jets, the resolution ceiling
src/plotting/            figure styling and binning helpers
src/maskformer/          the learned model: configs, training launchers, and the dump that
                         writes an event store. Needs hepattn and a GPU; see its README
scripts/                 command line entry points
setup/                   fresh clone -> reproduced result: both environments and the download
tests/                   scorer identity, the reference ceiling, CLUE's periodic metric
results/<dataset>/       committed tables, one directory per pileup condition
figures/thesis/          the eight figures in the thesis
```

Two things about that layout are load-bearing.

**`src/maskformer/` is the only part that needs a GPU.** The rest of `src/` imports nothing but
numpy, scipy, pandas and matplotlib. The two halves are kept apart by nothing importing across
the line, not by which directory they sit in:

```bash
grep -rn "src\.maskformer" --include="*.py" src scripts tests
```

returns nothing. `src/io/event_store.py` is a hand-written *mirror* of the dump's
`eval/format.py` rather than an import of it, so if one side changes the format the other fails
loudly on a version check instead of silently misreading.

**Only the dataset lives outside the repository.** It is ~300 GB and sits on a shared datastore.
Everything these scripts build goes under `external/`, which is gitignored — so "what is
generated?" is answered by `ls external/`, and deleting it returns the clone to its committed
state.

## Redrawing the thesis figures

This needs no GPU, no event store and no cluster access — only the repository.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib scienceplots pyyaml
python -m scripts.make_thesis_figures --datasets pu0 pu200 --from-summary
```

Six packages. `CLUEstering` and `fastjet` are deliberately not needed: they are imported lazily,
only on the paths that re-cluster events from the store, which this workflow never touches.

`--from-summary` is what makes it work off-cluster and is **not optional here**. It draws each
column from the committed `results/<dataset>/figure_summary.csv` and reads nothing else — not the
store, not the per-row tables. Without it the script tries to rebuild the shower profiles from
the event store and fails with `no chunk_*.npz found`.

Add `--latex` (or `export CALO_FIGURE_LATEX=1`) to set every label with a real LaTeX
installation, so figure text matches the document. It is off by default because neither cluster
has TeX and matplotlib raises at draw time rather than falling back. On Debian/Ubuntu:
`sudo apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super`.

Figures are already sized for **0.8 of a 12 pt textwidth**, with body text at 9 pt, so include
them without rescaling. Rescaling in LaTeX is what makes figure text disagree with body text; if
a figure needs a different size, change `figsize_for` in `src/plotting/thesis.py` and redraw.

Each figure function lives in `scripts/make_thesis_figures.py` as `fig_<name>`; shared styling is
in `src/plotting/thesis.py`, where colours are assigned by *role*, so `_COLOURS["maskformer"]`
recolours that method everywhere at once.

`--from-summary` cannot change any number. Redraw locally for anything cosmetic; go back to a
cluster to change a binning, a working point, a CLUE parameter, which algorithms appear
(`METHODS`), or anything needing the event store.

## Reproducing the analysis

### Install

Two environments, and they are separate on purpose:

| | built by | contents | needs |
|---|---|---|---|
| `external/venv-hepattn` | `setup/install_training_env.sh` | torch, hepattn, flash-attn | GPU |
| `external/conda-envs/calo-clustering` | `setup/install_analysis_env.sh` | numpy, scipy, pandas, matplotlib, CLUEstering, optuna | nothing |

The analysis env is conda rather than a venv for the reason `environment.yml` documents:
CLUEstering needs a scikit-learn with no wheel for this python/numpy combination, so pip falls
back to a source build and fails, and the CLUE CPU backends compile against Boost headers.
`setup/paths.sh` is the single place every location is defined; the other scripts read it rather
than hardcoding paths.

```bash
./setup/install_analysis_env.sh      # figures and scoring need only this one
./setup/install_training_env.sh      # only if you are training or dumping a store
python setup/download_data.py        # ~297 GB, resumable
python setup/verify_data.py          # checks the shards are complete and correctly paired
```

`verify_data.py` is not a formality. `ColliderMLDataset` pairs a particles shard with the
calo_hits shard of the *same filename* and uses the intersection of the two listings, so a shard
that downloaded in one collection but not the other is dropped silently — no error, just a lower
event count than you think you have.

### Dump a store, then score it

Store production is the only GPU step; see [`src/maskformer/README.md`](src/maskformer/README.md).
Everything below is numpy-only and runs from the repository root:

```bash
python -m scripts.show_config          # what the active dataset resolved to; opens no store
python -m scripts.tune_clue            # Optuna TPE, one study per subsystem, on the tune window
python -m scripts.scan_mf_threshold    # the MaskFormer (mask, object) working point
python -m scripts.score --algo maskformer
python -m scripts.score --algo clue                # picks up this dataset's tuned parameters
python -m scripts.score --algo oracle_resolution   # the resolution ceiling
python -m scripts.make_thesis_figures              # the eight figures + figure_summary.csv
```

Given an event store, `scripts.score` then `scripts.make_thesis_figures` regenerates every
figure. The remaining scripts each produce one supporting measurement quoted in the thesis —
together with the seven commands above, that is every file in `scripts/`:

| script | what it produces | where it lands |
|---|---|---|
| `scan_link_radius.py` | CLUE's cross-subsystem link radius, chosen on split/merge rather than f1 | `link_radius_scan.parquet` |
| `scan_working_points.py` | the efficiency/purity frontier both methods sit on | `wp_scan.parquet` |
| `rescore_mask_threshold.py` | the model re-scored at several mask thresholds, with jets rebuilt | `mask_threshold_rescore.csv` |
| `measure_truth_geometry.py` | exclusive-partition cost, target counts, energy coverage, crowding | `truth_geometry.csv`, `truth_isolation.csv` |
| `make_dataset_figures.py` | the selection cutflow, read from the raw parquet rather than the store | `dataset_features_selection.csv` |
| `score_soft.py` | the multi-owner capability study — how many ways each method divides a cell | `capability_summary.csv` |
| `scan_soft_threshold.py` | that same sharing traced against the mask threshold, to show it is not a working-point artefact | `soft_threshold_scan.parquet` |
| `measure_split_origin.py` | whether CLUE's residual splitting is intra- or cross-subsystem, and the oracle ceiling on repairing it by linking | `split_origin.csv` |

### Switching pileup condition

`dataset.active` in `config/experiment.yaml` is the only switch. It scopes the stores,
`results/<dataset>/` and the Optuna study names together, so a pu200 run cannot land on a pu0
table. Run `scripts.show_config` after changing it — the active dataset's `overrides:` block is
deep-merged over everything else before any consumer sees the settings, and that merge is the one
thing in the config that cannot be checked by reading the file.

When tuning CLUE on a new condition, **read the range-edge warnings** `tune_subsystem` prints and
widen in the direction they name. A baseline tuned in the wrong box is under-tuned, which is the
one way this comparison can be unfair to CLUE.

### Two clusters, one figure

pu0 was trained and scored on **DIAS**, pu200 on **ce-ai-1**, and the thesis puts them side by
side. The per-row tables never make that trip: `particles_*.parquet` and `clusters_*.parquet` are
20–30 MB each at pu0 and several times that at pu200, `.gitignore` excludes them, and no per-row
table has ever been in this repository's history.

What travels is **`results/<ds>/figure_summary.csv`** — the binned series the figures draw, ~20 KB
per dataset, written by `scripts.make_thesis_figures` and committed for exactly this purpose. So
the round trip is just git. A machine that *has* the per-row tables always rebuilds its own
summary rather than trusting the committed one, so a rescore cannot be silently plotted over.

## What is in `results/`

Committed, per dataset:

| file | contents |
|---|---|
| `figure_summary.csv` | the binned series every thesis figure draws |
| `events_*.parquet` | pooled per-event counts per algorithm |
| `clue_parameters.json` | the tuned CLUE parameters, with the study that produced them |
| `truth_geometry.csv`, `truth_isolation.csv` | target counts, energy coverage, crowding |
| `mask_threshold_rescore.csv` | the model re-scored at several mask working points |
| `capability_summary.csv` | claims per cell against truth's, for the multi-owner study |
| `wp_scan.parquet`, `mf_threshold_scan.parquet`, `link_radius_scan.parquet` | the parameter scans |
| `dataset_features_selection*.csv` | what each target cut costs, before the store exists |

Not committed: the per-row `particles_*`, `clusters_*` and `soft_particles_*` tables, and the
`anatomy_particles` / `dataset_features` caches. Regenerate them with `python -m scripts.score
--algo <name>` on a machine holding the store.

## Tests

```bash
python -m pytest
```

Runs on a bare clone: everything skips rather than fails when its prerequisites are absent.
Point `SMOKE_STORE` at a small event store to run the rest.

| test | covers | needs |
|---|---|---|
| `test_scorer_identity.py` | the scorer: truth fed back as a prediction must score exactly 1, with no splits, merges or fakes | store |
| `test_ceilings_and_weighting.py` | the match floor, hit- vs energy-weighted split/merge, the resolution ceiling | store |
| `test_soft_capability.py` | fractional ownership must neither favour nor suppress an overlapping method | store |
| `test_thesis_binning.py` | the binning and error bars behind every plotted point | — |
| `test_jet_matching.py` | jet `delta_r` (including the phi wrap), the greedy match, the four-vector sum | — |
| `test_config_resolution.py` | the tune/eval window guard and the pileup overrides merge | — |
| `test_colliderml_reader.py` | the raw-parquet reader: shower collapse, phi-wrapped isolation | — |
| `test_clue_linking.py`, `test_clue_periodic.py` | cross-subsystem linking, and CLUE's periodic phi metric | CLUEstering |

`test_scorer_identity.py` is the one the rest rest on: if the scorer does not return exactly 1
for a perfect clustering, no other number in the thesis means anything.

Known gaps, all needing an event store to close: `src/evaluation/anatomy.py` (the shower
profiles in figures 2–3) and `src/clue/search.py` (the Optuna objective) have no direct tests.
Both are exercised end to end whenever the figures are regenerated on a machine with a store.
