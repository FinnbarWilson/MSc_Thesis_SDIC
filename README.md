# Calorimeter clustering: a classical baseline against a learned model

Code for an MSc thesis comparing the classical CLUE density-based clustering algorithm with a MaskFormer set-prediction model on high-granularity calorimeter data from the [ColliderML](https://huggingface.co/datasets/CERN/ColliderML-Release-1) release, at pileup 0 and pileup 200.

The eight figures in `figures/thesis/` are the ones in the report. Every path below reproduces them; they differ only in how much of the chain you rerun yourself.

## What to run

| path | what you rerun | needs | takes |
|---|---|---|---|
| [1](#path-1-redraw-the-figures) | the plotting only, from the committed result tables | nothing | minutes |
| [2](#path-2-rerun-the-methods) | both methods, on event stores you make from the published checkpoints | a GPU and 260 GB | hours |
| [3](#path-3-full-run) | everything, including training and CLUE tuning | a GPU and 260 GB | days |

Every figure has a pileup-0 column and a pileup-200 column, so Paths 2 and 3 do both conditions throughout.

---

## Path 1: redraw the figures

```bash
conda env create -f environment.yml
conda activate calo-clustering

python -m scripts.make_thesis_figures --datasets pu0 pu200 --from-summary
```

Takes about a minute and rewrites `figures/thesis/*.pdf|png` over the committed copies. `--from-summary` draws each column from the committed `results/<dataset>/figure_summary.csv` and reads nothing else, which is what lets it run with no dataset and no GPU. Add `--latex` if you have a LaTeX installation and want the labels typeset with it.

Dependencies are all in `environment.yml`. Without conda, see [Installing without conda](#installing-without-conda).

Two more things worth running:

```bash
python -m pytest -rs        # 70 passed, 19 skipped; the 19 need an event store
python -m scripts.show_config
```

---

## Path 2: rerun the methods

Downloads the events both conditions are reported over, runs the published MaskFormer checkpoints on them to build event stores, then reruns MaskFormer and CLUE against those stores and redraws every figure from the result.

CLUE's parameters and both working points are already committed, so nothing is tuned here. That is Path 3.

Do the whole path for pu0, then repeat it for pu200. The two are independent until the last step, which draws both columns.

### 1. Install both environments

```bash
./setup/install_analysis_env.sh      # numpy-only, into external/
./setup/install_training_env.sh      # torch, hepattn and the thesis patch; needs a GPU
```

`install_analysis_env.sh` installs its own miniforge under `external/`, which is what a compute cluster usually needs. On a normal machine `conda env create -f environment.yml` gives the same environment.

### 2. Download both datasets

```bash
python setup/download_data.py --pileup pu0   --shards 21     # ~22 GB
python setup/verify_data.py   --pileup pu0   --shards 21

python setup/download_data.py --pileup pu200 --shards 80     # ~238 GB
python setup/verify_data.py   --pileup pu200 --shards 80
```

`download_data.py` fetches shards `0` to `N-1`. Pileup 0 holds 1,000 events per shard and its windows end at event 20,750, so 21 shards cover it. Pileup 200 holds only 100 events per shard and its windows end at event 8,000, so it needs 80. Set `COLLIDERML_DATA` first if the dataset must live somewhere other than `external/`.

Run `verify_data.py` both times. `ColliderMLDataset` pairs a particles shard with the calo_hits shard of the same filename and uses the intersection of the two listings, so a shard that downloaded in one collection but not the other is dropped without an error, leaving a lower event count than expected.

### 3. Fetch both checkpoints

```bash
python -m scripts.fetch_checkpoints
```

Downloads both, each about 112 MB, from [the repository's releases](https://github.com/FinnbarWilson/MSc_Thesis_SDIC/releases). They are release assets rather than git objects because GitHub rejects any single file over 100 MB. Each lands at the path `config/experiment.yaml` already names, so no config edit is needed, and each is verified against a SHA-256 recorded in `scripts/fetch_checkpoints.py`. `--list` shows what exists and what is already local.

### 4. Dump the event store

This is the GPU step. It runs the model over the 500 evaluation events and writes the cells, the truth partition and the model's predictions into `external/eventstores/`, which is where `config/experiment.yaml` already looks. Everything after this is CPU-only, because the predictions are in the store and the scoring reads them back as arrays.

```bash
mkdir -p external/slurm_logs        # sbatch writes its logs here and will not create it

# absolute, because the launchers cd into the hepattn checkout before running
CKPT=$(python -c "from pathlib import Path; from src.config import settings_for; print(Path(settings_for('pu0')['maskformer']['checkpoint']).resolve())")

CKPT=$CKPT sbatch src/maskformer/dias/dump_store.sh pu0 eval
```

Without a scheduler, the `ce_ai_1` launcher takes the same arguments and runs in the foreground:

```bash
cd src/maskformer/ce_ai_1 && CKPT=$CKPT ./dump_store.sh pu0 eval
```

Neither launcher will match a new cluster exactly; see [Running on a cluster of your own](#running-on-a-cluster-of-your-own) and [`src/maskformer/README.md`](src/maskformer/README.md). Lower `CHUNK` if the dump is killed for memory.

### 5. Score both methods

Set `dataset.active` in `config/experiment.yaml` to the condition you are working on, then:

```bash
python -m scripts.show_config                      # confirm it found the store you just made
python -m scripts.score --algo maskformer
python -m scripts.score --algo clue
python -m scripts.score --algo oracle_resolution
```

`score --algo clue` picks up `results/<dataset>/clue_parameters.json`, which is committed, and the MaskFormer working point comes from `config/experiment.yaml`. The scorer is given a label per cell and is not told which method produced it.

### 6. Rebuild this condition's figure data

```bash
python -m scripts.make_thesis_figures --datasets pu0        # or pu200, matching dataset.active
```

Name only the active condition. The shower profiles are recomputed from the event store, and the script only opens the store for the active dataset, so naming both here would overwrite the other condition's summary with one missing those rows.

### 7. Repeat for the other condition, then draw both columns

Go back to step 4 with `pu200` in place of `pu0` everywhere, including in the `CKPT` command and `dataset.active`. When both are done:

```bash
python -m scripts.make_thesis_figures --datasets pu0 pu200 --from-summary
```

Both columns of all eight figures now come from stores you produced. `git diff results/*/figure_summary.csv` shows whether any number moved.

### Also available

The store-backed tests, and the supporting measurements each of which produces one table quoted in the report:

```bash
SMOKE_STORE=external/eventstores/ttbar_pu0_20250_20750_v2 python -m pytest -rs

python -m scripts.rescore_mask_threshold  # the model at several mask thresholds
python -m scripts.score_soft              # fractional cell ownership
python -m scripts.measure_truth_geometry  # target counts and energy coverage
python -m scripts.bench_clue --backend "cpu serial"
python -m scripts.make_bench_table        # collects the timing runs into one table
```

---

## Path 3: full run

Path 2 with the checkpoints trained and CLUE tuned rather than taken from the repository. Same datasets and same downloads; what changes is days of GPU time and an extra store per condition.

### 1. Install and download

Path 2 steps 1 and 2, unchanged. The training windows sit inside the same shards: pileup 0 trains on events 0 to 20,000 and pileup 200 on 0 to 6,000.

### 2. Train both models

See [`src/maskformer/README.md`](src/maskformer/README.md). One run per condition, each a day or more.

### 3. Dump both stores per condition

```bash
CKPT=$CKPT sbatch src/maskformer/dias/dump_store.sh pu0 eval
CKPT=$CKPT sbatch src/maskformer/dias/dump_store.sh pu0 tune
```

`CKPT` is the checkpoint you trained. The tuning store is a separate 50-event window, disjoint from the evaluation one, and is what the next step needs.

### 4. Tune CLUE and pick the working point

With `dataset.active` set to the condition:

```bash
python -m scripts.tune_clue            # 80 Optuna trials per subsystem; hours
python -m scripts.scan_mf_threshold    # the MaskFormer (mask, object) working point
python -m scripts.scan_link_radius     # CLUE's cross-subsystem link radius
```

`tune_clue` overwrites `results/<dataset>/clue_parameters.json`. It prints a warning if a tuned parameter lands on the edge of its search range; widen that range in `config/experiment.yaml` and rerun rather than reporting the edge. `scan_mf_threshold` prints the grid, from which you set `maskformer.mask_threshold` and `.object_threshold` in the config.

### 5. Score and draw

Path 2 steps 5 to 7, unchanged.

### Running on a cluster of your own

No path needs editing. `setup/paths.sh` derives every location from the repository root, and the launchers, `config/experiment.yaml` and `scripts.fetch_checkpoints` all agree on `external/`. Use `COLLIDERML_DATA` and `CALO_STORE_ROOT` to move the dataset and the stores off it.

What will not carry over is the scheduler configuration. `src/maskformer/dias/` holds Slurm batch scripts for a cluster that needs an Apptainer container and has a faulty GPU to route around; `src/maskformer/ce_ai_1/` is the same sequence for a single machine with no scheduler. Expect to change the `#SBATCH` resource lines, the partition name and the container, and read `src/maskformer/dias/env.sh` for the module and compiler setup. Submit from the repository root, so `SLURM_SUBMIT_DIR` and the relative log paths resolve.

Comet logging in `src/maskformer/hepattn_colliderml/configs/*.yaml` points at a workspace you will not have access to. Change `workspace` to your own or set `online: false`.

---

## Reference

### How the pieces fit together

The work splits at one artefact, the event store: a directory of `.npz` files holding the calorimeter cells of each event, the truth partition over those cells, and the MaskFormer's raw predictions.

```
  dataset                                             event store              results/, figures/
       |                                                   |                          |
       |-- train the model ---------------------------.    |                          |
       |   src/maskformer/, GPU, hepattn              |    |                          |
       |                                              v    |                          |
       '-- dump ------------------------------> eval/dump.py --.                      |
           the model's own dataloader                          |                      |
                                                               v                      |
                                          CLUE clusters the same cells ---------------'
                                          scripts/, src/clue/, src/evaluation/, numpy only
```

Producing the store is the only step needing a GPU, `hepattn` and the dataset. Everything after it needs numpy, scipy, pandas and matplotlib. Because CLUE reads the cells the model was given rather than re-deriving them from the dataset, both methods are scored on identical input, and one scorer handles both without being told which produced a labelling.

`external/` holds everything the setup scripts build: the conda install, the training venv, the hepattn checkout, the dataset and the event stores. It is gitignored, so deleting it returns the clone to its committed state.

### Repository layout

```
config/experiment.yaml   every shared setting, and the expectations checked against the store
src/config.py            loads, merges and validates the experiment definition
src/io/event_store.py    reads the event store; validates it against the config
src/clue/                the CLUE baseline and its Optuna search
src/evaluation/          matching, metrics, differential binning, jets, the resolution ceiling
src/plotting/            figure styling and binning helpers
src/maskformer/          the learned model: configs, launchers, and the dump that writes an
                         event store. Needs hepattn and a GPU; see its README
scripts/                 command line entry points
setup/                   environments, the dataset download, and its verification
tests/                   the scorer, the references, the binning, the config, CLUE's metric
results/<dataset>/       committed tables, one directory per pileup condition
figures/thesis/          the eight figures in the report
```

### Scripts

Every file in `scripts/`, run as `python -m scripts.<name>` from the repository root. Output lands in `results/<active dataset>/` unless stated otherwise.

| script | what it does | produces |
|---|---|---|
| `show_config.py` | prints what the active dataset resolved to; opens no store | |
| `make_thesis_figures.py` | the eight report figures | `figures/thesis/*`, `figure_summary.csv` |
| `score.py` | scores one algorithm against the truth partition | `particles_*`, `clusters_*`, `events_*` |
| `tune_clue.py` | Optuna search for CLUE's parameters, per subsystem | `clue_parameters.json` |
| `scan_mf_threshold.py` | the MaskFormer (mask, object) working point | `mf_threshold_scan.parquet` |
| `scan_link_radius.py` | CLUE's cross-subsystem link radius, chosen on split/merge rather than f1 | `link_radius_scan.parquet` |
| `scan_working_points.py` | the efficiency/purity frontier both methods sit on | `wp_scan.parquet` |
| `rescore_mask_threshold.py` | the model re-scored at several mask thresholds, with jets rebuilt | `mask_threshold_rescore.csv` |
| `score_soft.py` | the multi-owner study: both methods with fractional cell ownership | `capability_summary.csv` |
| `measure_truth_geometry.py` | exclusive-partition cost, target counts, energy coverage, crowding | `truth_geometry.csv`, `truth_isolation.csv` |
| `bench_clue.py` | per-event CLUE latency, one backend per run | `bench_clue_<backend>.json` |
| `make_bench_table.py` | collects every `bench_*.json` into the timing table | `timing_summary.csv` |
| `fetch_checkpoints.py` | downloads the trained checkpoints from the GitHub release | files under `external/` |

MaskFormer's half of the timing measurement is `src/maskformer/hepattn_colliderml/eval/bench_maskformer.py`, which needs a GPU and draws its timing boundary in the same place as `bench_clue.py`.

### What is in `results/`

Committed, per dataset:

| file | contents |
|---|---|
| `figure_summary.csv` | the binned series every report figure draws |
| `events_*.parquet` | pooled per-event counts per algorithm |
| `jets.parquet` | anti-k_t jets per event, for the jet figure |
| `clue_parameters.json` | the tuned CLUE parameters, with the study that produced them |
| `truth_geometry.csv`, `truth_isolation.csv` | target counts, energy coverage, crowding |
| `mask_threshold_rescore.csv` | the model re-scored at several mask working points |
| `capability_summary.csv` | claims per cell against truth's, for the multi-owner study |
| `wp_scan.parquet`, `mf_threshold_scan.parquet`, `link_radius_scan.parquet` | the parameter scans |
| `bench_*.json`, `timing_summary.csv` | per-event latencies and the timing table |

Not committed: the per-row `particles_*`, `clusters_*` and `soft_particles_*` tables, which are 20 to 30 MB each at pu0 and several times that at pu200, and the `anatomy_particles` cache. Path 2 regenerates them.

`figure_summary.csv` is what moves between machines, at about 20 KB per dataset, which is how the two conditions, scored on different clusters, appear in one figure.

### Tests

| test module | covers | needs a store? |
|---|---|---|
| `test_config_resolution.py` | the tune/eval window guard and the pileup overrides merge | no |
| `test_thesis_binning.py` | the binning and error bars behind every plotted point | no |
| `test_jet_matching.py` | jet `delta_r` including the phi wrap, the greedy match, the four-vector sum | no |
| `test_clue_linking.py` | cross-subsystem linking, on synthetic events | no |
| `test_clue_periodic.py` | CLUE's periodic phi metric, and that the wrap flags are needed | no |
| `test_scorer_identity.py` | truth fed back as a prediction must score exactly 1, with no splits, merges or fakes | 4 of 9 |
| `test_ceilings_and_weighting.py` | the match floor, hit- against energy-weighted split/merge, the resolution ceiling | 7 of 13 |
| `test_soft_capability.py` | fractional ownership must neither favour nor suppress an overlapping method | 8 of 10 |

### Installing without conda

```bash
pip install numpy scipy pandas pyarrow matplotlib scienceplots pyyaml pytest
```

Runs Path 1, but gives 55 passed and 21 skipped instead of 70 and 19. `test_clue_linking.py` and `test_clue_periodic.py` hold 15 tests between them and are skipped whole, because CLUEstering has to compile against Boost and only the conda environment provides it.

### Figures

Each figure is a `fig_<name>` function in `scripts/make_thesis_figures.py`; shared styling and sizing are in `src/plotting/thesis.py`. `--from-summary` cannot change any plotted value, so redraw locally for anything cosmetic and use Path 2 to change a binning, a working point or a CLUE parameter.
