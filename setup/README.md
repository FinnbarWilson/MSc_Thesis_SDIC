# `setup/` — from a fresh clone to a reproduced result

Everything needed to go from `git clone` to the figures. Four scripts and one rule.

**The rule: code and build artefacts live in this repository. Only the dataset lives outside it.**
The ColliderML shards are ~300 GB and sit on a shared datastore
(`/mnt/ai-datastore/finnbar/ColliderML_data`); nothing else does. Everything these scripts build —
the hepattn checkout, both environments — goes under `external/`, which is gitignored. So "what is
generated?" is answered by `ls external/`, and deleting that directory returns the clone to its
committed state.

`setup/paths.sh` is the single place those locations are defined; the scripts and
`src/maskformer/ce_ai_1/env.sh` all read it rather than hardcoding paths.

## The two environments, and why there are two

| | built by | contents | needs |
|---|---|---|---|
| `external/venv-hepattn` | `install_training_env.sh` | torch, hepattn, flash-attn | GPU |
| `external/conda-envs/calo-clustering` | `install_analysis_env.sh` | numpy, scipy, pandas, matplotlib, CLUEstering, optuna | nothing |

This is not two ways of installing the same thing. `src/clue/`, `src/evaluation/`,
`src/plotting/` and `src/io/` import nothing but numpy, scipy, pandas and matplotlib, which is
what lets every figure in the thesis be regenerated with no GPU and no access to the dataset. The
root README calls that the repository's main design decision; two environments is what it looks
like at install time.

The analysis env is conda rather than a venv for the reason `environment.yml` documents:
CLUEstering needs a scikit-learn with no wheel for this python/numpy combination, so pip falls
back to a source build and fails, and the CLUE CPU backends compile against Boost headers.

## Order

```bash
./setup/install_analysis_env.sh      # figures only need this one
./setup/install_training_env.sh      # only if you are training or dumping a store
python setup/download_data.py        # ~297 GB of ttbar_pu200, resumable
python setup/verify_data.py          # checks the shards are complete and correctly paired
```

To regenerate the committed pu0 figures, the analysis env alone is enough — no dataset, no GPU:

```bash
external/conda-envs/calo-clustering/bin/python -m scripts.make_figures
```

For the pileup-200 run, continue at
[`src/maskformer/ce_ai_1/README.md`](../src/maskformer/ce_ai_1/README.md).

## Two clusters, and what has to change on DIAS

pu0 is trained and scored on **DIAS**, pu200 on **ce-ai-1**, and the thesis figures put them side by
side — so whichever machine draws them needs both columns. Three things differ on DIAS, and all
three are consequences of it being a shared RHEL7 cluster rather than a box you own.

**Everything runs inside a container.** DIAS is RHEL7 with glibc 2.17; hepattn needs 2.28+, and so
do the pinned conda-forge builds in `environment.yml` (numpy 2.4.6, pandas 3.0.3). Both
environments are therefore built *and* run inside `~/ubuntu22.sif`, including the analysis env:

```bash
apptainer build ~/ubuntu22.sif docker://ubuntu:22.04
apptainer exec --bind $HOME ~/ubuntu22.sif bash setup/install_analysis_env.sh
```

The image is the bare `ubuntu:22.04` base and ships no compiler and no `curl`. `dias/env.sh` points
`CC`/`CXX` at the pixi env's conda-forge GCC — without it Triton cannot initialise its driver and
training dies at the first kernel — and `install_analysis_env.sh` falls back to wget or python for
the miniforge download.

**The store lives somewhere else.** `config/experiment.yaml` holds ce-ai-1's `/mnt/ai-datastore`
paths. `CALO_STORE_ROOT` relocates them without editing the config, so one config stays valid on
both machines. Only the directory moves — the store *name* still comes from the config, because it
encodes the window and format version `EventStore` checks:

```bash
export CALO_STORE_ROOT=$HOME/eventstores     # put this in ~/.bashrc on DIAS
```

**Slurm, not `nohup`.** The launchers live in [`src/maskformer/dias/`](../src/maskformer/dias/):

```bash
sbatch src/maskformer/dias/train.sh                                  # pu0 is the default
CKPT=<ckpt> sbatch src/maskformer/dias/dump_store.sh pu0 tune
CKPT=<ckpt> sbatch src/maskformer/dias/dump_store.sh pu0 eval
sbatch src/maskformer/dias/analysis.sh                               # tune, scan, score, figures
```

`analysis.sh` runs on the CPU `COMPUTE` partition and is the DIAS counterpart to
`scripts/run_pu200_pipeline.sh`. Both `--mem` values look absurd for the work being done and are
not negotiable: Slurm here applies `VSizeFactor`, so `--mem` is a hard *virtual* memory cap of
1.1 × the request, and `expandable_segments` reserves a large virtual range at start-up. An
under-request surfaces as `CUDA driver error: out of memory` on an idle 80 GB card.

### Getting both columns into one figure

The per-row tables never move. `results/<ds>/particles_*.parquet` and `clusters_*.parquet` are
~200 MB at pu0 and several times that at pu200, `.gitignore` excludes them, and no per-row table
has ever been in this repository's history. What travels is
**`results/<ds>/figure_summary.csv`** — the binned series the five thesis figures draw, ~20 KB,
written by `scripts.make_thesis_figures` and meant to be committed.

So the round trip is just git: score pu0 on DIAS, commit its summary, push; pull on ce-ai-1, score
pu200 there, and `make_thesis_figures` draws both columns from the two CSVs. A machine holding the
per-row tables always rebuilds its own summary rather than trusting the committed one, so a rescore
cannot be silently plotted over.

## `verify_data.py` is not a formality

`ColliderMLDataset` pairs a particles shard with the calo_hits shard of the **same filename** and
uses the *intersection* of the two directory listings. A shard that downloaded in one collection
but not the other is therefore dropped silently — no error, just a lower event count than you
think you have. That is what this checks, along with every file's parquet footer parsing and
reporting 100 rows, because a truncated download usually still has a plausible size.

## A note on the hepattn pin

`src/maskformer/README.md` names commit `30ccb9f` as the reference version.
**That commit no longer exists upstream** — a fresh clone reports `fatal: Not a valid object
name`, so it was rebased or squashed away. `install_training_env.sh` therefore pins `cb4fb10`
("Add ColliderML Experiment", #231), the surviving commit that introduces the experiment this
thesis uses. `hepattn-changes.patch` applies cleanly to it — and, as it happens, to `main`,
`1df05cc` and `93b2842` as well, so this is the most specific surviving choice rather than the
only one that works.

Override it if you need to: `HEPATTN_COMMIT=<sha> ./setup/install_training_env.sh`.

This is worth a decision rather than a footnote. hepattn is cloned rather than vendored because
`src/maskformer/README.md` makes that an explicit authorship choice, and cloning still satisfies
"clone the repo, download the dataset, reproduce". But upstream has now deleted the pinned commit
once, and can again — at which point the pu0 checkpoint's provenance is no longer reconstructible
from this repository alone. If that provenance matters more than the authorship boundary, vendor
the tree (hepattn is GPL-3.0, so it can be redistributed with attribution and the licence intact,
which also makes this repository GPL-3.0).
