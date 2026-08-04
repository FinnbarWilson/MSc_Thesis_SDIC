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
