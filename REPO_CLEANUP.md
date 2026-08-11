# Repository audit and cleanup brief

For whoever restructures this repository. Written 2026-08-10 from a full survey of the tracked
tree (44 MB, ~150 tracked files) plus the untracked work in progress.

**The audience that matters is an MSc assessor**, who will spend perhaps thirty minutes in here.
They need to answer three questions without asking anyone:

1. *What produced the numbers in the thesis?*
2. *Could I reproduce them?*
3. *Is anything here that contradicts the thesis, or that I cannot account for?*

Question 3 is the one that costs marks. A directory of superseded outputs is worse than no
directory, because an assessor who finds `events_clue_v1.parquet` alongside `events_clue.parquet`
has to work out which produced the tables — and cannot.

---

## 0. Do this first, before deleting anything

There is **uncommitted work in the tree**, including today's figures, two new modules and several
config changes. Commit it before any restructuring, or the cleanup will destroy it.

```
git add -A && git commit -m "..."
```

Untracked: `src/plotting/thesis.py`, `src/evaluation/anatomy.py`, `src/evaluation/jets.py`,
`scripts/make_thesis_figures.py`, `scripts/analyse_anatomy.py`, `scripts/run_pu200_pipeline.sh`,
`src/maskformer/ce_ai_1/{train_pu0.sh,run_variants.sh}`,
`configs/{overlay_pu0_dice.yaml,overlay_variants_short.yaml}`, `figures/pu200/`,
`figures/thesis/`, `results/pu200/`, and the three planning documents.

Modified: `config/experiment.yaml`, `scripts/tune_clue.py`, `setup/download_data.py`,
`src/plotting/{figures.py,style.py}`, `src/maskformer/hepattn_colliderml/eval/{dump.py,geometry.py}`,
`src/maskformer/hepattn_colliderml/mask_variants.py`, and four overlay configs.

**Also note: a pu0 training run is in progress.** It writes into `external/` (gitignored) and
reads configs from `src/maskformer/hepattn_colliderml/`, which `ce_ai_1/env.sh` copies into the
hepattn checkout on every launch. **Do not move or rename anything under
`src/maskformer/hepattn_colliderml/` until that run finishes**, or a resume will pick up a broken
path.

---

## 1. What the assessor actually needs

### Tier 1 — the thesis stands or falls on these

| path | why |
|---|---|
| `README.md` | the entry point; must describe the layout that exists after cleanup |
| `config/experiment.yaml` | single source of truth for every shared decision |
| `src/config.py` | reads it, enforces the store contract |
| `src/io/event_store.py` | the controlled-comparison mechanism |
| `src/clue/` | the CLUE baseline (`pipeline.py`, `tuning.py`) |
| `src/evaluation/` | `metrics.py`, `matching.py`, `differential.py`, `oracle.py`, `soft.py`, `anatomy.py`, `jets.py` |
| `src/postproc/chain.py`, `merge.py` | the two post-processing steps that are reported |
| `src/plotting/thesis.py`, `style.py` | the thesis figures |
| `scripts/` | `score.py`, `tune_clue.py`, `scan_working_points.py`, `make_thesis_figures.py`, `show_config.py` |
| `src/maskformer/hepattn_colliderml/` | the model: `data.py`, `model.py`, `main.py`, `eval/`, the configs actually used |
| `src/maskformer/hepattn-changes.patch` | the three library modifications |
| `setup/` | how to rebuild the environments and fetch the data |
| `tests/` | six tests; evidence the scorer is checked, not asserted |
| `figures/thesis/` | the figures that appear in the document |
| `results/pu200/`, `results/pu0/` | the tables behind them |

### Tier 2 — supporting evidence an examiner may want

- `src/maskformer/HIGH_ENERGY_STATUS.md` — the six training interventions and eleven
  post-processing methods. This is the provenance for the negative results in §4.7 and should
  survive, though it belongs in a `docs/` directory rather than buried under `src/`.
- `src/maskformer/ce_ai_1/PU200_STATUS.md` — the pu200 sizing measurements.
- The sweep and variant configs (`sweep_*.yaml`, `overlay_v*.yaml`, `overlay_probe_*.yaml`) —
  each is a documented arm of a controlled experiment reported in the thesis. **Keep them**, but
  they need a README saying which thesis section each corresponds to, or they read as clutter.

### Tier 3 — delete

| path | reason |
|---|---|
| `attic/` (6 files) | its own README says "nothing here is imported, run, or tested" and that none of it currently imports. Its stated purpose — keeping decisions one `ls` away rather than one `git log` away — is not worth the confusion it causes an assessor. `git log` is the right place. |
| `src/maskformer/checkpoint/` (40 MB) | a tracked checkpoint from `epoch=003-val_loss=18.06191`, an objective (dice 5 + focal 20 + incidence head) that **no longer exists in the thesis**. It is 90% of the repository's tracked bytes and actively misleading. |
| `results/pu0/*_v1.*`, `*_v2.*`, `*_smoke.*` | superseded and smoke-test outputs sitting beside the real ones with no way to tell them apart. This is the single worst thing in the repo for question 3 above. |
| `results/pu0/*.log` (10 files) | run logs are not results. If provenance matters, keep one and say what it is. |
| `figures/pu0/*` produced by the old pipeline | superseded by `figures/thesis/`. Keep **only** `diagnosis_high_energy.png` if the thesis cites it; delete the rest once the pu0 column exists. |
| `configs/overlay_pu200.yaml` | the non-barrel first attempt, marked "superseded" in its own header. Delete, or move to a clearly-labelled `superseded/`. |

### Tier 4 — decide deliberately

- **`src/maskformer/dias/` (11 files).** Launchers for a cluster you no longer use. They produced
  the earlier pu0 results, so they are provenance — but if the final pu0 result comes from
  ce-ai-1, these describe a machine that contributed nothing to the submitted numbers. **Suggest
  deleting** once the new pu0 run lands, keeping `compare_probes.py` if the slope analysis is
  cited.
- **`src/maskformer/checkpoint/`** — see above. If you want *a* checkpoint in the repo for
  reproducibility, it should be the one the thesis uses, and it should probably be a release
  asset rather than a tracked blob.
- **`scripts/scan_soft_threshold.py`, `score_soft.py`, `src/evaluation/soft.py`** — the
  multi-owner capability study. Only meaningful with an incidence head, which the current model
  does not have. Keep only if the thesis discusses it; otherwise it is dead weight that invites
  the question "why is this here?".
- **`scripts/analyse_anatomy.py`** — superseded by `make_thesis_figures.py`, which produces the
  same material in thesis form. Delete unless you want the exploratory seven-figure version.

---

## 2. The structural problem, and the fix

The current layout has three genuine confusions:

**(a) `src/maskformer/hepattn_colliderml/` is a mirror, not a module.** It is a verbatim copy of a
subtree of the `hepattn` checkout, kept in sync by `verify_sync.sh` and copied into the checkout
by `env.sh` at every launch. Nothing in the repository imports it. An assessor reading `src/` will
assume it is library code and be wrong. **Fix:** move it to a top-level `model/` (or
`hepattn_experiment/`) with a README stating in the first line that it is a mirror of the hepattn
experiment directory, that `env.sh` copies it into the checkout, and that the checkout is where
the code actually executes.

**(b) Machine-specific launchers are buried inside `src/`.** `ce_ai_1/` and `dias/` are shell
scripts for particular clusters. **Fix:** a top-level `run/` directory, or `model/launchers/`.

**(c) Documentation is scattered across four directories.** `README.md` (31 KB — too long for an
entry point), `src/maskformer/README.md`, `HIGH_ENERGY_STATUS.md`, `ce_ai_1/README.md`,
`PU200_STATUS.md`, `VARIANTS_HANDOVER.md`, `attic/README.md`, `setup/README.md`. **Fix:** one
short top-level `README.md` that orients and points; everything else in `docs/`.

### Suggested target layout

```
README.md              short: what this is, how to install, how to reproduce, where things are
config/experiment.yaml
src/                   the analysis library — everything here is importable
  config.py  io/  clue/  evaluation/  postproc/  plotting/
scripts/               the entry points, one job each
model/                 the MaskFormer half
  hepattn_colliderml/    mirror of the hepattn experiment dir (README explains this)
  hepattn-changes.patch
  launchers/             ce_ai_1/ (and dias/ only if kept)
setup/                 environment build + data download
tests/
docs/                  HIGH_ENERGY_STATUS, PU200_STATUS, the three chapter plans
results/pu0/  results/pu200/     scored tables only, no logs, no versioned duplicates
figures/thesis/        the figures in the document
external/              gitignored build artefacts (unchanged)
```

---

## 3. Can it be reproduced? Honestly, partly

**What works today.** The analysis half is genuinely reproducible: given an event store,
`scripts/score.py` → `scripts/make_thesis_figures.py` regenerates every table and figure, and
`config/experiment.yaml` plus the store contract make the inputs verifiable rather than assumed.
`setup/` rebuilds both environments and fetches the data. That is better than most projects.

**What does not.** Four gaps an assessor could hit:

1. **No trained checkpoint is distributed.** Every figure needs an event store, and every store
   needs a checkpoint the repository does not contain. Reproduction therefore means a ~20–45 hour
   training run on an A100. **Fix:** state this plainly in the README, and give the checkpoint a
   home — a release asset, a DOI, or a documented path on the group storage.
2. **Event stores live outside the repository** (`/mnt/ai-datastore/finnbar/eventstore_pu200/`) and
   are referenced by absolute path in `config/experiment.yaml`. Anyone else gets a path error.
   **Fix:** document the expectation, and make the failure message say so.
3. **`huggingface_hub` was missing** from both environments despite `download_data.py` requiring
   it. Fixed by hand today; add it to `setup/install_analysis_env.sh` so the next clone works.
4. **The pu0 numbers currently in `results/pu0/` come from a superseded objective**, and at least
   two vintages of output are mixed in that directory. Until the new run lands and those files are
   replaced, the repository contains results that contradict the thesis.

---

## 4. A pre-submission checklist

- [ ] Commit everything currently untracked and modified
- [ ] Wait for the pu0 run to finish before moving `hepattn_colliderml/`
- [ ] Replace `results/pu0/` wholesale with the new run's output; delete every `_v1`, `_v2`,
      `_smoke` and `.log` file
- [ ] Delete `attic/`, `src/maskformer/checkpoint/`, superseded `figures/pu0/`
- [ ] Restructure per §2; update every path in `README.md`, `env.sh`, `verify_sync.sh`,
      `paths.sh` and `config/experiment.yaml`
- [ ] Re-run `python -m scripts.make_thesis_figures --events 500` from a clean clone and confirm
      the figures in the thesis regenerate byte-identically
- [ ] Run `pytest` and record the result in the README
- [ ] Run `src/maskformer/verify_sync.sh` and confirm the mirror matches the checkout
- [ ] Confirm every figure and table in the thesis has a named producing script
- [ ] Add `huggingface_hub` to the analysis environment spec
- [ ] Write the "how to reproduce" section against a **clean clone**, and follow it literally once

---

## 5. One thing not to do

Do not delete the sweep configs, the variant overlays, the probe overlays, or
`HIGH_ENERGY_STATUS.md` in the name of tidiness. They are the evidence for every negative result
the thesis reports, and an examiner asking "you say you tried six interventions — where are they?"
should be able to see them. They need a README, not a bin.
