# Handover: the three mask-head variants, and why they exist

Written 2026-08-09 on DIAS, for whoever picks this up on ce-ai-1. Everything below was measured on
the **pu0** dataset; you will be running on **pu200 barrel**, which is why the comparison rules at
the bottom matter.

Read `../HIGH_ENERGY_STATUS.md` for the full evidence. This file is the short version plus the
instructions.

---

## 1. Where the project stands

The thesis compares **CLUE** (a classical density-based calorimeter clusterer) against a
**MaskFormer**-style transformer, calorimeter cells only, no tracking information on either side.

**The headline result is already won.** On pu0, MaskFormer plus two CPU post-processing steps beats
a tuned CLUE across the board, on identical events at matched purity:

| | efficiency | purity | eff > 5 GeV | eff > 20 GeV |
|---|---|---|---|---|
| CLUE (tuned) | 0.315 | 0.251 | 0.297 | 0.224 |
| **MaskFormer + merge + chaining** | **0.414** | 0.248 | **0.371** | **0.262** |

Paired bootstrap over 500 events: overall **+0.100** [+0.097, +0.102], above 20 GeV **+0.038**
[+0.030, +0.046]. The post-processing lives in `src/postproc/` and needs no GPU.

**These variants are not needed for that result.** They are an attempt to fix the model itself,
because we know precisely how it fails and want to have tried before writing up.

## 2. How the model fails

The mask head assigns cells with an **independent sigmoid per (query, cell)** — a dot product
between a query vector and a cell vector. So a cell can only ask *"do I look like this query"*,
never *"am I attached to a cell that does"*. A calorimeter shower is defined by connectivity across
physically diverse cells, which is the second question.

Measured consequences:

- The matched cluster holds **~6 cells whether the particle deposits 13 or 38**, and *shrinks* as
  showers grow (6.2 → 5.5 cells while truth goes 13.5 → 37.6).
- It spans ~0.06 in angle against true shower radii of 0.074–0.228, and **1 cm of depth out of a
  42 cm shower**.
- **41%** of a >20 GeV particle's energy is claimed by NO query, at any threshold down to 0.02 —
  confident exclusion, not hedging.
- The model emits **753 clusters per event for 538 true particles**.

**Six training-side interventions moved none of it**: masked attention off, incidence head restored,
exclusive mask targets, positional-encoding bandwidth at ×2 and ×5, and an auxiliary cell-pair
affinity head. Each one epoch, one variable, matched control.

**Eleven post-processing methods were tried.** Two work (chaining, fragment merging — both in the
pipeline above). Nine do not, including density-flow repartitioning, splitting at local maxima, and
a learned affinity over encoder embeddings that beats geometry on pair AUC (0.817 vs 0.742) and
changes nothing end to end.

## 3. What the three variants change

Each is a single switch on `ExtendedObjectHitMaskTask`
(`../hepattn_colliderml/mask_variants.py`), a subclass of hepattn's `ObjectHitMaskTask`. **With all
switches off it is exactly the parent**, which is what makes each overlay a one-variable change.
No patch to hepattn is involved — the class is named by `class_path` and reached through the same
mirror sync as everything else.

| overlay | switch | attacks |
|---|---|---|
| `overlay_v1_coverage.yaml` | `coverage_weight: 2.0`, energy-weighted | DICE is size-normalised, so covering a 6-cell fragment scores like covering a 38-cell shower. Adds a per-target penalty on the energy fraction its query failed to claim — the reported metric, used as a loss. |
| `overlay_v2_recall.yaml` | `bce_pos_weight: 20.0` | Each query sees ~22,000 cells with ~13 positives (1:1700). Plain BCE is nearly minimised by claiming nothing. hepattn's `mask_bce_loss` has no `pos_weight`; this reimplements it with one. |
| `overlay_v3_propagation.yaml` | `propagation_lambda: 0.5` | **The structural one.** Adds `logit_i ← logit_i + λ·mean(logit_j over neighbours j)`, turning "do I look like this query" into "do I, or does my neighbourhood". Smallest change that gives the head a relational term. |

v3 targets the biggest measured weakness. It is listed last only because it is the only one that
alters the forward pass.

## 4. How to run them

One at a time — this box has a single A100 and pu200 peaked at 68 GB of 81, so two concurrent runs
will OOM.

```bash
cd <repo>/external/hepattn/src/hepattn/experiments/colliderml   # or wherever env.sh points
# smoke first, especially for v3: its kNN + sparse-matmul path has never executed anywhere
NUM_TRAIN=20 OVERLAYS="overlay_pu200_barrel.yaml overlay_v3_propagation.yaml" \
  ../../../../../src/maskformer/ce_ai_1/train_pu200.sh

# then the real runs, in this order
OVERLAYS="overlay_pu200_barrel.yaml overlay_v1_coverage.yaml"    nohup ./train_pu200.sh > ~/v1.log 2>&1 &
OVERLAYS="overlay_pu200_barrel.yaml overlay_v2_recall.yaml"      nohup ./train_pu200.sh > ~/v2.log 2>&1 &
OVERLAYS="overlay_pu200_barrel.yaml overlay_v3_propagation.yaml" nohup ./train_pu200.sh > ~/v3.log 2>&1 &
```

`OVERLAYS` (plural, space-separated, later files win) is new. `OVERLAY` (singular) still behaves
exactly as before, so a bare `./train_pu200.sh` is unchanged.

`env.sh` re-syncs the mirrored subtree into the checkout on every run, so `mask_variants.py` and the
three overlays arrive automatically. Do not edit the checkout copies — the repository is the source
of record and the sync will overwrite them.

## 5. How to judge the result — READ THIS BEFORE CONCLUDING ANYTHING

**You need a control on the same dataset.** These are pu200-barrel numbers and are NOT comparable
to the pu0 table in section 1 — different target definition, different cuts. Run (or reuse) plain
`OVERLAY=overlay_pu200_barrel.yaml` and compare variant-versus-that.

**Do not judge on total loss.** v1 and v2 add terms to the objective, so their loss is a different
sum and is not comparable to the control's. This trap already caught the incidence arm's `kl_div`.

**Do not judge on efficiency alone.** Every one of these buys recall with precision. An arm that
lifts efficiency while dropping purity has moved along the trade curve, not improved anything. Score
both and read them together.

**The quantity that would actually mean something** is the cluster-size-versus-particle-size curve:
does the matched cluster start growing with the shower instead of sitting at ~6 cells? That is the
pathology, and it is more sensitive than efficiency at short training lengths.
`../dias/compare_probes.py` computes exactly this as its `slope` column and can be pointed at any
dumped store.

**Two epochs is what `overlay_pu200_barrel.yaml` sets**, sized from a measured 0.153 it/s. That is a
signal test, not a converged result.

## 6. Honest state of the evidence

Every confident architectural prediction made during this investigation was wrong — masked
attention, positional encoding, cluster splitting, the affinity head. The two things that worked
(chaining, merging) were both post-processing, and both were found by measuring where the energy
went rather than by reasoning about the architecture.

Treat these three as ranked hypotheses with their evidence attached, not as a plan that will work.
The value of running them is that the report can then say they were tried, with controls, rather
than speculating.

## 7. Pointers

- `../HIGH_ENERGY_STATUS.md` — the full evidence: six training arms, eleven post-processing methods,
  the energy budgets, the ceilings
- `../hepattn_colliderml/mask_variants.py` — the three switches, with the measurement behind each
- `../../postproc/` — chaining and merging, the two things that work
- `README.md`, `PU200_STATUS.md` — this machine, and what pileup-200 forced to change
