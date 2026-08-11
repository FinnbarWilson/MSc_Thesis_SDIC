# Methodology chapter — writing plan

Everything you need to write Section 3. Every number here was read out of the running code,
the configs, or the scored tables on 2026-08-10; where a number is not yet measured it is
marked **[TBD]** with the command that produces it.

---

## STATE OF EVIDENCE — what exists, what does not, and what may be assumed

**Read this before writing any number.** Last updated 2026-08-10, 23:00.

### Complete and safe to quote

| | status |
|---|---|
| pu200 barrel, 5-epoch model, 500-event evaluation | **complete** — all tables, all six figures |
| pu200 CLUE tuning (2 sub-detectors, 80 trials each) | **complete**, no range-edge warnings |
| pu200 jet analysis (499 events with ≥1 reference jet) | **complete** |
| Mask-objective sweep (4 arms, 4,000 steps) | **complete** |
| Mask-head variant arms v1/v2/v3 + control | **complete**, with bootstrap CIs |
| Working-point threshold scan | **complete** |
| Training-length trend at 4k / 12k / 30k steps | **complete** |

### Not yet available

**The entire pu0 column.** The pu0 run using the current objective (`dice 20 + focal 1`) is
training now: at the time of writing it is in epoch 1 of 7, ~28,000 of 140,000 optimiser steps,
running at 0.87 it/s with roughly 35 hours remaining. Nothing pu0-shaped from that run exists yet —
no store, no scored tables, no figures, no jets.

**What must happen after it finishes**, in order:
1. dump the pu0 event store from the final checkpoint (tune window and eval window),
2. re-tune CLUE on the new pu0 tune store — the store changes because the checkpoint changes,
3. re-derive the MaskFormer working point on that store,
4. score every algorithm,
5. re-run `python -m scripts.make_thesis_figures --events 500 --rebuild-anatomy --rebuild-jets`.

Every figure then gains its pu0 column with **no code change**.

### Old pu0 numbers exist, but do not belong in this thesis without a warning

`results/pu0/reference_table.csv` holds a complete scored result from a **previous** pu0 run using
a **different mask objective** (`dice 1 + bce 1`, no dice-dominance). Headline, all particles:

| | efficiency | purity | eff > 5 GeV | eff > 20 GeV |
|---|---|---|---|---|
| CLUE | 0.315 | 0.251 | 0.297 | 0.224 |
| MaskFormer | 0.371 | 0.274 | 0.239 | 0.150 |
| MaskFormer + chaining | 0.447 | 0.216 | 0.310 | 0.197 |
| oracle_geometric | 0.608 | 0.194 | 0.452 | 0.295 |
| oracle_resolution | 0.965 | 0.487 | 0.962 | — |

**Two problems with using these.** First, the objective differs from pu200's, so the thesis would
report its two conditions under two different training objectives — on top of already differing in
η coverage. Second, the provenance inside `results/pu0/` is mixed: `capability_summary.csv`
contains a `maskformer_incidence` row, which can only come from a store *older* than the ep006
checkpoint the config names, so at least two vintages are present in that directory. Note also
that the existing introduction quotes 0.414 for MaskFormer + chaining where this table says 0.447.

**Use them only as a placeholder while writing**, clearly marked, and replace every one when the
new run lands.

### What may reasonably be assumed while drafting

The old pu0 result and the new pu200 result agree on the qualitative pattern, so the following are
**likely but unconfirmed** and must be written as gaps rather than as findings:

- MaskFormer is expected to be **ahead of CLUE on overall efficiency at pu0** (it was, 0.371 vs
  0.315) while **behind above 20 GeV** (0.150 vs 0.224). The energy-dependent crossover is the one
  pattern seen in both conditions and both objectives.
- The cluster-scale behaviour is expected to persist: pu0's own diagnostic figure showed the
  matched cluster holding ~6 cells whether the particle deposited 13 or 38.
- Chaining is expected to raise efficiency and lower purity.

**Do not write these as results.** Draft the prose with the shape, leave the numbers as visible
placeholders (`\todo{pu0}` or `XX`), and fill them in once measured. If the new objective changes
the pu0 picture — which is the whole reason for running it — assumed numbers left in the text
would become false statements.

---

## 0. What changed since your draft — read this first

Your existing draft describes a configuration that no longer exists. Writing from it would
produce a methodology that does not match the results.

| Your draft says | Current reality |
|---|---|
| Lion optimiser | **AdamW** |
| `accumulate_grad_batches: 4` | **1** |
| gradient clipping 0.1 | **1.0** |
| Incidence head, `kl_div` weight 100 | **Removed entirely** |
| Loss `dice 5 + focal 20` | **`dice 20 + focal 1`** |
| pu200: cell cut 1e-3, pt ≥ 2.0, 500 queries, ~117k cells, 277 targets | **Barrel-only \|η\|<0.88, cell cut 2e-4, pt ≥ 0.5, 2100 queries, ~56.5k cells, 1499 targets** |
| pu200 uses 4 sub-detectors | **2** (ecb, hcb) — the η cut removes both endcaps |
| pu0: 4 epochs | **7 epochs** |
| No jet metrics | **Anti-k_t jets implemented** |
| Two mask/incidence readings compared | **Only one reading** (no incidence head to compare) |

There is also a **provenance trap**: the pu0 numbers currently in `results/pu0/` were produced on
DIAS by the *old* objective (`dice 1 + bce 1`, 7 epochs, checkpoint `epoch=006-val_loss=4.92620`).
The pu0 run training now uses `dice 20 + focal 1`. **Do not mix them.** Every pu0 number in the
thesis must come from the new run, re-scored. Note this explicitly in your notebook so it cannot
happen by accident.

### On your first subsection

Your instinct is right, and the problem is specific rather than general: **§3.1 currently mixes
three different jobs**. It describes the dataset, then justifies a MaskFormer memory constraint,
then defines the target selection. The middle part — mask-logit counts, 80 GB accelerators, batch
sizes — is not a property of the dataset. It is a property of *your* model, and a reader meeting
it in the dataset section has no idea yet what a query is.

Fix by splitting: the dataset section states what the sample is and what the selections are; the
*reason* the pu200 selections differ moves into §3.5 (the model), where the reader already knows
what a mask logit is. That removes the waffle without losing the content — and the content is
good, it is just in the wrong place.

---

## 1. Recommended structure

Written as an examiner would want to read it: **what is being compared → what the data is → how
it is measured → how each method is configured → what is held fixed**. Measurement comes *before*
the methods because both methods are tuned against it, and an examiner will want to know the
metric was fixed before anyone optimised anything.

```
3.1  Experimental design            what is controlled, and how that control is enforced
3.2  Dataset and event selection    ColliderML, the two conditions, the target definition
3.3  The event store                the mechanism guaranteeing both methods see the same cells
3.4  Evaluation
     3.4.1  Cluster-level metrics
     3.4.2  Jet-level metrics
     3.4.3  Uncertainties
3.5  The CLUE baseline              pipeline, coordinates, tuning protocol
3.6  The MaskFormer model           inputs, architecture, objective, training, inference
3.7  Computing environment          hardware, software, reproducibility
```

Put §3.3 early. It is the single strongest methodological claim you have, and everything after
it inherits the guarantee.

---

## 2. §3.1 Experimental design

**One paragraph, no more.** State the design and move on.

- Controlled comparison: identical events, identical cells, identical target definition,
  identical scoring code, at two pileup conditions.
- The two conditions are **two configurations of one experiment**, not two experiments.
- **Critical honesty point, state it here:** pu0 and pu200 results are *not* directly comparable
  to each other, because the pu200 configuration restricts to the barrel (|η| < 0.88) while pu0
  covers |η| < 4.0. Each condition is internally controlled; cross-condition statements are not
  supported. An examiner will look for this, and stating it up front is worth more than a
  footnote later.
- Forward-reference the table of conditions.

---

## 3. §3.2 Dataset and event selection

### The dataset
- **ColliderML**, simulated, built on the experiment-agnostic **OpenDataDetector** geometry.
  Cite `elitezColliderMLFirstRelease2025`, `murnaneColliderML2025`.
- Sample: **top-quark pair production (ttbar)** — fills the calorimeter with both isolated
  particles and dense jet cores, so it exercises both regimes in Section 2.2. (if you include this you will need to find a refernce, if you cannot dont include it)
- The property that makes the task supervised: the simulation records, per cell, **which truth
  particles deposited energy in it and how much each contributed** — the fractional incidence
  matrix `I_ia = E_ia / E_a` of Section 2.2.
- Four calorimeter sub-detectors: ECAL barrel/endcap, HCAL barrel/endcap (`ecb`, `ece`, `hcb`,
  `hce`).

### Release size and sharding — a fact worth stating because it is counter-intuitive (but dont spend too much time on it as its not really that important. maybe a couple sentences about it)
| | events/shard | shards | total events | size (calo_hits + particles) |
|---|---|---|---|---|
| ttbar pu0 | **1,000** | 1,000 | **1,000,000** | 1,063 GB |
| ttbar pu200 | **100** | 1,000 | **100,000** | 297 GB |

A pu0 event holds ~22k cells against pu200's ~532k under identical cuts, which is why pu0 packs
10× more events into a shard a third the size. **Downloaded and used:** 100 shards of pu0
(100,000 events, 106 GB) and 100 shards of pu200 (10,000 events, 297 GB).

### Target particle definition (shared, and it is one definition doing two jobs)
A target particle must satisfy:
- `pt ≥ 0.5 GeV`
- `|η| ≤ 4.0` (pu0) / `|η| ≤ 0.88` (pu200 barrel)
- **≥ 3 calorimeter cells** counted *after* zero-suppression

This same set is simultaneously the MaskFormer's training target and the denominator of CLUE's
efficiency, so the two methods are asked to reconstruct exactly the same objects. Say this
explicitly — it is what makes the efficiency comparable at all.

**Why ≥ 3 cells:** particles depositing one or two cells are not reconstructable but would
otherwise sit in the efficiency denominator. The cut drops ~15% of targets while keeping 97.7% of
the calorimeter energy.

### Cells belonging to no target are kept, deliberately
- pu0: **63% of cells, 46% of energy** have no target owner.
- pu200 barrel: **61.5% of cells, 56.8% of calibrated energy** *(measured on 30 events of the
  eval store)*.

These are real deposits from particles below the pt cut plus, at pu200, pileup. They are **not**
removed, so both methods must reject them unaided. They can only ever count against purity, never
against efficiency. This is a design decision with a reason — state the reason.

### The two conditions

| | pileup 0 | pileup 200 |
|---|---|---|
| Cell energy threshold | 2×10⁻⁴ GeV | 2×10⁻⁴ GeV |
| Target pt | ≥ 0.5 GeV | ≥ 0.5 GeV |
| Pseudorapidity (cells **and** particles) | \|η\| ≤ 4.0 | **\|η\| ≤ 0.88** |
| Minimum cells per target | ≥ 3 | ≥ 3 |
| Cells / event | ~22,100 | **56,517** |
| Target particles / event | 538 | **1,499** |
| Object queries | 1,000 | 2,100 |
| Training window | [0, 20000) | [0, 6000) |
| Validation window | [20000, 20250) | [6000, 6250) |
| CLUE tuning window | [20000, 20050) | [7000, 7050) |
| Evaluation window | [20250, 20750) | [7500, 8000) |
| Sub-detectors present | 4 | **2** (ecb, hcb) |

Note the cell threshold and pt cut are now **identical** across conditions — only the η range
differs.

### Why barrel-only (`|η| ≤ 0.88`) at pu200 — put the justification here in one sentence, the detail in §3.6
Two independent reasons, and the physics one is the better one to lead with:
1. **Containment.** The HCAL barrel reaches r = 3441 mm at |z| ≤ 3450 mm, so a particle steeper
   than η = 0.883 exits through the barrel end and deposits the remainder of its shower in the
   endcap. Cutting at 0.88 keeps every target's shower fully contained in one region.
2. **Tractability.** A faithful pu200 event under pu0 cuts holds 532,507 cells and 8,182 targets;
   the mask-logit tensor is queries × cells, so that is 4.4×10⁹ entries against pu0's 2.2×10⁷ —
   roughly 200× the footprint, two orders of magnitude past an 80 GB accelerator.

Applying the η cut to **both cells and particles** matters: cutting cells alone would leave
targets whose cells had been removed.

### Truncation — a limitation to state, not hide
`event_max_num_particles` caps the target list. **6 of 500** pu200 eval events and **18 of 500**
pu0 events hit that cap and were truncated. Small, but an examiner will ask what happens when an
event has more particles than queries.

### Detector calibration and layer geometry (pileup-independent)
- Per-sub-detector sampling calibration `κ_s`: **ecb 37.5, ece 38.7, hcb 45.0, hce 46.9**.
  These do not cancel in a ratio, because a cluster's effective calibration depends on how its
  energy divides between ECAL and HCAL — so they must be applied per cell before any sum.
- Layer counts: **ECAL 48 layers at 5.05 mm; HCAL 36 layers at 51.0 mm**, barrel and endcap alike.
- **Barrel layer recovery:** the barrel is built from **16 flat staves**, so a "layer" spans a
  range of radii and is not recoverable from radius alone. It is recovered by projecting each cell
  onto its stave normal. Endcap layers are planes of constant |z|.
- Layer calibration is derived once per store by pooling several events (a single event leaves
  most HCAL barrel layers unlit — a per-event derivation finds 26 of 36) and is then **frozen** and
  written into the store, so both methods and every figure use identical layer indices.

---

## 4. §3.3 The event store — the control mechanism

This is your strongest methodological argument. Give it its own short subsection.

**What the store physically is.** A directory of compressed `.npz` chunks (10 events each) plus a
`meta.json`. Per event it holds: cell positions `x, y, z`, raw cell energy, a sub-detector code
(0–3 for ecb/ece/hcb/hce), a calibrated layer index, the exclusive truth label per cell, the full
multi-owner truth as a particle-major CSR (`indptr`, `indices`, `incidence`), the truth particle
table (id, pdg, four-momentum, η, φ, pt, energy, cell count), and the MaskFormer's own output —
per accepted query a validity probability and a sparse row of mask logits over cells. It is
~756 kB/event at pu200 barrel (378 MB for the 500-event store) and ~736 kB/event for the 50-event
tune store.

**Why the model's dataloader writes it.** The alternative — two pipelines each reading the raw
ColliderML parquet and each applying the cuts — makes "both methods saw the same cells" a claim
resting on two configuration files agreeing. Writing once from the network's own dataloader makes
it a property of the data path.

- Cells are written **once**, from the MaskFormer's own dataloader, into an event store. CLUE then
  clusters that store. So the cells CLUE sees are **byte-for-byte** the cells the network was given.
  This is structural, not two configs agreeing by inspection.
- The store carries the selections it was produced under as **metadata**, and
  `src/io/event_store.py` checks them against the experiment config every time it is opened,
  refusing to proceed on a mismatch. *(This is not hypothetical: it caught a real discrepancy —
  the store had `particle_max_abs_eta: 0.88` while the config still declared 4.0.)*
- The store also asserts the evaluation window is **disjoint from the checkpoint's training
  window**, so scoring on trained-on events fails loudly rather than silently inflating results.
- Format version 2. Mask logits stored quantised to `uint8` over the range [−8, +8] in 256 levels,
  which is what allows any working point at or above the storage floor to be re-derived later
  without a GPU.
- **One honest gap to disclose:** the dump records `calohit_min_energy` under `hit_selection` but
  *not* `calohit_max_abs_eta`. The barrel cell cut therefore travels in the config that produced
  the store rather than in its metadata, and cannot be contract-checked. The cells themselves are
  in the store either way, so this is a provenance gap, not a correctness one.

---

## 5. §3.4 Evaluation

Lead with: **both methods are scored by one implementation, which is given a cluster label per
cell plus the event truth, and is never told which method produced it.** The measurement is
defined before either method because both are tuned against it.

### 3.4.1 Cluster-level metrics

**Notation.** Particles `i`, cells `a`, clusters `k`. Cell energy `E_a`, of which particle `i`
contributed `E_ia`, so `I_ia = E_ia/E_a`. Calibrated: `Ê_a = κ_s(a) E_a`.

**Exclusive truth partition.** `t(a) = argmax_i E_ia`.
- *Why:* CLUE returns a partition and cannot represent a shared cell. Scoring it against a
  fractional truth would penalise the *form* of its output rather than its clustering.
- *Cost, measured:* the exclusive partition keeps ~83% of (particle, cell) associations, carrying
  ~94% of each particle's deposited energy.
- **pu200 barrel measurement:** only **0.78%** of cells have ≥2 *target* contributors, so the
  exclusive approximation is far cheaper here than the ~12.6% quoted at pu0. Worth stating — it
  weakens the "MaskFormer can represent overlap" argument on this sample, and saying so is better
  than letting an examiner find it.

**Overlap matrix and margins:**
```
O_ik = Σ_{a∈C_k, t(a)=i} Ê_ia      T_i = Σ_{a: t(a)=i} Ê_ia      P_k = Σ_{a∈C_k} Ê_a
```
Hit-counted forms computed alongside; **energy-weighted are primary**, because cell energies span
orders of magnitude and a calorimeter measures energy, not occupancy.

**Matching.** Global one-to-one assignment maximising shared calibrated energy, solved with the
Hungarian algorithm, retained only if `O_iπ(i) ≥ 0.05 · min(T_i, P_π(i))`.
- *Why global, not greedy:* greedy is asymmetric — one particle can win several clusters while
  another wins none.
- *Why a floor:* without it, a pair sharing one low-energy cell counts as a match, and an
  energetic cluster in a busy event is nearly certain to graze some particle.
- *Why `min(...)`:* keeps the test symmetric between the two objects.
- *Why not the training-time matcher:* that compares query `i` with target `i` and is meaningful
  only after the loss has permuted them.

**Efficiency and purity.**
```
ε_i = O_iπ(i) / T_i        p_k = O_π⁻¹(k),k / P_k
```
with `ε_i = 0` for unmatched particles and `p_k = 0` for unmatched clusters, so a missed particle
is an inefficiency and a fake counts against purity rather than leaving the denominator.
Reported as **the fraction of objects reaching a working point**: 0.5 primary, 0.75 and 1.0
alongside.

Two conventions, each removing a bias:
- Efficiency is **per truth particle**, not per cluster — a cluster-side denominator holds only
  particles that won a cluster, so a particle merged into a neighbour would vanish instead of
  counting as a miss.
- Quantities are **pooled across events**, not averaged per event, so every particle carries equal
  weight regardless of how busy its event was.

**Structural failures**, from the same matrix with `f = 0.10`:
```
n_frag_i = |{k : O_ik ≥ f·T_i}|      n_own_k = |{i : O_ik ≥ f·T_i}|
```
Split when `n_frag > 1`, merged when `n_own > 1`. Both thresholds are fractions of the *truth
particle's* total, so a cluster swallowing one particle whole and a tenth of a neighbour has merged
them however its own energy divides.
- **State the blind spot:** a particle divided among more than `1/f = 10` clusters gives none of
  them a tenth of itself and is recorded as *unsplit* — understating splitting for exactly the
  methods that fragment most. The continuous alternative `φ_i = 1 − max_k O_ik / T_i` has no such
  blind spot and is plotted where the two disagree.

**Differential reporting.** Against particle energy, pseudorapidity, and local density (ΔR to the
nearest other target, and count within ΔR < 0.2), the density variables taken from generator
momenta so they belong to the event, not to either method's output.

### 3.4.2 Jet-level metrics

**Why jets:** per-cluster metrics say how well cells are grouped; they do not say whether the error
survives into an observable. A jet sums many candidates, so per-cluster errors can cancel,
accumulate, or push a jet across a threshold.

**The reference is the truth partition, not generator jets.** For every target particle the cells
it actually owns are summed into a four-vector and *those* are clustered — so reference jets are
what a **perfect clusterer** would produce from these cells under these cuts.
- *Why not generator jets:* that would fold in zero-suppression, the target selection, and the
  calorimeter's sampling resolution — all properties of the detector, not of the algorithm under
  test. They would degrade both methods equally and mask the thing being measured.

**Four-vectors from cells.** A cell measures energy and position, not momentum, so each is treated
as massless from the interaction point through the cell centre:
```
p = Ê_a · (x,y,z)/|(x,y,z)|,    E = Ê_a
```
Cluster four-vector = sum over its cells. Reference and methods use the **same rule over the same
cells**, so any jet difference comes from *which cells were grouped*.

**Settings and their justification.**
| Choice | Value | Why |
|---|---|---|
| Algorithm | anti-k_t | IRC-safe, regular boundaries (`cacciariAntik_tJetClustering2008`) |
| Radius R | 0.4 | LHC default for ttbar final states |
| Jet pt threshold | 25 GeV | standard ttbar analysis threshold |
| Match cone | ΔR < 0.3 | smaller than R, so a reco jet cannot sit equidistant between two adjacent reference jets |
| Implementation | FastJet 3.5.1.3 via scikit-hep bindings | `cacciariFastJetUserManual2012` |

Matching is greedy by reference jet pt, and a reco jet is consumed once taken. Unmatched reference
jets are **lost**; unmatched reco jets are **fakes**. Both are reported — a method can buy response
by splitting one jet into two.

**Reported:** jet efficiency, median `pt_reco/pt_ref` (energy scale), and `σ_pt/pt` from
IQR/1.349, all against **reference jet pt**.

**Caveat to state in the caption:** both methods over-measure because the reference excludes
pileup by construction. The meaningful quantity is the *difference between methods*, not either
one's distance from unity.

### 3.4.3 Uncertainties
- Fractions are binomial proportions → **Clopper–Pearson** at 68.27% (α = 0.3173).
- Pooled ratios of sums are not proportions → **bootstrap**.
- **The resampling unit is the event, not the particle**, because particles in one event share
  cells and occupancy and are not independent trials. Say this — it is the kind of detail that
  distinguishes a careful analysis.

---

## 6. §3.5 The CLUE baseline

**Pipeline** (from Section 2.3), spelled out as implemented:

1. Select the cells of one sub-detector (by the store's sub-detector code).
2. **Pass 1, per layer.** For each calibrated layer index in turn, run CLUE in 2D over that
   layer's cells in `(η, φ)`, weighted by raw cell energy. Cells CLUE marks as outliers get no
   label. Labels are made unique across layers by the caller, since CLUE restarts numbering in
   every call.
3. **Reduce.** Each layer cluster becomes one point: energy-weighted `η̄`, circular energy-weighted
   `φ̄`, and a third coordinate `ℓ/D` (layer index over a tuned depth scale). Its weight is the
   summed energy of its cells.
4. **Pass 2, over centroids.** Run CLUE in 3D over those points, producing tracksters.
5. **Propagate.** Every cell inherits the trackster label of its layer cluster; cells that were
   outliers in pass 1, or whose layer cluster was an outlier in pass 2, remain unclustered.
6. Repeat per sub-detector, then offset labels so they are unique across the event.

Clusters smaller than `min_cluster_hits` (1, i.e. no cut) are dropped.

**Coordinates:** projective `(η, φ)`.
- Centroids: `η̄` energy-weighted; `φ̄` via `atan2(Σ E sinφ, Σ E cosφ)` — circular, so a cluster
  straddling ±π is not placed on the opposite side of the detector.
- Third coordinate: **layer index ℓ divided by a tuned depth scale D**. Using the index rather than
  a physical depth keeps the depth axis meaning the same thing everywhere — ECAL layers are 5.05 mm
  apart and HCAL layers 51 mm, so one linking radius in metres would be 10× more permissive in one
  than the other.

**Periodicity needs two changes, not one** — worth stating, it is a real implementation trap:
the periodic metric alters the distance function, but CLUE first bins points into a **tile grid**
to decide which pairs are compared at all, and that grid wraps only when told to. With the metric
alone, points either side of the discontinuity land in non-adjacent tiles and are never compared,
so a shower straddling the boundary still returns as two clusters.

**Tuning protocol.** Seven parameters per sub-detector (`d_c`, `ρ_c`, `d_o` per pass, plus `D`),
searched with a **tree-structured Parzen estimator, 80 trials** (Optuna).
- Objective: `F₁ = 2εp/(ε+p)` at the 0.5 working point, **computed by the same function that
  produces the reported tables** — the search optimises the reported quantity, not a proxy.
- Parameters sampled logarithmically (ranges span 2–3 decades); each linking radius sampled as a
  **multiple of its own density radius**, below which it has no effect.
- Tuned on the **tune window**, which is disjoint from the evaluation window. `src/config.py`
  asserts this disjointness.

**The search ranges are an output, not an input.** An optimum landing in the outer 5% of its range
means the true optimum lies outside the box and the baseline would be reported under-tuned; the
range is widened in the direction named and the search repeated until nothing presses a bound.
- At **pu0** the first pass returned three such parameters and the ranges were widened.
- At **pu200** the loop was re-run rather than inheriting pu0's ranges, because `ρ_c` is a local
  energy density and pileup raises the occupancy it is measured over. **Result: no parameter
  pressed a bound** — every optimum sat between 8% and 72% of its log range (closest:
  `hcb.ρ_c^2D` at 8.1%, `ecb.d_c^2D` at 10.8%). So the pu0 ranges transfer. Report this; it is a
  measurement that answers a question the design raised.

**Tuned parameters — pu200 barrel** (two sub-detectors; `ece`/`hce` contain no cells and are
skipped rather than tuned against an empty selection):

| Sub-detector | F₁ | d_c^2D | ρ_c^2D | d_o^2D | d_c^3D | ρ_c^3D | d_o^3D | D |
|---|---|---|---|---|---|---|---|---|
| ECAL barrel | 0.345 | 0.001646 | 4.012×10⁻⁵ | 0.004754 | 0.01294 | 2.404×10⁻³ | 0.02478 | 505.0 |
| HCAL barrel | 0.527 | 0.007368 | 3.488×10⁻⁵ | 0.01831 | 0.04406 | 6.541×10⁻⁴ | 0.08574 | 187.8 |

**Tuned parameters — pu0** *(from the existing tune; re-verify against the final run)*:

| Sub-detector | F₁ | d_c^2D | ρ_c^2D | d_o^2D | d_c^3D | ρ_c^3D | d_o^3D | D |
|---|---|---|---|---|---|---|---|---|
| ECAL barrel | 0.313 | 0.007248 | 1.228×10⁻⁴ | 0.01984 | 0.01459 | 2.083×10⁻³ | 0.03242 | 386.1 |
| ECAL endcap | 0.247 | 0.006097 | 1.136×10⁻³ | 0.008112 | 0.02086 | 3.260×10⁻³ | 0.03180 | 186.5 |
| HCAL barrel | 0.460 | 0.002855 | 9.037×10⁻⁵ | 0.003362 | 0.05033 | 1.644×10⁻⁴ | 0.13700 | 119.2 |
| HCAL endcap | 0.259 | 0.05403 | 3.636×10⁻⁴ | 0.07905 | 0.06938 | 4.932×10⁻³ | 0.18300 | 26.36 |

*(F₁ here is per-subsystem, computed against a truth partition masked to that subsystem — it is
not comparable to the whole-event F₁ in the results chapter. Say so.)*

**Known limitation of the configuration, not of CLUE:** because sub-detectors are clustered
independently, a hadron showering from ECAL into HCAL yields **at least two clusters by
construction**. The split rate records this as such.

---

## 7. §3.6 The MaskFormer model

**Provenance.** Built with `hepattn` (`stroudTransformersChargedParticle2024`), pinned at commit
`cb4fb10`. Written for this thesis: the ColliderML dataset interface, the model configuration, the
evaluation stage that produces the event store, and three modifications to the library kept as a
214-line patch rather than a fork, so the upstream version in use stays identifiable.

**The three modifications, and why each exists:**
1. `models/loss.py` — adds `mask_dice_weighted_loss`, a DICE variant accepting a per-cell weight.
   Plain DICE ignores sample weights, so without this an energy-weighted objective would have
   weighted only the focal term. It reduces exactly to plain DICE under uniform weights.
2. `models/task.py` — adds an `eval_threshold` argument to `ObjectClassificationTask`. Upstream
   the object head's decision is an implicit argmax (equivalent to 0.5); this exposes it so the
   operating point can be chosen by measurement rather than inherited.
3. `callbacks/prediction_writer.py` — adds an `output_name` argument, so several evaluation passes
   can write to distinct files instead of colliding on one default name.

None of the three changes the architecture; they expose settings and add one loss variant.

### Inputs
Each cell enters as `(x, y, z, r, η, φ)` plus **log energy**.
- *Why log:* the input network applies no normalisation, and raw cell energies (mean 0.0035 GeV)
  against coordinates of order unity would be numerically negligible — the single most important
  feature would be drowned out.
- A **Fourier positional encoding** of `(r, η, φ)` is added to each embedded cell. *Why Fourier
  rather than the default:* the default frequencies alias at the scale that matters (a shower spans
  ~0.007 in η), giving cells of the same particle decorrelated encodings; measured 2.3× more
  locality signal.

### Architecture
| | |
|---|---|
| Width | 256 |
| Encoder | 4 layers, windowed attention, window 1024, sorted by φ, **wrapped** |
| Decoder | 4 layers, mask attention on, decoder window 512 |
| Queries | 1,000 (pu0) / 2,100 (pu200) |
| Parameters | **9.6 M** (pu0) / **9.9 M** (pu200) |
| Precision | bf16-mixed |

Query count caps the clusters the model can return, and must equal `event_max_num_particles`.

### Heads and objective
**Two heads only** — the incidence head has been removed.
1. **Classification head** — probability the query is a real particle.
2. **Mask head** — independent sigmoid per (query, cell).

```
L = L_class(object_bce, w=1) + 20·L_dice + 1·L_focal
```
Applied after **every decoder layer**. Matching cost: **dice only** (weight 1.0).
- `loss.py` has no `mask_bce` entry in its cost functions, so BCE could not be a matching cost.
- **Justify the weights from the sweep, not by assertion.** Three arms at 4,000 steps on the pu200
  barrel config, scored on 10 test events:

| objective | max mask prob | eff@0.5 | eff@0.75 | pur@0.5 | cells/flow (truth 15) |
|---|---|---|---|---|---|
| dice 20 + focal 1 | 1.0000 | 0.664 | 0.419 | 0.118 | 31.7 |
| dice 5 + focal 20 | 1.0000 | 0.665 | 0.411 | 0.118 | 32.8 |
| dice only | 0.0000 | — collapsed — | | | |
| dice 1 + bce 1 | 0.0995 | — collapsed — | | | |

**The finding:** focal's *presence* is what matters, not its weight. Both arms with it are
statistically indistinguishable despite an 80× difference in the dice:focal ratio; both without it
collapse, by two different mechanisms — BCE converges to predicting the target prior everywhere,
dice alone saturates the sigmoid at zero on a flat gradient region. Dice-dominance won a tie on
mechanism (dice governs predicted set size; over-prediction was the remaining failure).

Worth adding: with AdamW the **absolute loss scale is near-irrelevant** (Adam normalises by the
second moment), so only the *ratio* carries information — and that has now been sampled across 80×.

### Training
| | pu0 | pu200 |
|---|---|---|
| Optimiser | AdamW | AdamW |
| LR schedule | OneCycle, 1e-5 → 1e-4 → 1e-6, `pct_start` 0.01 | same |
| Weight decay | 1e-5 | 1e-5 |
| Gradient clipping | 1.0 | 1.0 |
| `accumulate_grad_batches` | 1 | 1 |
| Batch size | 1 event | 1 event |
| Epochs × events | 7 × 20,000 = **140,000 steps** | 5 × 6,000 = **30,000 steps** |
| Measured throughput | ~0.63 it/s at epoch 0 **[TBD final]** | 0.43 it/s (epoch 0), 0.46 it/s (epoch 4) |
| Wall time | **[TBD]** | 8 h 40 m total; epoch 0 4 h 45 m, epoch 1 3 h 41 m |
| Seed | 42 | 42 |

**Justify each of these, briefly:**
- *AdamW over Lion:* Lion updates on the **sign** of the gradient, which is too noisy at batch 1
  and is what forced gradient accumulation. AdamW normalises by a running second moment.
- *Accumulation 1:* follows directly from the optimiser change, and gives 4× the optimiser steps
  for the same wall time.
- *Clipping 1.0, not 0.1:* 0.1 was sized for Lion's bounded updates; under AdamW, whose gradient
  norms here are O(1), it would clip essentially every step and silently replace the LR schedule
  with a constant tiny step.
- *Batch 1:* two events gave no throughput gain and four exhausted an 80 GB A100.
- *Seed fixed at 42:* these runs are arms of ablations; LightningCLI seeds *randomly* when the key
  is absent, so omitting it is "an unrecorded seed", not "no seed".

**pu200 is trained as its own model**, not evaluated with pu0 weights — a model trained at one
occupancy and tested at another measures domain shift, not the architecture.

**Schedule sizing — worth one honest sentence.** Both schedules were sized from measured
throughput on the real run, not from a short benchmark. A 200-event benchmark returned 0.98 ev/s
against a real 0.58 ev/s at pu200, because the benchmark's working set fitted the dataloader's
row-group cache; and OneCycle sized from too few total steps never reaches its decay phase,
leaving the final checkpoint at a high learning rate.

### Post-processing (scored as a separate method, not as part of the model)
`maskformer_chained` applies **single-linkage chaining** to the model's output: unclaimed cells
within `chain_link_distance = 0.05` in `(η, φ)` of a cell already belonging to a cluster are
absorbed into it, iterated to a fixed point (at most 32 rounds). Nothing else is changed — the same
scorer, the same thresholds — so the difference between the two rows isolates the post-processing.
It is reported because it is the only post-processing of eleven tried that improved the headline
metric, and because a purely geometric repair working at all is itself informative.

### Inference
A cell is claimed if some query of validity ≥ **0.2** gives it a mask probability ≥ **0.5**.
- The validity threshold replaces an implicit argmax (equivalent to 0.5) and was chosen by a sweep
  on held-out tune events; it affects prediction only — never the loss, never the training matcher.
- **State honestly:** a full mask × object threshold scan on the pu200 checkpoint found F₁ **flat
  at 0.338–0.359 across the entire grid**, and the tool's own diagnostic reported that the object
  head accepts 92.9% of true particles while masks alone reproduce the full number — *the masks
  are the bottleneck*. A flat plateau means the cells are ordered wrongly rather than cut wrongly,
  and no threshold can reorder a ranking. This belongs in the methodology because it justifies not
  spending further effort on the working point.

---

## 8. §3.7 Computing environment

- **Hardware:** one NVIDIA A100 80 GB PCIe (MIG-enabled, single 7g.80gb instance — the whole card),
  104 CPU cores, 1.5 TB RAM.
- **Software:** PyTorch 2.9 / CUDA 12.8, `hepattn` at `cb4fb10`, CLUEstering 2.11.0, FastJet
  3.5.1.3, Optuna, scienceplots.
- Two environments by necessity: a training venv (torch + hepattn) and a conda analysis
  environment (CLUEstering must compile against Boost headers).
- **Reproducibility:** every run's resolved configuration is written next to its checkpoint by a
  `SaveConfig` callback, and the event store carries the checkpoint it was dumped from in its
  metadata, so a stale store and a new checkpoint disagree and are caught.

---

## 9. Things to state as limitations (an examiner will find them anyway)

Put these in the methodology where they arise, not buried in the discussion:

1. **pu0 and pu200 are not cross-comparable** — different η coverage.
2. **Tuning asymmetry.** CLUE received a systematic 80-trial search per sub-detector; MaskFormer
   received a three-arm objective sweep. The comparison is therefore *conservative* toward
   MaskFormer, which strengthens rather than weakens any MaskFormer advantage you report.
3. **Single seed.** No seed-variance estimate.
4. **The exclusive-partition approximation** discards the fractional truth (§3.4.1), and only
   0.78% of pu200 barrel cells are multi-owner anyway.
5. **Truncation** at the query cap (6/500 pu200, 18/500 pu0 events).
6. **Convergence.** Report the F₁-versus-steps trend (pu200: 0.312 → 0.423 → 0.523 at 4k/12k/30k
   steps, `val_loss` still falling at the final checkpoint) and scope claims to "at N steps".
   This is the single most likely line of attack; owning it with the trend as evidence is far
   stronger than silence.

---

## 10. Numbers still to fill in

| Item | Command |
|---|---|
| pu0 final throughput, wall time, epochs completed | read `external/train_pu0_dice.log` when done |
| pu0 store dump + score | `CKPT=<pu0 ckpt> ./scripts/run_pu200_pipeline.sh` pattern, with `dataset.active: pu0` |
| pu0 CLUE re-tune (new checkpoint ⇒ new store) | `python -m scripts.tune_clue` |
| pu0 cells/event, targets/event under the final config | `results/pu0/events_*.parquet` |
| pu0 unowned-cell fraction | same snippet used for pu200 |
| Working-point re-derivation for pu0 | `python -m scripts.scan_working_points` |
