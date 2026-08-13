# Methodology chapter — writing plan

Rewritten 2026-08-13. **This describes only the current setup.** Everything superseded — the
incidence head, the dice-20 + focal-1 objective, the per-Geant-particle truth, the mask-head
variants, the probe arms, the chaining post-processor, the multi-owner capability study — has been
removed rather than annotated. `git log -- docs/METHODOLOGY_PLAN.md` recovers the earlier version
if a negative result needs citing.

Numbers marked **[pending]** are not yet measured. Do not invent them.

---

## 1. §3.1 What the comparison is, and what makes it controlled

One sentence for the claim: **a classical density-based clustering algorithm (CLUE) and a learned
set-prediction model (MaskFormer) are given identical calorimeter cells, asked to reconstruct
identical target particles, and scored by identical code.**

The mechanism is the **event store**, and it is the main design decision worth a paragraph. The two
methods do not each read the ColliderML release and apply matching cuts. A single store is dumped
once from the *model's own dataloader*; CLUE clusters the cells in that store. "Both algorithms saw
identical input" is therefore structural rather than a promise two config files make to each other.

Three consequences worth stating:

- Every selection — cell energy threshold, particle cuts, the truth definition — is applied once,
  by the dump, and travels inside the store as metadata.
- `config/experiment.yaml` holds *expectations*, which `src/io/event_store.py` checks against that
  metadata and refuses to proceed on a mismatch. A store produced under a different truth
  definition cannot be scored against the wrong config.
- The scoring code never learns which method produced a clustering. It takes a label per cell plus
  the truth partition and returns numbers.

## 2. §3.2 Dataset, cells and the two conditions

ColliderML ttbar, two pileup conditions. The detector is a high-granularity calorimeter with four
subsystems — ECAL barrel/endcap, HCAL barrel/endcap — of 48, 48, 36 and 36 layers.

| | pu0 | pu200 |
|---|---|---|
| events on disk | 100,000 (100 shards × 1,000) | 10,000 (100 shards × 100) |
| cells/event after zero suppression | ~22,000 | ~117,000 |
| cell energy threshold | 2 × 10⁻⁴ GeV | 2 × 10⁻⁴ GeV |
| cell \|η\| cut | 4.0 | **≤ 0.88** |
| train / val / test | [0, 20000) / [20000, 20250) / [20250, 20750) | [0, 6000) / [6000, 6250) / [6250, 6750) |
| CLUE tune store | [20000, 20050), 50 events | [7000, 7050), 50 events |
| evaluation store | [20250, 20750), 500 events | [7500, 8000), 500 events |

**Barrel-only at pu200.** The HCAL barrel reaches r = 3441 mm at |z| ≤ 3450 mm, so a particle
steeper than η = 0.883 leaves through the barrel end and deposits the rest of its shower in the
endcap. Cutting at 0.88 keeps every target fully contained. Applied to **cells as well as
particles** — cutting cells alone would leave targets whose cells had been removed.

## 3. §3.3 The target definition — one definition doing two jobs

This is the most important subsection in the chapter and deserves the most space. The same
definition is simultaneously the model's training target and the denominator of CLUE's efficiency.

**A target is one particle entering the calorimeter, not one Geant particle depositing in it.**

ColliderML's particle table is the full Geant record: when a particle showers, every secondary it
produces is its own row, and those secondaries produce their own. Each deposit is re-pointed at the
ancestor that *entered* the calorimeter — climb `parent_id` while the particle was born inside the
calorimeter volume, stop at the first one produced before the front face.

**Deliberately not the generator-level root.** A π⁰ decays to two photons at the primary vertex;
those are two genuinely separate showers and collapsing to the root would fuse them (measured 91
targets/event at 137 cells each — too coarse). Stopping at the calorimeter face keeps them apart.

The front face is measured from the calo-hit cloud, not assumed: ECAL barrel spans r = 1252–1519
mm, ECAL endcap starts at |z| = 3202 mm. The answer is insensitive to moving it inwards
(r > 1000 / |z| > 2800 gives 156 targets/event against 176) and only breaks if pushed *inside* the
ECAL, where secondaries made in the first layers stop counting as shower products (r > 1400 gives
401).

### Why the per-Geant-particle alternative was rejected — quote these measurements

Measured over 2,800 pu0 events:

| | per-Geant-particle | shower-level |
|---|---|---|
| targets/event | 536 | 186 |
| non-primary secondaries | 85.7% | — |
| born inside the calorimeter | 71.8% | — |
| in a shower split into several targets | 83% | — |
| median separation of sibling fragments | ΔR 0.045 (37% within 0.02) | — |
| calorimeter energy owned by a target | **0.317** | **0.864** |
| PDG composition | 21% p, 37% e±, **1% γ** | 43% π±, **14% γ** |

Two sentences carry the argument. Fragments separated by ΔR 0.02 sit inside one shower core and
cannot be told apart by any algorithm, so that definition capped efficiency and purity for reasons
unrelated to the method being measured. And a ttbar calorimeter truth that is 1% photons and 21%
protons is describing nuclear spallation products and shower electrons, not the physics objects
anyone wants reconstructed.

### The remaining cuts

| cut | value | why |
|---|---|---|
| pT | ≥ 0.5 GeV | |
| \|η\| | ≤ 4.0 (pu0), ≤ 0.88 (pu200) | containment, see §3.2 |
| calorimeter cells | **≥ 10** | fewer is not reconstructable; costs 7% of targets (183 → 170/event) and almost nothing in coverage (0.819 → 0.816) |

**Cells belonging to no target are kept, deliberately.** They are inputs, not targets. Under the
shower-level definition 86.4% of pu0 calorimeter energy belongs to some target; the remainder is
from particles entering below the pT floor. State that 13.6% as an explicit truth floor — it is the
ceiling on any energy-based metric and it is honest to name it rather than let a reader discover it.
At pu200 the same figure is 32.5%, and the condition is correspondingly harder.

### Query budget and truncation

`event_max_num_particles` must equal the model's `num_queries`, and targets beyond it are **silently
truncated** by the dataloader. Both were sized from measurements, not chosen:

| | targets/event | p99 | max | queries | events truncated |
|---|---|---|---|---|---|
| pu0 | 186 | 361 | 488 | **400** | 0.32% (realised: 1 in 500) |
| pu200 | 1129 | 1481 | 1712 | **1600** | 0.125% |

State the truncation rate as a limitation. The event store carries a `truncated` column so it is
checkable after the fact rather than assumed.

## 4. §3.4 Evaluation

### Matching

Global **one-to-one assignment** by `scipy.optimize.linear_sum_assignment`, maximising shared
calibrated energy. Justify the choice against the two alternatives:

- The model's own training metric is not reusable — it compares query *i* with target *i*, which is
  only meaningful because the loss has already Hungarian-permuted them. CLUE has no such
  permutation.
- A greedy "assign each cluster to whichever particle contributed most of its energy" is not
  symmetric: one particle can win several clusters while another wins none, and a particle merged
  into a neighbour vanishes from the denominator instead of counting as a miss.

Unmatched targets are inefficiencies; unmatched clusters are fakes. Both are reported.

### Metrics

- **Truth partition: exclusive.** Each cell belongs to the particle depositing the most energy in
  it. CLUE produces a partition and cannot do otherwise, so scoring it against multi-owner truth
  would be structurally unfair. **[pending]** re-measure what the exclusive partition discards
  under shower-level truth — the old figures (83% of associations, 94% of energy) were measured
  under the per-fragment definition and most of what they discarded were cells shared *between
  fragments of one shower*, which no longer exist as separate targets.
- **Energy-weighted**, not cell-counted: a calorimeter measures energy and cell energies here span
  orders of magnitude. `eff_e` = matched cluster's share of the particle's deposit; `pur_e` = the
  particle's share of the matched cluster.
- **Working point 0.5**, with 0.75 and 1.0 available.
- **Match floor** `min_overlap_frac = 0.05` of the smaller of the two totals, so a cluster merely
  grazing a particle does not count as reconstructing it.
- **Split / merge** at a 0.10 fraction of calibrated energy.
- **Uncertainties** from bootstrapping **events**, not particles — particles within an event are
  not independent.

### Differential binning — bin by pT

All differential figures bin by **truth particle pT**, in `[0.5, 1, 2, 5, 10, 20, 50, 200]` GeV.
Energy remains the metric *weight*; only the binning variable is pT.

Worth one sentence of justification, because it changes a conclusion: on this sample median E/pT is
1.75 and 10.9% of targets sit beyond |η| = 2.44, so binning by energy mixes "energetic" with
"forward" and manufactures a high-energy fall that is partly geometry. Binned by pT the same
efficiency curve dips and recovers (0.590 → 0.682 in the top bin) rather than falling away. It is
also the variable the comparison literature plots.

### Working points are chosen on a disjoint window, by one criterion

Both methods' free parameters are chosen on the **tune store**, never on the evaluation events, and
both by **f1**. This symmetry is the point and should be stated explicitly — neither method may pick
its operating point on the events it is reported over, and neither may pick it by a friendlier rule.

- MaskFormer: `mask_threshold` × `object_threshold` grid. Current: **0.05 / 0.5**.
- CLUE: Optuna, 80 trials per subsystem, objective `cluster_f1`.

## 5. §3.5 The CLUE baseline

A two-stage pipeline over calorimeter cells, using the CLUEstering library unmodified.

1. **Per-layer (2D).** Cluster within each detector layer in (η, φ), with φ treated as periodic.
2. **Linking (3D).** Reduce each layer cluster to a centroid, then cluster the centroids in
   (η, φ, depth) to link the per-layer pieces of one shower into a trackster.
3. **Cross-subsystem linking.** Union clusters in *different* subsystems whose energy-weighted
   (η, φ) centroids are within `link_radius = 0.05`, transitively.

Running CLUE once in three dimensions merges showers that overlap in depth, which is why stage 1
exists. Stage 3 is new and needs justifying honestly (below).

**Each subsystem is tuned independently** — ECAL layers sit 5.05 mm apart and HCAL layers 51 mm, so
one density radius cannot mean the same thing in both. Five parameters per subsystem
(`d_c_2d`, `rho_c_2d`, `d_c_3d`, `rho_c_3d`, `depth_scale`), with the two outlier distances sampled
as a multiple of their density radius.

### Stage 3 needs an honest paragraph

CLUE clusters one subsystem at a time, so a shower crossing ECAL into HCAL is split by
construction. Under the per-fragment truth that cost little (24.6% of targets spanned a boundary);
under shower-level truth it is **42.2% of targets carrying 10.8% of target energy**. Without
linking, CLUE could not represent 42% of the target set, and its efficiency would be capped for a
reason belonging to the harness rather than to the algorithm — the mirror image of the unfairness
that collapsing the truth removed from the model's side.

**But report that it barely helps:** f1 0.5056 unlinked against 0.5070 at r = 0.05, which is noise
on 50 events. It is kept on the split/merge trade — 2.4 points of splitting removed for 0.2 added —
not on f1. The reason efficiency does not rise is worth a sentence: at pu0 the angular separation
between *different* showers is comparable to the offset between one shower's two halves, so a
centroid-distance criterion cannot tell them apart and the gains and losses cancel. A depth-aware
criterion, requiring the two clusters to be radially consecutive and roughly collinear from the
origin, is the obvious improvement and is not implemented.

## 6. §3.6 The MaskFormer model

Built from `hepattn`, an existing library for transformer-based reconstruction. State clearly what
is not yours: the transformer encoder, the MaskFormer decoder, the task heads, the loss functions
and the Hungarian matcher. Three small modifications to the library are kept as a patch
(`hepattn-changes.patch`) so what changed is separable from what was already there.

### Inputs

Calorimeter cells only — no tracker, no muon system. Per cell: `x, y, z, r, η, φ, log_energy`.

- `log_energy` rather than raw energy: the input net applies no normalisation and raw cell energy is
  ~1000× smaller than the O(1) coordinates, so the single most important feature was numerically
  drowned out.
- **FourierPositionEncoder** on `(r, η, φ)`, not the default position encoder, whose frequencies
  alias at the scale that matters here — a shower spans ~0.007 in η — giving cells of the same
  particle decorrelated encodings. Measured 2.3× more locality signal.

### Architecture

| | |
|---|---|
| dim | 256 |
| encoder | 4 layers, flash-varlen attention, window 1024 with wrap-around, hybrid norm, SwiGLU |
| decoder | 4 layers, mask attention on, 400 queries (pu0) / 1600 (pu200) |
| matcher | scipy solver, `parallel_solver: true`, adaptive off |

Cells are sorted by **φ** before windowed attention, so that field decides which cells share a
window.

### Heads and objective — two heads, no incidence head

1. **Object classification** (`flow_valid`) — is this query a real particle or an empty slot?
   Decides *how many* objects there are.
2. **Mask head** — independent sigmoid per (query, cell). Decides *which cells* belong to each.

```
losses:  0.1 · object_bce  +  1.0 · mask_dice  +  1.0 · mask_bce
costs:   0.1 · object_bce  +  1.0 · mask_dice
```

Applied after every decoder layer. `null_weight` 1.0 throughout.

**The distinction between `losses` and `costs` is worth a sentence**, because it is not obvious and
it mattered: `losses` are what gradients minimise; `costs` are what the Hungarian matcher uses to
pair queries with truth particles, and are detached. A term with zero cost weight influences the
gradient but not the pairing.

**Why `object_bce` appears in both.** An earlier configuration had its cost weight at zero, so
queries were matched on mask overlap alone: a query the object head confidently called fake could
still be handed a real particle and was then trained to call itself real — the head deciding *how
many* objects exist was excluded from deciding *which*. This is worth reporting as a methodological
finding, with the measurement that identified it: across the entire mask-threshold range f1 moved
only 0.546–0.548, while the object threshold alone moved it 0.512 → 0.548 and the cluster count
310 → 160 against 185 true targets. The masks were near-bimodal and essentially solved; object
cardinality was the whole bottleneck.

### Training

| | pu0 | pu200 |
|---|---|---|
| optimiser | AdamW | AdamW |
| LR schedule | OneCycle, 1e-5 → 1e-4 → 1e-6, `pct_start` 0.01 | same |
| weight decay | 1e-5 | 1e-5 |
| gradient clipping | 1.0 | 1.0 |
| `accumulate_grad_batches` | 1 | 1 |
| batch size | 1 event | 1 event |
| epochs × events | **6 × 20,000 = 120,000 steps** | **[pending — see below]** |
| seed | 42 | 42 |

Justify briefly:

- **AdamW over Lion.** Lion updates on the *sign* of the gradient, too noisy at batch 1, which is
  what forced gradient accumulation and quartered the optimiser step count. AdamW normalises by a
  running second moment.
- **Clipping 1.0, not 0.1.** 0.1 was sized for Lion's bounded updates; under AdamW, whose gradient
  norms here are O(1), it would clip essentially every step and silently replace the LR schedule
  with a constant tiny step.
- **Batch 1.** Two events gave no throughput gain and four exhausted an 80 GB A100.
- **Seed fixed.** LightningCLI seeds *randomly* when the key is absent, so omitting it is "an
  unrecorded seed", not "no seed".
- **pu200 is trained as its own model**, not evaluated with pu0 weights — a model trained at one
  occupancy and tested at another measures domain shift, not the architecture.

**Schedule sizing deserves one honest sentence.** OneCycleLR is constructed with
`total_steps = trainer.estimated_stepping_batches`, so the schedule length is fixed before the run
starts and cannot be extended afterwards: resuming a finished checkpoint under a longer schedule
fails on the restored scheduler state, and would in any case restart at a learning rate already
decayed to 1e-6. Both schedules must be sized from a measured rate on the real machine, not from a
short benchmark — a 200-event benchmark returned 0.98 events/s against a real 0.58 at pu200,
because the benchmark's working set fitted the dataloader's row-group cache.

**pu200's schedule is the one number still open.** Its config carries `max_epochs: 3` as a
placeholder. Launch, watch ~300 steps, size it from the measured rate, relaunch.

### Inference

A cell is claimed if some query of validity ≥ **0.5** gives it a mask probability ≥ **0.05**. Both
thresholds are post-hoc — the store keeps mask probabilities down to 0.02, so any working point
above that is re-derived offline with no GPU — and both were chosen on the tune store by f1.

Worth reporting: the grid is **flat in the mask direction and not in the object direction**. At
object 0.5, f1 moves only 0.7073–0.7099 across the entire mask range; at mask 0.05 it moves
0.5175–0.7099 across the object range. The cell probabilities are near-bimodal; how many objects
there are is the harder question.

## 7. §3.7 Computing environment

Single NVIDIA A100 80 GB (ce-ai-1), no scheduler, no container. Training is the only GPU step;
so is the event-store dump. Everything downstream — CLUE, matching, scoring, plotting — runs on
CPU against the store and depends on nothing but numpy, scipy, pandas and matplotlib.

State the wall times once measured. **[pending]** pu0 at 6 epochs; pu200 unsized.

## 8. Limitations to state, because an examiner will find them anyway

1. **Calorimeter only.** No tracker, no muon system, so this is calorimeter clustering rather than
   particle flow. Comparisons to published full-detector results are not like-for-like, and the
   difference is not small: in a jet roughly 60% of the energy is charged hadrons, and a method
   with tracking can attribute that and read the neutrals off the remainder.
2. **13.6% of pu0 calorimeter energy (32.5% at pu200) belongs to no target**, from particles
   entering below the pT floor. This is a floor on any energy-based metric.
3. **Truncation** at the query budget: 0.32% of pu0 events, 0.125% of pu200 events.
4. **No isolation or primary-particle requirement** in the target definition, unlike the
   reconstructability criteria used in some published comparisons. The denominator here includes
   particles sitting on top of one another. On this sample that matters a lot: median ΔR to the
   nearest other target falls from 0.116 in the lowest pT bin to 0.019 in the highest, and
   restricting to isolated targets turns the efficiency curve from falling to rising.
5. **CLUE's cross-subsystem linking uses centroid distance only**, which cannot separate a shower's
   two halves from two neighbouring showers.
6. **The models are step-limited, not converged.** Validation loss was still falling at the final
   checkpoint of the 3-epoch pu0 run, which is why the schedule was doubled.

## 9. Numbers still to fill in

- pu0 results under the 6-epoch / 120,000-step run — every headline number
- pu200: the whole condition, once trained
- CLUE re-tuned against the 10-cell truth (its current parameters are from the 3-cell tune)
- exclusive-partition loss under shower-level truth
- wall times for both conditions
