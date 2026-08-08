# Why MaskFormer loses to CLUE above ~5 GeV, and what does not fix it

Written 2026-08-07, after the seven-epoch pu0 run (job 48247) and three one-epoch probe arms
(48255–48257). Everything here is measured on ColliderML ttbar pu0, events `[20250, 20750)` for the
headline numbers and `[20250, 20350)` for the probes.

**The finding is negative and it is solid.** The mechanism is identified and reproducible; three
targeted interventions failed to shift it, two of them significantly for the worse.

## 1. The result being explained

Efficiency at the 0.5 working point, 500 events:

| selection | CLUE | MaskFormer |
|---|---|---|
| all particles | 0.315 | **0.371** |
| E > 5 GeV | **0.297** | 0.239 |
| E > 20 GeV | **0.224** | 0.150 |

MaskFormer wins overall and below ~2 GeV, and loses above ~5 GeV. The crossover is stable: it is
present in the epoch-3 checkpoint, the epoch-6 checkpoint, and every probe arm.

## 2. The mechanism: a fixed cluster scale

Cells held by the matched cluster, against the particle's true cell count:

| particle E | true cells | MaskFormer | CLUE |
|---|---|---|---|
| < 2 GeV | 13.5 | 6.4 | 7.7 |
| 2–5 | 19.5 | 6.5 | 9.0 |
| 5–10 | 24.5 | 6.5 | 9.8 |
| 10–20 | 28.8 | 5.9 | 9.9 |
| **> 20** | **38.2** | **5.5** | **10.0** |

The mask head recovers ~6 cells whatever it is looking at, and slightly *fewer* as the shower grows.
CLUE scales weakly but in the right direction. A > 20 GeV shower is therefore split across ~4.9
predicted clusters, the matcher keeps one, and it holds a fifth of the energy. MaskFormer emits 753
clusters per event for 538 true particles; its median cluster is 8 cells against a truth median of
12.

The consequence is a distribution effect, not a mean effect. At E > 20 GeV the *mean* energy
recovered is 0.269 for MaskFormer against CLUE's 0.259 — MaskFormer is marginally better — but 61%
of its particles land in the 0.1–0.5 band and only 15% clear 0.5, while CLUE misses 36% outright
and clears 0.5 on 22.5%. A "≥ 0.5 recovered" metric rewards CLUE's bimodality.

## 3. What it is not

**Not a working point.** Dropping the mask threshold from 0.5 to 0.02 — a 25× change — moves
E > 20 efficiency from 0.143 to 0.152 and the unclaimed-energy fraction only from 0.403 to 0.358.
42% of a high-energy particle's energy is in cells no query claims at any probability the model
emits; it is not hiding under the cut.

**Not the energy weighting.** `overlay_metric_aligned.yaml` argues the loss counts cells while the
metric weights energy, and calls this "the likeliest reason" for this deficit. It is not supported:
`eff_e` exceeds `eff_n` in every energy bin for both algorithms (0.313 against 0.167 at E > 20), so
the model already captures energetic cells preferentially rather than ignoring them. That file's
exclusive-target half was tested separately and also failed — see below.

**Not undertraining.** Across seven epochs, cells recovered at E > 20 went 3.9 → 4.7 → 5.5 against a
true 38, and efficiency went 0.122 → 0.134 → 0.136. Five epochs after the first bought +0.002. The
pathology is fully formed after one epoch and is why the probes below are one epoch each.

## 4. The three probes, and their results

One epoch each, ~2.9 h, scored over the same 100 events against the epoch-0 checkpoint of run 48247
(the same one-epoch budget under the unmodified config). `slope` = cells recovered at > 20 GeV
divided by cells recovered at < 2 GeV; > 1 would mean clusters finally grow with the shower.
95% intervals from bootstrapping over events.

| arm | change | slope | eff > 20 GeV | jet core | all |
|---|---|---|---|---|---|
| baseline | — | 0.69 [0.65, 0.74] | 0.122 [0.111, 0.137] | 0.287 | 0.325 |
| 1 | `mask_attention: false` | 0.66 [0.61, 0.70] | 0.099 [0.085, 0.113] | 0.242 | 0.281 |
| 2 | incidence head restored | 0.72 [0.67, 0.77] | 0.116 [0.104, 0.129] | 0.294 | 0.335 |
| 3 | `particle_calohit_exclusive: true` | 0.67 [0.64, 0.71] | 0.086 [0.073, 0.098] | 0.285 | 0.325 |
| 4a | `posenc scale: 0.5` | 0.67 | 0.109 | 0.284 | 0.328 |
| 4b | `posenc scale: 0.2` | 0.69 | 0.118 | 0.277 | 0.319 |

**No arm improves the slope.** Every interval overlaps the baseline's. Arms 1 and 3 are
significantly *worse* on efficiency; arms 2, 4a and 4b are indistinguishable from the baseline on
everything.

Arms 4a/4b deserve their own note, because the hypothesis behind them was quantitative and it was
still wrong. `FourierPositionEncoder` induces similarity `exp(-2 pi^2 scale^2 d^2)`, a correlation
length of ~0.16 at the default `scale: 1`, and the measured aperture matched that length in BOTH
the angular and the longitudinal direction. Widening it 2x and 5x changed nothing — not the gain,
and not the jet-core cost that was supposed to be the trade.

**The reason it was mis-specified: the InputNet already feeds `x, y, z, r, eta, phi` as raw features
to its Dense layer.** Position is therefore available to the network whatever the encoder's
bandwidth does, so `posenc.scale` was never the bottleneck it was argued to be. That was visible in
`calo_clustering.yaml` before the arms were run.

## 4b. Nor is it the working point, at any setting

Post-hoc sweep on the epoch-6 checkpoint, 120 events, thresholds taken to the floor the store
supports:

| mask | object | eff > 20 | unclaimed energy > 20 | eff < 2 | all |
|---|---|---|---|---|---|
| 0.50 | 0.20 | 0.140 | 0.405 | 0.428 | 0.374 |
| 0.50 | 0.05 | 0.143 | 0.397 | 0.440 | 0.382 |
| 0.10 | 0.01 | 0.146 | 0.378 | 0.455 | 0.395 |
| 0.02 | 0.001 | 0.152 | 0.353 | 0.467 | 0.408 |

With filtering effectively switched off — a cell joins if any query gives it 2% and any query
survives a 0.1% object cut — high-energy efficiency reaches 0.152 and **35% of the particle's
energy is still in cells no query assigns even 2% to.** The object head is not discarding
high-energy clusters, and the model is not hedging near the threshold: it confidently excludes
those cells.

The masked-attention hypothesis was the leading one and it is the one most clearly refuted. The
argument was that `mask_attention: true` restricts each query to the cells its own previous-layer
mask claims, so a query that starts small can never discover the rest of a large shower — the
standard Mask2Former pathology, which predicts exactly a saturating cluster size. Turning it off
changed nothing about the scale and cost convergence speed. One caveat survives: removing masked
attention is expected to slow convergence, so a one-epoch arm is somewhat unfair to it. But `slope`
was chosen because it is more schedule-robust than efficiency, and it did not move either.

Arm 2 deserves one note: it is the only arm that did not hurt, and it restores
`maskformer_incidence` as a scoreable algorithm, which the current config cannot produce
(`has_incidence: false`). On the epoch-3 checkpoint, scoring the same model through incidence rather
than the mask head lifted E > 20 from 0.147 to 0.173. That effect is an assignment-rule change at
inference, and it is not reproduced here as a training-time gain.

## 5. RESOLVED, in post-processing rather than in the model

Two CPU-only steps applied to the existing checkpoint and the committed event store close the gap
and reverse it. Measured on the full 500-event eval window, both methods scored on identical events,
with the merge classifier fitted on the TUNE window and its threshold chosen there by a rule fixed
in advance (highest efficiency whose purity still matches CLUE's):

| method | eff | purity | eff > 5 GeV | eff > 20 GeV |
|---|---|---|---|---|
| CLUE (tuned) | 0.315 | 0.251 | 0.297 | 0.224 |
| **MaskFormer + merge + chaining** | **0.414** | 0.248 | **0.371** | **0.262** |

Paired bootstrap over 500 events: overall **+0.100** [+0.097, +0.102], above 20 GeV **+0.038**
[+0.030, +0.046]. The high-energy interval excludes zero. 0.262 is 89% of the geometric ceiling of
0.295, against CLUE's 76%.

The two steps, in `src/postproc/`:

- **`merge.py`** decides which predicted clusters are fragments of one shower, with a small
  classifier over cross-claims, adjacency and energy. Its dominant feature by 2x is `cross_max` --
  the mask probability one query assigns to ANOTHER cluster's cells. Geometry cannot make this call
  (siblings sit 0.048 apart, distinct particles 0.008), and the mask head can.
- **`chain.py`** then grows the merged clusters outward through neighbouring cells.

Order matters and so does restraint: **every method that reassigned cells the model had already
placed made things worse.** `flow.py` re-partitioned everything by density and scored 0.267 against
a 0.377 baseline; `axis.py` grew along the shower axis and never beat chaining on both axes at once.
Chaining and merging succeed because neither overwrites a core -- chaining only fills unclaimed
cells, merging only relabels which cluster a group belongs to.

Still owed before this is quoted: a cross-validated AUC in place of the in-sample 0.831, the merger
wired into `scripts/score.py` as a first-class algorithm, and a `tune_merge.py` that fits and
persists the model the way `tune_clue` does.

## 5b. Where the remaining loss is, and two things that do not recover it

Energy budget per particle above 20 GeV, after merging and chaining (100 events):

| pipeline | in matched cluster | unclaimed | in another cluster |
|---|---|---|---|
| raw model | 0.265 | 0.410 | 0.324 |
| merge + chaining | 0.309 | **0.086** | **0.605** |
| CLUE | 0.259 | 0.178 | 0.562 |

**Coverage is finished** — 8.6% unclaimed against CLUE's 17.8%. Every remaining point is attribution:
which of several overlapping showers owns a contested cell.

Two attempts, both instructive:

- **Fractional scoring** (`scripts/score_soft.py`, existing machinery). The soft masks recover
  **60.2%** of "impossible" particles — those owning no cell exclusively — against CLUE's **1.1%**,
  which is a capability a partitioning method cannot have. But they over-claim, at 1.74 effective
  claims per cell against truth's 1.16, so the headline soft efficiency is *worse* (0.283 vs CLUE's
  0.311). Worth quoting as a capability result, not as a win.
- **Learned per-cell attribution** (`src/postproc/attribute.py`, 5-fold CV AUC 0.765). Its dominant
  feature by 3x is `d_cell`, the distance to the nearest cell already in the cluster — i.e. it
  rediscovered the proximity rule chaining already applies, and stacking it on chaining gives
  bit-identical results. Used alone it is a genuine high-purity operating point (0.374 / 0.324,
  dominating CLUE's 0.317 / 0.255) but it loses above 20 GeV.

### How much of the misassignment is real error rather than shared energy

Measured after merge + chaining, over 478,134 assigned cells: **48.6% sit in the wrong particle's
cluster**, carrying 48.8% of the assigned energy. Splitting those by how much of the cell the true
owner actually contributed:

| owner's share | of correct cells | of misassigned cells |
|---|---|---|
| 0.50-0.70 (contested) | 4.1% | 12.0% |
| exactly 1.0 (sole contributor) | 92.8% | **82.1%** |

**82.1% of misassigned cells have a single contributing particle**, so the truth is unambiguous and
the assignment is simply wrong. Only ~12% is genuinely shared energy that exclusive scoring cannot
represent. An earlier draft of this document claimed the reverse -- that much of the misassignment
was irreducibly shared -- and that was wrong by roughly a factor of seven. The headroom is real.

It also puts chaining in its place: it wins the >= 0.5 metric by claiming cells indiscriminately and
being right often enough, not by assigning well.

The conclusion is that **the information needed to attribute contested cells is not in the store**.
The mask head has no opinion on 82% of the cells requiring a decision, so no post-processing can
make that decision better. Improving this needs either a model that represents shared cells (the
incidence head, whose soft output is the 60.2% above) or information the calorimeter alone does not
carry.

Cross-validated AUCs, replacing the in-sample figures quoted earlier: merge 0.819, attribution 0.765.

## 5c. Splitting, and the representation probe -- both negative

**Splitting over-merged clusters at local energy maxima** (`src/postproc/split.py`, the ATLAS
topocluster move) makes things worse on BOTH axes: purity 0.251 -> 0.215 at the mildest setting and
0.145 at the most aggressive, with misattributed energy unchanged (0.604 -> 0.607). Two showers
0.008 apart do not leave two resolvable energy maxima at this granularity, so the watershed carves
clusters along arbitrary lines. The ATLAS analogy does not transfer: their splitting separates
showers that are genuinely separated, and these are not.

**The encoder does not represent co-membership either.** `dias/probe_encoder_affinity.py` takes the
post-encoder cell embeddings straight off a forward hook and asks how well their cosine similarity
predicts that two nearby cells share a particle, against plain 3D distance on the same pairs:

    400,000 pairs within 0.06 m, 18.4% same-particle
      encoder embedding cosine   AUC 0.683
      plain 3D distance          AUC 0.661

+0.022 over geometry. So there is nothing to read out of the trained model, and affinity-driven
chaining is not available without retraining.

### The pattern across every post-hoc attempt

Three independent probes reach the same place: the learned per-cell attribution model's dominant
feature was distance by 3x; stacking it on chaining changed nothing; and the encoder's embeddings
barely beat distance. **Every signal available after training reduces to proximity**, because
nothing in the objective ever asked for a relation between cells -- the mask head is a per-cell
independent sigmoid by construction, and the encoder was never given a reason to encode
co-membership.

That makes the remaining avenue specific rather than speculative: an auxiliary cell-pair affinity
head, trained jointly so the encoder is forced to represent the relation. It needs a new task class
in `hepattn` (a fourth entry in `hepattn-changes.patch`) and a training run, so it is the first step
here that is not a config change or CPU post-processing.

## 5d. The auxiliary affinity head (arm 5) -- negative, and it degraded the encoder

`hepattn_colliderml/affinity.py` adds a `ConstituentAffinityTask` to `encoder_tasks`: it projects
post-encoder cell embeddings into a 32-dim space and trains the cosine similarity of nearby cell
pairs to predict shared truth ownership. It needs no patch to hepattn -- the task lives in this
repository and is named by `class_path` -- and it changes nothing about the mask head, the
classification head, the matcher or the decoder.

One epoch (job 48300, 2:53). Measured with `dias/probe_encoder_affinity.py`, 400k local cell pairs:

| checkpoint | raw encoder AUC | affinity-projected AUC | plain distance |
|---|---|---|---|
| 7-epoch baseline | 0.683 | — | 0.661 |
| **1-epoch baseline (matched control)** | **0.688** | — | 0.661 |
| 1-epoch + affinity head | 0.652 | **0.619** | 0.661 |

**The head made the encoder worse** (0.688 -> 0.652) and its own trained projection reached 0.619,
below a ruler. Note the projection is a function OF the encoder embedding, so it cannot carry more
information than the embedding does -- 0.619 < 0.652 means the head lost information it was handed,
which is an optimisation or conditioning failure in the head rather than proof the relation is
unlearnable. Candidate causes, untested: `loss_weight: 1.0` contributing ~1.2 of a ~5 total and
pulling the encoder off the main objective; the `pos_weight` rebalancing at 18.4% positives; the
normalise-and-learnable-temperature projection under bf16 autocast.

The clustering metrics moved not at all (slope 0.70 against baseline 0.69, eff>20 0.107 against
0.122), which was predicted in advance -- the head shapes the encoder and cannot reach the mask
head's output in one epoch, so those numbers were never the test.

### The finding that survives the head's failure

**The encoder's co-membership signal is real, weak, and flat with training**: AUC 0.688 at one epoch
and 0.683 at seven, against 0.661 for plain 3D distance. It picks up whatever relational structure
it gets almost immediately, never refines it, and ends only +0.027 above geometry. That is a
property of the trained model rather than of any post-processing, and it explains why every
downstream method reduced to proximity.

## 5e. THE ENCODER DOES CARRY THE RELATION -- raw cosine was just a bad readout

This overturns the conclusion in 5c. Encoder embeddings were extracted for 150 events with
`dias/extract_embeddings.py` (LIGHTGPU, inference only, 4:54) and a classifier was fitted on the
tune window and scored on the eval window, over cell pairs within 0.06 m:

| model | eval AUC |
|---|---|
| plain 3D distance, no model | 0.670 |
| raw embedding cosine, no model | 0.671 |
| learned: geometry only | 0.742 |
| learned: embedding only | **0.789** |
| learned: geometry + embedding | **0.817** |

**Embeddings add +0.075 AUC over geometry alone.** The relation IS in the encoder.

Every earlier negative followed from measuring the wrong thing. `probe_encoder_affinity.py` compared
raw cosine (0.671) against distance (0.670), found them level, and concluded the encoder knew
nothing -- but cosine over 256 dimensions is dominated by variance unrelated to co-membership, and a
learned readout of the SAME vectors reaches 0.789. The per-cell attribution model in 5b reached only
0.765 and "rediscovered proximity" because embeddings are not in the event store and it never saw
them. Arm 5's projected space scored 0.619 because a randomly-initialised head trained for one epoch
against a competing objective, not because the relation is unlearnable.

### What this makes available, none of it needing the contended GPU partition

1. Store the embeddings: a change to `eval/dump.py` and a re-dump on LIGHTGPU (~10 min for 500
   events, ~5.5 GB at float16).
2. Replace `chain.py`'s tie-break -- currently "nearest already-claimed cell", the crudest decision
   in the pipeline -- with a learned affinity over embeddings and geometry. It governs most of the
   misassigned energy: 48.6% of assigned cells land in the wrong cluster and 82.1% of those have a
   single unambiguous owner.
3. Re-fit the per-cell attribution model of 5b with embeddings included.

**A higher pair AUC does not automatically become a higher clustering efficiency** -- the 0.765
attribution model proved that by scoring well and changing nothing end to end.

### ...AND IT DID NOT TRANSLATE EITHER. The end-to-end test, in full

Embeddings were extracted for the whole 500-event eval window (`dias/extract_embeddings.py`,
LIGHTGPU, 14:53, 4.2 GB), a classifier was fitted on the tune window, and `chain.py` gained an
`affinity=` callable that replaces the nearest-neighbour tie-break with the model's score.

| variant | eff | pur | eff>5 | eff>20 | misattributed>20 |
|---|---|---|---|---|---|
| CLUE | 0.318 | 0.255 | 0.304 | 0.230 | 0.553 |
| distance, link 0.05 | 0.418 | 0.251 | 0.375 | 0.253 | 0.596 |
| affinity, link 0.05 | 0.418 | 0.251 | 0.375 | 0.251 | 0.596 |
| distance, link 0.10 | 0.434 | 0.236 | 0.392 | 0.257 | 0.637 |
| affinity, link 0.10 | 0.434 | 0.236 | 0.394 | 0.257 | 0.637 |
| distance, link 0.15 | 0.438 | 0.231 | 0.394 | 0.261 | 0.649 |
| affinity, link 0.15 | 0.438 | 0.230 | 0.395 | 0.262 | 0.648 |

Indistinguishable at every radius, and the gap does not open as the candidate set grows -- which was
the hypothesis after the first null. Widening the radius alone only slides along the usual trade
curve: efficiency up, purity down, misattribution UP.

**Why a +0.075 AUC signal changes nothing.** Two measurements explain it. Of unclaimed cells with a
claimed neighbour in range, only **36.9%** have candidates spanning more than one cluster, so the
tie-break is a live decision for barely a third of them; and the correct particle's cluster is among
the candidates at all only **39.5%** of the time. For three cells in five the right answer is not on
the menu, and no scoring rule over the menu can help. Where the decision IS live, the affinity model
ranks a cell's few candidates the same way distance does, even though it separates the global pair
population much better.

**THE METHODOLOGICAL LESSON, which cost two builds to learn: pair-level AUC is a poor proxy for
clustering gain.** AUC is global separability; a tie-break needs local ranking within a handful of
candidates. A model can win the first and be identical on the second. Both the attribution model
(0.765) and this one (0.817 offline, 0.747 operationally) did exactly that.

The embedding signal is real and the extraction is kept -- it is the right starting point if anyone
revisits this at the level where it could matter, which is growing a cluster beyond the ~6-cell
aperture in the first place, not arbitrating between cores that are already too small.

## 5f. Two further post-processing negatives

**Incidence-guided attribution.** On the fully-trained ep003 checkpoint, the incidence head
re-attributes mask-claimed cells slightly WORSE than the mask head itself (73.3% against 75.0%), and
on cells the mask leaves unclaimed it reaches only 32.0% and gets 14.6% right against proximity's
26.4%. Note also that `maskformer_incidence_labels` is gated on `_claimed_cells`, so incidence can
only re-attribute cells the mask already claims -- it can never reach new ones.

**Iterating merge -> chain.** Identical to three decimals at one, two and three rounds
(0.413/0.251 -> 0.412/0.251). The pipeline reaches its fixed point after a single pass, because
chaining leaves only 8.6% of energy unclaimed for a second round to work with.

## 6. What is left on the training side

The fragmentation is worth fixing if it can be: merging the fragments each particle dominates — an
oracle, so a ceiling rather than a method — takes E > 20 from 0.183 to 0.294, clearing CLUE's 0.224.
The information is in the model; the assignment discards it.

Untested and cheap, in order of how much the evidence supports them:

- **`costs: object_bce: 1.0`** on the classification head. It is currently `0`, so the matcher
  decides correspondence on mask DICE alone and the "is this query real" decision is decoupled from
  matching. With 1000 queries for 538 particles, ~460 must be suppressed by `object_bce` at loss
  weight 1.0 with no help from the cost. `overlay_metric_aligned.yaml` already sets this to 1.0.
- **A post-hoc merger** over predicted clusters (high mask overlap, spatial contiguity), which needs
  no GPU at all and chases the 0.183 → 0.294 headroom directly.
- **DICE's size normalisation** is the deeper suspect and is not a flag: a per-query DICE rewards
  covering 6 of 6 cells of a fragment exactly as much as 38 of 38 of a shower, so nothing in the
  objective prefers one correct large cluster over several correct small ones. Testing this means
  changing the loss, not the config.

Set against that, three targeted config-level interventions have now failed, so the prior on the
next one should be lower than it was on these.
