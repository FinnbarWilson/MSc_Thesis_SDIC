# Discussion chapter — writing plan

Everything held back from `RESULTS_PLAN.md`. Section 4 states what was measured; Section 5
explains it, bounds it, and says what follows. Numbers repeated here are for the writer's
convenience — in the text, cite the result rather than restating it.

**Evidence grading used throughout.** Label every claim in your own head as you write:

- **[M]** measured in this work
- **[I]** inferred, with the supporting measurement named
- **[H]** hypothesis, untested, with the test that would settle it

An examiner will forgive an unproven hypothesis that is *labelled*. They will not forgive one
presented as a finding.

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

## 1. Recommended structure

```
5.1  Principal findings                  three paragraphs, no new evidence
5.2  How each method fails               mechanism, grounded in §4.3-4.5
5.3  Which failure matters for physics   bias, variance, and particle flow
5.4  Why MaskFormer's clusters do not scale   diagnosed limitations of the formulation
5.5  What the interventions tell us      the negative results, and what they bound
5.6  Threats to validity                 own these before the examiner raises them
5.7  Relation to prior work
5.8  What a good calorimeter clusterer would do    the synthesis
5.9  Future work
```

§5.6 must come *before* §5.8. A synthesis offered before the caveats reads as overreach; offered
after, it reads as considered.

---

## 1b. Which arguments in this chapter depend on pu0

Write the chapter so these survive either outcome. Each is currently supported by pu200 alone.

| argument | status without pu0 | what pu0 would add |
|---|---|---|
| Bias vs variance (§5.3) | **stands** on pu200 | shows whether CLUE's variance is pileup-driven |
| Fixed cluster scale (§5.4) | **stands** on pu200; pu0's earlier run showed the same pathology under a different objective | confirms it is formulation-bound, not occupancy-bound |
| Jet-level advantage (§5.3 step 4) | **stands** on pu200 | shows whether it survives without pileup to absorb |
| "A good clusterer needs intrinsic scale" (§5.8) | stands | strengthens considerably if both conditions agree |
| Any statement of the form "at both pileup conditions…" | **cannot be written yet** | required |

The safest construction is to argue from pu200 throughout and add pu0 as corroboration, rather
than building an argument that needs both legs to stand.

---

## 2. §5.1 Principal findings

Three short paragraphs. No new numbers beyond the headline; no mechanism yet.

1. **At cluster level the two methods are close on efficiency and differ on purity**, and the
   comparison inverts with energy — CLUE ahead above 5 GeV, MaskFormer ahead on purity throughout.
2. **At jet level, the observable used in analysis, the ordering is clear**: MaskFormer's jet
   energy scale is 1.15 against CLUE's 1.62, with 6.3 fake jets per event against 11.7.
3. **The two methods fail in different currencies**: MaskFormer loses energy it never claims
   (0.271 of the deposit); CLUE loses it to neighbouring clusters (0.228) and absorbs unassociated
   energy in exchange.

State plainly here that the headline research question — does a learned set-prediction model
outperform a tuned classical algorithm — **does not have a single-valued answer** on this
evidence. It depends on the metric, and the honest result is that the metric choice is itself a
finding.

---

## 3. §5.2 How each method fails

### MaskFormer: coverage that collapses as showers grow
**[M]** The fraction of a shower's cells correctly assigned falls 0.48 → 0.11 between 0.5 GeV and
20-50 GeV, while truth cluster size itself grows 2.7× (14.9 → 40.0 cells). Below ~3 GeV MaskFormer
is *ahead* of CLUE; the curves cross there and MaskFormer then falls about twice as steeply.

⚠️ **Do not phrase this as "MaskFormer's clusters shrink while CLUE's grow."** That was an earlier
reading taken from a mean over a bimodal distribution and it does not survive: on the median,
CLUE's recovered cells fall too. The defensible statement is about *coverage fractions* and their
*relative slopes*, which is what the figure now plots.
**[M]** Transversally it loses the halo: recovery falls from 0.68 at the shower axis to 0.15 at
ΔR = 0.275, and sits below CLUE beyond ΔR ≈ 0.1.
**[M]** 27.1% of target energy is claimed by no cluster at all — its single largest loss channel.
**[I]** These are three views of one behaviour: the head emits a compact object of roughly fixed
extent, so as showers grow it covers a shrinking fraction of them.

### CLUE: fragmentation, with occasional runaway growth
**[M]** CLUE's cluster-size distribution at high energy is **bimodal**: at 50-200 GeV the matched
cluster holds a median of 2 cells but a mean of 122.9, with a 90th percentile of 449.6. It usually
shatters a large shower into small fragments -- individually rather pure, which is how its purity
holds up -- and occasionally produces one cluster that swallows a whole region.
**[I]** That bimodality is the natural reading of the resolution result: sometimes a fragment,
sometimes everything, which is precisely what a σ_E/E of 6.37 looks like. The two measurements
support each other.
**[M]** 2,495 clusters per event against 1,499 targets (+66%); 17.5 jets per event against 6.1.
**[M]** 22.8% of target energy ends up in some other particle's cluster.
**[I]** Density-peak growth has no stopping criterion tied to the object: it walks outward until
density falls off, so in a dense environment it absorbs whatever is adjacent — neighbouring
showers and, at pileup 200, unassociated energy.

### One failure of CLUE is configurational, not algorithmic
**[M/I]** Sub-detectors are clustered independently, so a hadron showering from ECAL into HCAL
yields **at least two clusters by construction**. This is a property of the pipeline as configured
(Section 3.5), not of CLUE, and it inflates the split rate. Say so — it is a limitation of your
baseline that you should name yourself.

---

## 4. §5.3 Which failure matters for physics

This is the strongest argument in the thesis. Build it in four steps.

**Step 1 — the two failures have different statistical character. [I]**
MaskFormer's response falls smoothly 1.11 → 0.56 with σ_E/E confined to 0.61–1.01 across three
decades. CLUE's response wanders (1.24 → 1.04 → 1.32 → 0.58) with σ_E/E rising to 6.37. One is a
**bias**; the other is **variance**.

**Step 2 — they are not equally correctable. [I]**
A smooth, monotonic response is a calibration curve: in the language of Section 2.1 it contributes
to the constant term *c* and can be corrected downstream. A factor-7.7 spread cannot be corrected
by any per-jet scale factor; it enters the noise term *b* and propagates.

**Step 3 — particle flow converts each into a different error. [I]**
From your Section 2.2: after track subtraction, a cluster that absorbs a neighbour's energy leaves
too little behind and **deletes a real neutral**; one that is too small leaves too much and
**creates a neutral that never existed**. CLUE's 11.7 fake jets per event against 6.1 real ones is
the observable form of the first; MaskFormer's 27.1% unclaimed energy is the second.

**Step 4 — the jet result is where it is settled. [M]**
Jet energy scale 1.15 vs 1.62; fake jets 6.3 vs 11.7. Both find ~92–95% of reference jets, so the
difference is entirely in *what the jets contain*.

**Then the honest qualification. [M]** At jet level CLUE's cluster-level variance largely averages
out — jet σ_pT/pT is comparable below 100 GeV and if anything slightly better for CLUE. A jet sums
many clusters, so over- and under-assignments partially cancel. **The cluster-level σ_E/E = 6.37
does not propagate to jets.** Say this; omitting it would overstate the case.

---

## 5. §5.4 Why MaskFormer's clusters do not scale

Four candidate explanations, in descending order of how well the evidence supports them. Be
explicit about which is which — this section is where an examiner will probe hardest.

### (a) The mask head has no representation of object scale. **[I, strong]**
The mask logit is a dot product between one query vector and one cell vector, scored
independently per cell. Nothing in the head parameterises *extent*: whether a query claims 6 cells
or 60 is an emergent consequence of where its logits happen to cross a threshold. CLUE, by
contrast, grows outward until density falls off, so scale is intrinsic to the algorithm.
**Supporting measurement:** predicted size can be *anti-correlated* with true size (13.2 → 9.6
against truth 14.9 → 40.0) — nothing in the formulation forbids it.

### (b) Masked attention plausibly reinforces compactness. **[H, untested — label clearly]**
The configuration used is `mask_attention: true` with `mask_attention_threshold: null`, which
inherits `pred_threshold = 0.5`. Each query's cross-attention at layer *L+1* is therefore
restricted to cells its mask exceeded 0.5 at layer *L*. A shower's halo starts below 0.5, so the
query never attends to it, so it receives no gradient to claim it — a self-reinforcing tunnel.
`unmask_all_false: true` rescues only queries whose mask is *entirely* empty; it does nothing for
one claiming 10 cells of a 40-cell shower.

**Consistent with:** compact converged clusters; core captured (0.68 recovery at the axis) and
halo lost (0.15); the effect worsening with energy as halos grow; and the flat F₁ plateau, since
the representation was built under restricted attention and no threshold can reorder it.

**How it could be tested:** by letting the attention mask and the prediction mask use different
thresholds, so the query attends more permissively than it commits. If the account above is right,
cluster extent should begin to track shower size; if extent is unchanged, the explanation is wrong
and attention masking is not the constraint. Either outcome is informative, which is what makes it
worth doing. Present it as the natural next experiment rather than as a fix.

*Note for context, not defence:* Mask2Former uses 0.5 because in image segmentation objects
occupy a large fraction of the image and masks converge outward. Here a shower is ~0.05% of the
event.

### (c) DICE is normalised by object size. **[I, moderate]**
Covering a 6-cell fragment perfectly scores as well as covering a 40-cell shower perfectly, so
nothing in the objective prefers one correct large cluster to several correct small ones.
**Counter-evidence to state:** the sweep found dice-dominant and focal-dominant objectives
statistically indistinguishable across an 80× ratio change, so the *weighting* is not the lever
even if the normalisation is a real property.

### (d) The failure is one of ranking, not thresholding. **[M, established]**
The working-point scan found F₁ flat at 0.338–0.359 across the entire grid, with the diagnostic
reporting that the object head accepts 92.9% of true particles and masks alone reproduce the
result. A flat plateau with a correct object head means the cells are **ordered** wrongly, and no
cut point can reorder a ranking. This bounds what any post-hoc working point can achieve and
justifies not pursuing it further.

---

## 6. §5.5 What the interventions tell us

Frame these as **bounds**, not failures. A controlled negative result constrains the space.

**The objective sweep. [M]** Focal's *presence* determines whether the mask head trains at all;
its weight does not. Two arms containing focal are indistinguishable across an 80× ratio change;
both arms without it collapse, by two distinct mechanisms — BCE converges to predicting the target
prior everywhere, dice alone saturates the sigmoid at zero on a flat gradient region.
**[I]** With AdamW the absolute loss scale is near-irrelevant (Adam normalises by the second
moment), so only the ratio carries information — and that has now been sampled across 80×. Further
weight tuning is therefore not a promising direction.

**The mask-head variants. [M + I]** Only the coverage arm (v1) separated from the control at 95%
(slope 1.844 [1.684, 2.022] vs 1.648 [1.481, 1.842]); propagation and recall did not.
**State the scope limitation as the main conclusion:** all four arms ran for 4,000 steps, where the
control *over*-predicts cluster size (~2.5× truth), whereas the converged 30,000-step model
*under*-predicts. Propagation spreads confidence outward — it pushes in the direction the model was
already erring at 4,000 steps and in the needed direction at convergence. **[I]** The variants were
therefore evaluated in the wrong regime, and the negative result for v3 in particular should not be
read as a refutation of the mechanism. That is an honest and useful thing to report.

**Chaining. [I]** A purely geometric post-process recovers +0.092 efficiency and cuts unclaimed
energy 0.271 → 0.092. Read as *evidence*: the masks are fragmented in a geometrically predictable
way. Read as a *method* it is weaker — purity falls 0.356 → 0.266 and the median response at
0.5–1 GeV rises to 1.81, so it buys the headline by over-merging soft particles. Note also that it
makes the comparison "learned model + classical post-process vs classical algorithm", which is a
different question from the one posed.

**Overlap representation. [M]** MaskFormer claims ≥2 queries on 55% of genuinely multi-owner cells
against 7.7% of single-owner cells, so the non-exclusive output your Section 2.4 argues for is
genuinely used, and used discriminatingly. **But [M]:** only 0.78% of cells in the pu200 barrel
sample have two or more *target* contributors, so the capability has little to act on here. The
architectural claim is validated; its practical value on this sample is small. Report both halves.

---

## 7. §5.6 Threats to validity

Own all of these explicitly. This section is worth more marks than it costs space.

**1. Training length. [M] — state it accurately, neither overclaimed nor overstated.**

The facts, and they support a measured statement rather than an alarming one:

| optimiser steps | efficiency | purity | F₁ | Δ F₁ per 10k steps |
|---|---|---|---|---|
| 4,000 | 0.676 | 0.202 | 0.312 | — |
| 12,000 | 0.635 | 0.317 | 0.423 | +0.139 |
| 30,000 | 0.663 | 0.432 | 0.523 | **+0.056** |

`val_loss` fell 70.93 → 57.25 over the five epochs, with the last three checkpoints at
57.32 → 57.27 → 57.25.

**The accurate description is that the metrics are flattening, not that training was cut short.**
The rate of improvement more than halved between the two intervals, and the final validation-loss
steps are of order 0.05 in 57 — a curve approaching its asymptote rather than one still climbing
steeply. The model saw 30,000 optimiser steps over 8 h 40 m of dedicated A100 time on a shared
card; that is a real training run, not a truncated one.

What can honestly be said: **these are conservative estimates of what the architecture achieves,
and the remaining headroom appears modest rather than transformative.** Word it that way. Avoid
"undertrained" as a bare adjective — it invites the reading that the comparison was not seriously
attempted.

**Worth one sentence of context, without excuse-making:** published transformer reconstruction
results are typically trained for substantially longer on larger or dedicated GPU allocations, so
the absolute performance here should be read as what the architecture reaches under a
single-card, single-student compute budget. The *comparison* to CLUE remains sound, because both
methods were developed under the same constraint and CLUE was, if anything, the better-tuned of
the two (see threat 2).

**2. Tuning asymmetry, and it runs against the model. [M]**
CLUE received a systematic 80-trial Optuna search per sub-detector with a range-widening protocol;
MaskFormer received a three-arm objective sweep. The CLUE numbers are therefore near CLUE's
ceiling while MaskFormer's are not, making any MaskFormer advantage **conservative** and the
high-energy deficit the honest headline.

**3. The two pileup conditions are not cross-comparable. [M]**
pu200 restricts to |η| < 0.88; pu0 covers |η| < 4.0. Each is internally controlled; no
cross-condition statement is supported.

**4. Single seed.** No estimate of run-to-run variance. State it.

**5. The exclusive-partition approximation. [M]**
Scoring uses a hard partition, discarding the fractional truth. It keeps ~83% of (particle, cell)
associations carrying ~94% of each particle's energy — and at pu200 barrel only 0.78% of cells are
multi-owner, so the approximation is cheap here. But it does structurally disadvantage the method
whose distinguishing feature is non-exclusive output.

**6. Truncation.** 6 of 500 pu200 events and 18 of 500 pu0 events exceeded the query budget.

**7. The jet reference excludes pileup by construction. [M]**
Both methods over-measure jet pT for that reason. Only the *difference* between methods is
interpretable; neither number's offset from unity is.

**8. Reference clusterings are not upper bounds. [M]**
`oracle_geometric` is one specific rule (nearest shower axis given perfect seeds) and CLUE beats it
above 20 GeV; `oracle_resolution` bounds efficiency but not purity. They were removed from the
figures for this reason. The one defensible statement: `oracle_resolution` reaches 0.989
efficiency, so **the cells carry the information and the shortfall is algorithmic, not intrinsic**.

**9. Part of the difficulty is the sample, not the methods. [M]**
`oracle_geometric` reaches only 0.265 above 20 GeV — below CLUE's 0.304. A perfect nearest-axis
clusterer also struggles at high energy in the barrel sample.

---

## 8. §5.7 Relation to prior work

- **Tracking MaskFormers** (`stroudTransformersChargedParticle2024`) reach 97% track efficiency at
  0.6% fake rate. **Do not treat this as a target missed.** Different subsystem, different object
  definition, different dataset. What it establishes is that the architecture operates at HL-LHC
  scale; the calorimeter case replaces trajectories with showers, and a shower has spatial extent
  that varies by nearly 3× over the energy range while a track does not. **[I]** The scale problem
  identified in §5.4 is specific to objects whose size varies — which is a genuine contribution to
  make.
- **Object condensation** (`kieselerObjectCondensationOnestage2020`,
  `qasimLearningRepresentationsIrregular2019`) reports gains over particle-flow baselines at
  moderate multiplicity. Note the differences honestly: those results are on events with at most a
  few tens of particles against ~1,500 here, and against author-implemented baselines rather than a
  systematically tuned one.
- **GLOW** (`kobylianskiiGLOWUnifiedParticle2025`) predicts the incidence matrix directly. Relevant
  because this work *removed* its incidence head, and §5.9 proposes reinstating it in a form that
  feeds the prediction path.
- **CLUE/TICL** (`rovereCLUEFastParallel2020`, `pantaleoIterativeClusteringFramework2023`) — note
  that this study's per-sub-detector configuration is a simplification of TICL and is responsible
  for part of the measured split rate.

---

## 9. §5.8 What a good calorimeter clusterer would do

The synthesis, and the thesis's own contribution. Four properties, each grounded in a measurement:

1. **Intrinsic, learned object scale.** Cluster extent must grow with shower energy. Truth grows
   2.7×; MaskFormer's predicted extent shrinks, CLUE's grows 9× without bound. Neither has a
   *calibrated* notion of how large this shower should be.
2. **Depth reach with transverse restraint.** MaskFormer holds its longitudinal profile better
   beyond ~18 layers (0.56 vs 0.50 at 30); CLUE holds the transverse halo better beyond ΔR ≈ 0.1
   (0.22 vs 0.15). Neither does both.
3. **Contamination control in a pileup environment.** The jet result is the evidence: 1.15 vs 1.62
   energy scale, 6.3 vs 11.7 fakes. Whatever else it does, a clusterer at pileup 200 must decline
   energy that is not its object's.
4. **The ability to represent a shared cell** — validated as used (55% of multi-owner cells claimed
   twice) but of limited value at 0.78% occupancy. State it as a property worth having, on a
   sample that rewards it more.

Then the honest closing observation: **above ~10 GeV both methods and the geometric reference all
degrade together**, which suggests a shared limit not addressed by either approach.

---

## 10. §5.9 Future work

Write these as **directions worth exploring**, not as a repair list. Each is framed around the
question it would answer, because the value is in what would be learned rather than in a
particular setting turning out to be right. Roughly in order of how directly each addresses the
measured behaviour.

**Decoupling attention masking from prediction.** The masked-attention mechanism restricts a
query's cross-attention to the cells its own mask already claims, which raises the question of
whether a query can ever discover the parts of a shower it did not initially find. It would be
interesting to see how the model behaves when the attention mask and the prediction mask are
allowed to differ — attending more permissively than it predicts — and whether cluster extent then
tracks shower size more closely. This is the most direct probe of the mechanism proposed in
§5.4(b), and its outcome would either support that account or rule it out.

**Learning object scale explicitly.** The measured behaviour suggests the head has no
representation of how large its object should be. A natural line of enquiry is whether predicting
an explicit per-query extent — a radius, a covariance, or a predicted cell count — alongside the
mask gives the model a handle on scale that thresholded logits cannot provide. This is the deepest
of the directions here and the one most likely to generalise beyond calorimetry, since any
set-prediction task with objects of widely varying size faces the same issue.

**Revisiting the relational variants at convergence.** The propagation arm was assessed while the
baseline over-claimed; at convergence the baseline under-claims, so the same modification acts in
the opposite direction relative to the error. Re-examining it in that regime would establish
whether the mechanism has value or whether the earlier null result was regime-dependent — a
question the present evidence genuinely cannot settle.

**Predicting the fractional truth.** The task's truth is an incidence matrix and the model predicts
binary masks, so the supervision discards the energy fractions entirely. Following the direction
taken by GLOW, it would be worth investigating whether supervising against the fractions changes
how the model treats a shower's low-energy periphery, where an energy-weighted target penalises an
omission in proportion to what is omitted.

**Reducing the candidate set before clustering.** The working-point scan indicates that the
ordering of cells, rather than the choice of cut, limits performance. A learned hit filter applied
before the clustering stage would shrink the set over which that ordering has to be correct, and
it would be informative to know how much of the ranking problem is a consequence of the sheer
number of candidates at pileup 200.

**Cardinality pressure on the query budget.** Both methods produce more clusters than there are
particles, and in the present configuration nothing in the matching cost discourages this. Whether
an explicit penalty on surplus objects reduces the fake rate without costing efficiency is an open
and easily posed question.

**Longer training on larger allocations.** The metric trend in §5.6 suggests modest remaining
headroom, but that is an extrapolation from three points. Establishing where the curve actually
flattens would put a number on it, and would make the comparison with published results — trained
under quite different compute budgets — more directly meaningful.

---

## 11. A note on tone

The most damaging thing you can do in this chapter is present MaskFormer's high-energy deficit as
a disappointment. It is the central finding, it was diagnosed three independent ways — cluster
size, shower profiles, and morphology — and it has a stated mechanism with a proposed test. A
thesis that finds a classical algorithm competitive with a transformer, and explains precisely
why, is a stronger piece of work than one that reports a win. Write it as the result.
