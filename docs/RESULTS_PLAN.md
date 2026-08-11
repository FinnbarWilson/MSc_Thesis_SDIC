# Results chapter — writing plan

Everything needed to write Section 4. All numbers read from `results/pu200/*.parquet`,
`results/pu200/jets.parquet` and `results/pu200/anatomy_particles.parquet` on 2026-08-10, on the
**500-event evaluation window [7500, 8000)**. That last file is a rebuildable cache and is
gitignored rather than tracked (44 MB); `make_thesis_figures.py --rebuild-anatomy` regenerates it
from the event store in a few minutes. Figures live in `figures/thesis/` as `.pdf` and
`.png`; all are 5.2 in wide (0.8 textwidth at 12pt A4, 1 in margins).

**pu0 is not yet available.** Its training run is in progress; every pu0 cell below is marked
**[pu0 TBD]** and §9 lists the commands that fill them.

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

## 0. The results/discussion split — read before writing a word

The single most common way this chapter goes wrong is smuggling mechanism into it. Use this rule:

| Belongs in **Results** | Belongs in **Discussion** |
|---|---|
| "MaskFormer recovers 5.1 cells for a 40-cell shower" | "because the dot-product mask head encodes no notion of object scale" |
| "CLUE's σ_E/E rises to 6.37" | "which is variance rather than bias, and is not correctable" |
| "chaining raises efficiency 0.509 → 0.601 and lowers purity 0.356 → 0.266" | "so it buys the headline by over-merging" |
| "the jet energy scale is 1.15 (MaskFormer) against 1.62 (CLUE)" | "under particle-flow subtraction this creates fake neutrals" |
| "F₁ rose 0.312 → 0.423 → 0.523 at 4k/12k/30k steps" | "so the model is undertrained and these are lower bounds" |

**Results states what was measured, with the conditions under which it was measured.**
Comparative statements are fine ("A is higher than B"); causal and evaluative ones are not
("A is better", "because", "this means"). Where a result is *surprising*, say so plainly and
leave the explanation to the discussion — a forward reference ("Section 5.3 returns to this") is
the right tool.

---

## 1. Recommended structure

```
4.1  Overview and reading guide       what is reported, on what sample, at what working point
4.2  Cluster-level performance        the headline comparison           fig_eff_purity
4.3  The energy budget                where a particle's energy ends up fig_energy_budget
4.4  Cluster scale and shower profiles the anatomy of the difference    fig_cluster_size, fig_shower_profile
4.5  Energy response and resolution   bias and variance                 fig_response
4.6  Jet-level performance            does it survive into an observable fig_jets
4.7  Attempts to close the gap        post-processing and objective arms  (table only)
4.8  Summary of findings              a bulleted list of what was measured, no interpretation
```

Rationale for this order: aggregate → decomposition → mechanism-revealing measurements → physics
observable. Each section supplies the quantity the next one explains. §4.7 sits last among the
measurements because it is about interventions rather than about the two methods.

---

## 2. §4.1 Overview and reading guide

State once, so it need not be repeated in every caption:

- All numbers on the **500-event evaluation window**, disjoint from training and from the CLUE
  tuning window.
- Working point **0.5** throughout unless stated: a particle is *reconstructed* if the cluster
  matched to it holds ≥50% of its deposited (calibrated) energy; a cluster is *pure* if ≥50% of
  its energy belongs to its matched particle.
- **Energy-weighted** metrics are primary.
- MaskFormer operating point: mask threshold 0.5, object threshold 0.2.
- **pu0 and pu200 are not cross-comparable** (different η coverage) — repeat the §3.1 warning here
  in one sentence, because this is where a reader will be tempted to compare columns.
- Reference clusterings (`oracle_geometric`, `oracle_resolution`) appear in tables only, not in
  figures. Say why in one sentence: neither bounds both metrics, so neither is a ceiling in the
  ordinary sense (details in §3.4 or a footnote).

---

## 3. §4.2 Cluster-level performance → `fig_eff_purity.pdf`

**Figure.** Two rows (efficiency, purity) × columns per pileup condition. x = true particle
energy, log scale, 0.5–200 GeV. Blue = MaskFormer, green = CLUE. 5.2 × 4.4 in.

### Headline table (pu200, 500 events)

| method | efficiency | purity | F₁ | eff > 5 GeV | eff > 20 GeV | clusters/event |
|---|---|---|---|---|---|---|
| MaskFormer | 0.509 | **0.356** | **0.419** | 0.239 | 0.115 | 1,918 |
| CLUE | **0.524** | 0.267 | 0.354 | **0.377** | **0.304** | 2,495 |
| MaskFormer + chaining | **0.601** | 0.266 | 0.369 | 0.302 | 0.158 | 1,918 |
| *oracle_geometric* | 0.798 | 0.247 | 0.377 | 0.436 | 0.265 | 1,499 |
| *oracle_resolution* | 0.989 | 0.424 | 0.594 | 0.987 | 0.993 | 1,483 |

Reference: **1,499 target particles/event**.

### What to state
1. **Overall efficiency is close**: 0.509 vs 0.524, a 0.015 gap.
2. **Purity differs**: 0.356 vs 0.267, MaskFormer +0.089. F₁ 0.419 vs 0.354.
3. **The gap opens with energy.** Above 5 GeV CLUE leads 0.377 to 0.239; above 20 GeV, 0.304 to
   0.115 — a factor of 2.6.
4. **From the figure:** the curves are level below ~2 GeV (both ≈0.55 at 1.4 GeV) and separate
   monotonically above it. Purity is level across the whole range (0.43–0.64 both methods) with
   MaskFormer marginally ahead in the middle and CLUE ahead in the top bin.
5. **Cluster multiplicity**: 1,918 (MaskFormer) and 2,495 (CLUE) against 1,499 true — both
   over-produce, CLUE by 66%, MaskFormer by 28%.
6. `oracle_geometric` reaches only 0.265 above 20 GeV, *below* CLUE's 0.304. State it; do not
   explain it here.

**Do not write here:** why the gap opens with energy; whether purity or efficiency "matters more".

**[pu0 GAP]** Same table, same figure column. Write the prose now with placeholders; the
sentences that will need numbers are: the overall efficiency comparison, the purity comparison,
the >5 and >20 GeV rows, and the cluster multiplicity against 538 targets/event. Expect — but do
not state — MaskFormer ahead overall and behind above 20 GeV.

---

## 4. §4.3 The energy budget → `fig_energy_budget.pdf`

**Figure.** Horizontal stacked bars, one per method, three segments, values printed inside.
Lightness encodes fate (darkest = recovered). 5.2 × 2.65 in.

### Table — fractions of total deposited target energy (they sum to 1 by construction)

| method | recovered | taken by another cluster | never claimed |
|---|---|---|---|
| MaskFormer | 0.533 | 0.195 | **0.271** |
| CLUE | **0.589** | 0.228 | 0.183 |
| MaskFormer + chaining | 0.613 | 0.295 | 0.092 |

### What to state
- Definition: *recovered* = `Σ(eff_e × e_dep) / Σ e_dep` (the part of the particle's own deposit
  held by its matched cluster); *taken by another cluster* = energy in some other cluster;
  *never claimed* = energy in no cluster. Exhaustive and mutually exclusive.
- CLUE recovers 5.6 points more energy than MaskFormer (0.589 vs 0.533).
- **The methods differ mainly in how they lose the rest**: MaskFormer's largest loss is energy it
  never claims (0.271); CLUE's is energy taken by another cluster (0.228).
- Chaining converts unclaimed energy into both recovered *and* misassigned energy: unclaimed falls
  0.271 → 0.092 while misassignment rises 0.195 → 0.295.

**Do not write here:** that unclaimed energy is "systematic" or misassignment is "variance".

**[pu0 GAP]** One extra bar group. The interesting question this figure answers for pu0 is whether
MaskFormer's largest loss channel is still *unclaimed* energy at an occupancy 25× lower — if it is,
the behaviour is a property of the method rather than of pileup, which would be a stronger claim
than pu200 alone supports. Leave a sentence for it.

---

## 5. §4.4 Cluster scale and shower profiles

Two figures, one section. This is the most original material in the chapter.

### 5a. `fig_cluster_size.pdf`
One row, columns per condition. x = true particle energy (log). y = **fraction of the shower's
cells correctly assigned** — the total cells correctly assigned in an energy bin divided by the
total truth cells in it. Bands are a bootstrap over events. 5.2 × 2.65 in.

| E [GeV] | 0.5–1 | 1–2 | 2–5 | 5–10 | 10–20 | 20–50 | 50–200 |
|---|---|---|---|---|---|---|---|
| MaskFormer | 0.480 | 0.519 | 0.375 | 0.235 | 0.157 | 0.109 | **0.127** |
| CLUE | 0.402 | 0.453 | 0.368 | 0.349 | 0.325 | 0.299 | **0.277** |

**What to state**
- A flat line would mean cluster size growing in step with shower size. **Neither method is flat.**
- Below ~3 GeV MaskFormer is ahead (0.48–0.52 against 0.40–0.45). The curves **cross at ~3 GeV**.
- Above the crossing MaskFormer falls about twice as steeply, reaching 0.11 at 20–50 GeV against
  CLUE's 0.30. Quote the crossing point: it locates where the learned model stops competing.
- For context, truth cluster size itself grows 2.7× across this range (mean 14.9 → 40.0 cells),
  so a constant fraction would already require the clusters to grow.

⚠️ **THIS FIGURE PLOTS A RATIO OF SUMS, AND IT MUST.** An earlier version plotted the *mean
number of cells recovered per particle* and gave a **materially different and wrong** answer.

The per-particle distributions are bimodal, badly so for CLUE. At 50–200 GeV its matched cluster
recovers a **mean of 11.1 cells but a median of 1** — it usually fragments a large shower and
occasionally captures most of one. On the mean, CLUE's recovered cells *rise* with energy
(6.0 → 11.1); on the median they *fall* (3 → 1). The two statistics disagree on the **sign of the
trend**, and neither describes the underlying behaviour, because there are two behaviours.

The same skew affects predicted cluster size: at 50–200 GeV CLUE's matched cluster is a **mean of
122.9 cells, median 2.0, 90th percentile 449.6**. Do not quote the mean. If a statement about
CLUE over-claiming is wanted, use the **jet energy scale (1.62)** instead — that number is
tightly bounded and makes the point better.

Summing before dividing removes the problem: "what fraction of all truth cells in this bin were
correctly assigned" is well posed whatever the per-particle distribution looks like.

**Reference numbers if a distributional statement is needed** (matched cluster size in cells):

| 50–200 GeV | mean | median | 90th pct |
|---|---|---|---|
| truth | 40.0 | 35.0 | — |
| MaskFormer | 9.6 | 7.0 | 23.6 |
| CLUE | 122.9 | 2.0 | 449.6 |

**[pu0 GAP]** One extra column. The pu0 comparison matters most here: if pu0 also shows the
crossover, the behaviour is a property of the formulation; if pu0 is flat, occupancy makes it
worse. Those are different claims — write the paragraph so either can be dropped in.

### 5b. `fig_shower_profile.pdf`
Two rows (transverse, longitudinal) × columns per condition, `sharex=False` because the rows are
different coordinates. 5.2 × 4.4 in.

- **Shower frame:** for each target particle, `ΔR` is the angular distance from that particle's
  energy-weighted axis in (η, φ); `depth` is the calibrated layer index minus the particle's own
  first lit layer, so showers starting at different depths are compared at the same stage.
- y = fraction of the energy in that bin recovered by the matched cluster.

**Transverse (ΔR), read from the figure:**

| ΔR | ~0.003 | 0.0075 | 0.015 | 0.025 | 0.04 | 0.065 | 0.10 | 0.16 | 0.275 |
|---|---|---|---|---|---|---|---|---|---|
| MaskFormer | 0.68 | 0.62 | 0.56 | 0.49 | 0.42 | 0.33 | 0.25 | 0.19 | 0.15 |
| CLUE | 0.76 | 0.71 | 0.62 | 0.48 | 0.39 | 0.33 | 0.31 | 0.28 | 0.22 |

**Longitudinal (layers past shower start), read from the figure:**

| layers | 0.5 | 1.5 | 2.5 | 3.5 | 5 | 7 | 10 | 14 | 20 | 30 | 42 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MaskFormer | 0.43 | 0.50 | 0.55 | 0.58 | 0.59 | 0.58 | 0.55 | 0.53 | 0.55 | 0.56 | 0.49 |
| CLUE | 0.56 | 0.61 | 0.64 | 0.66 | 0.66 | 0.65 | 0.62 | 0.57 | 0.53 | 0.50 | 0.45 |

**What to state**
- Transversally both fall steeply; they **cross twice** — CLUE ahead in the core (ΔR < 0.02),
  level around 0.025–0.065, CLUE ahead again in the halo (ΔR > 0.1, 0.22 vs 0.15 in the last bin).
- Longitudinally CLUE leads for the first ~15 layers and **MaskFormer leads beyond ~18 layers**
  (0.56 vs 0.50 at 30 layers).
- MaskFormer's profile rises over the first ~4 layers (0.43 → 0.58) before flattening.
- These are pooled over all energies. *(If you want the energy-split version, the earlier
  three-band analysis showed the longitudinal ordering reverses above 10 GeV. Re-run it before
  quoting; it is not in the current figure.)*

**[pu0 GAP]** Both panels gain a column. The pu0 comparison matters here more than anywhere else:
pu0's own earlier diagnostic showed the matched cluster holding ~6 cells whether the particle
deposited 13 or 38, i.e. a *flat* profile, whereas pu200 shows a *falling* one (7.1 → 5.1). If pu0
under the new objective also falls, the two conditions agree and the pathology is a property of the
formulation; if pu0 is flat and pu200 falls, occupancy makes it worse. Those are different claims —
write the paragraph so either can be dropped in.

---

## 6. §4.5 Energy response and resolution → `fig_response.pdf`

**Figure.** Two rows: median `E_reco/E_true` (linear y, 0.4–2.0, dotted line at unity) and
`σ_E/E` (log y, ticks 0.5/1/2/5). σ from IQR/1.349, normalised by the median. 5.2 × 4.4 in.

| E [GeV] | 0.5–1 | 1–2 | 2–5 | 5–10 | 10–20 | 20–50 | 50–200 |
|---|---|---|---|---|---|---|---|
| MaskFormer, median response | 1.11 | 1.04 | 0.92 | 0.78 | 0.70 | 0.59 | **0.56** |
| CLUE, median response | 1.24 | 1.04 | 1.06 | 1.25 | 1.32 | 0.86 | **0.58** |
| MaskFormer, σ_E/E | 0.68 | 0.61 | 0.67 | 0.75 | 0.83 | 1.01 | **0.82** |
| CLUE, σ_E/E | 0.83 | 0.75 | 0.89 | 1.21 | 1.81 | 3.88 | **6.37** |

**What to state**
- MaskFormer's response falls monotonically 1.11 → 0.56, crossing unity between 2 and 5 GeV.
- CLUE's response is non-monotonic: 1.24, dipping to 1.04, rising to a maximum of 1.32 at
  10–20 GeV, then falling to 0.58.
- **σ_E/E is the largest single difference in the chapter.** MaskFormer stays within 0.61–1.01
  across three decades; CLUE rises from 0.83 to **6.37**, a factor of 7.7.
- Both converge to a similar median response in the top bin (0.56 vs 0.58) with very different
  spreads.

**Do not write here:** "bias versus variance", or that a stable bias is calibratable. That is the
core of the discussion.

**[pu0 GAP]** Two extra panels. Watch specifically whether CLUE's σ_E/E still blows up at high
energy without pileup — at pu200 it reaches 6.37, and if much of that comes from absorbing pileup
then pu0 should be far tamer. That single comparison would separate "CLUE over-merges" from "CLUE
absorbs pileup", which the pu200 data alone cannot.

---

## 7. §4.6 Jet-level performance → `fig_jets.pdf`

**Figure.** Three rows — jet efficiency, median `pT_reco/pT_ref`, `σ_pT/pT` — columns per
condition. x = reference jet pT, log, ticks 25/35/50/75/100/150/200. 5.2 × 5.25 in.

**Setup to restate briefly** (full detail in §3.4.2): anti-k_t, R = 0.4, jets above 25 GeV,
matched within ΔR < 0.3. Reference jets are built from the **truth partition** — the jets a
perfect clusterer would produce from these cells.

### Integrated (pu200, 499 events with ≥1 reference jet)

| | reference | MaskFormer | CLUE |
|---|---|---|---|
| jets / event | **6.11** | 11.95 | 17.51 |
| fake jets / event | — | **6.32** | **11.69** |
| jet efficiency | — | 0.921 | **0.954** |
| median pT_reco/pT_ref | — | **1.153** | **1.615** |

### Jet energy scale by reference pT

| ref pT [GeV] | 25–35 | 35–50 | 50–75 | 75–110 | 110–160 | 160–250 |
|---|---|---|---|---|---|---|
| MaskFormer | 1.22 | 1.13 | 1.04 | 0.96 | 0.87 | 0.84 |
| CLUE | 1.67 | 1.61 | 1.54 | 1.46 | 1.39 | 1.37 |

**What to state**
- Both methods find nearly all reference jets: efficiency 0.87–0.93 in the lowest bin, ≈1.0 above
  40 GeV. Integrated 0.921 (MaskFormer) and 0.954 (CLUE).
- **Jet energy scale differs by ~0.46**: 1.153 vs 1.615 integrated; the gap is 0.45 at 25–35 GeV
  and 0.53 at 160–250 GeV. MaskFormer crosses unity near 75 GeV; CLUE never approaches it.
- **Fake jets**: 6.32 (MaskFormer) and 11.69 (CLUE) per event against 6.11 real ones. CLUE
  produces 2.9× the reference jet count.
- **Resolution** (bottom panel): 0.12 at threshold for both, falling to ~0.04–0.06 by 200 GeV,
  CLUE generally lower. **State conservatively** — above ~110 GeV the bins are sparse and the
  MaskFormer curve is non-monotonic (dips to 0.056 at 130 GeV, rises to 0.102 at 200 GeV). Say
  the resolutions are comparable below ~100 GeV and that the statistics above that do not support
  a statement.

**Mandatory caveat, in the caption:** both methods over-measure because the reference contains
only target-particle cells and excludes pileup by construction. The meaningful quantity is the
**difference between the methods**, not either one's offset from unity.

**[pu0 GAP]** Three extra panels, and this is the most diagnostic gap in the chapter. At pu200 both
methods over-measure jet pT (1.15 and 1.62). Without pileup there is far less unassociated energy
to absorb, so if the over-measurement largely disappears at pu0 it was pileup contamination; if it
persists, it is neighbour-stealing. State whichever the data shows; do not guess.

---

## 8. §4.7 Attempts to close the gap — **table, not a figure**

Seventeen arms do not plot legibly. One table, two blocks.

### 8a. Mask-objective sweep (pu200 barrel, 4,000 steps each, matched control, 10 test events)

| objective | max mask prob | eff@0.5 | eff@0.75 | pur@0.5 | cells/flow (truth 15) |
|---|---|---|---|---|---|
| dice 20 + focal 1 *(adopted)* | 1.0000 | 0.664 | 0.419 | 0.118 | 31.7 |
| dice 5 + focal 20 | 1.0000 | 0.665 | 0.411 | 0.118 | 32.8 |
| dice only | 0.0000 | — | — | — | collapsed |
| dice 1 + bce 1 | 0.0995 | — | — | — | collapsed |

State: the two arms containing focal are statistically indistinguishable despite an 80× difference
in the dice:focal ratio; both arms without focal collapse to a degenerate solution in which no cell
anywhere exceeds the 0.5 threshold.

### 8b. Post-processing

| variant | efficiency | purity | never-claimed energy | median response @ 0.5–1 GeV |
|---|---|---|---|---|
| MaskFormer | 0.509 | 0.356 | 0.271 | 1.11 |
| + chaining | 0.601 | 0.266 | 0.092 | 1.81 |

### 8c. Mask-head variants

Three single-switch modifications to the mask head, each trained for 4,000 steps against a control
identical in every other respect, on the pu200 barrel configuration. **What each does:**

- **v1, coverage** (`coverage_weight = 2.0`, energy-weighted): adds a per-target penalty on the
  fraction of that particle's energy its matched query failed to claim. Motivated by DICE being
  size-normalised, so covering a small fragment perfectly scores as well as covering a large
  shower perfectly.
- **v2, recall** (`bce_pos_weight = 20`): replaces the focal term with a binary cross-entropy
  carrying an explicit positive-class weight, testing whether an explicit `pos_weight` handles the
  ~1:1700 class imbalance better than focal's easy-example suppression does.
- **v3, propagation** (`λ = 0.5`, 8 neighbours, radius 0.06 m): adds one message-passing step over
  the cell neighbour graph before thresholding, `logit_i ← logit_i + λ·mean(logit_j over
  neighbours)`. This converts the mask head's question from "do I look like this query" into "do
  I, or does my neighbourhood" — the only arm that alters the forward pass.

**The metric.** `slope` = mean cells recovered for particles above 20 GeV ÷ the same below 2 GeV.
It measures whether the matched cluster **grows with the shower**: > 1 means it does, ≈ 1 means a
fixed scale, < 1 means it shrinks. Confidence intervals from bootstrapping **events** (2,000
resamples), because particles within an event are not independent.

| arm | slope | 95% CI |
|---|---|---|
| control | 1.648 | [1.481, 1.842] |
| v1 coverage term (`coverage_weight 2.0`) | **1.844** | [1.684, 2.022] |
| v2 recall (`bce_pos_weight 20`) | 1.621 | [1.458, 1.798] |
| v3 neighbour propagation (`λ = 0.5`) | 1.723 | [1.566, 1.895] |

State: only v1 separates from the control at 95%. **Also state the scope limitation as a fact**:
all four arms were trained for 4,000 steps, at which point the control over-predicts cluster size
(~2.5× truth), whereas the converged 30,000-step model under-predicts. Leave what follows from
that to the discussion.

### 8d. Working-point scan (2-epoch pu200 checkpoint)
F₁ flat at **0.338–0.359** across the entire mask × object threshold grid (mask 0.05–0.9,
object 0.05–0.5); best 0.359 at mask 0.90 against 0.353 at the default 0.50. The object head
accepts 92.9% of true particles, and masks alone reproduce 0.491 of the 0.490 reported efficiency.

---

## 9. Training-length results (report as fact; interpretation is discussion)

pu200 barrel, identical objective and thresholds, same 500-event evaluation:

| optimiser steps | efficiency | purity | F₁ | mask size vs truth |
|---|---|---|---|---|
| 4,000 | 0.676 | 0.202 | 0.312 | 1.7× |
| 12,000 (2 epochs) | 0.635 | 0.317 | 0.423 | 1.3× |
| 30,000 (5 epochs) | 0.663 | 0.432 | 0.523 | 1.15× |

`val_loss` fell 70.93 → 57.25 across the 5-epoch run and was **still decreasing at the final
checkpoint** (57.32 → 57.27 → 57.25). Report both the table and that fact; the consequence belongs
in the discussion's limitations.

*(These use best-overlap matching from the training-diagnostic script. Either re-derive them with
the scorer for consistency, or label the table explicitly as diagnostic. Do not mix them with §4.2
without saying so.)*

---

## 10. §4.8 Summary of findings

A bulleted list of measurements, no interpretation. Suggested:
1. Efficiency is comparable (0.509 vs 0.524); purity favours MaskFormer (0.356 vs 0.267).
2. The efficiency gap grows with energy, reaching 2.6× above 20 GeV in CLUE's favour.
3. Truth cluster size grows 2.7× with energy; MaskFormer's predicted size falls, CLUE's rises 9×.
4. MaskFormer's energy loss is dominated by unclaimed energy (0.271); CLUE's by misassignment.
5. Energy resolution: MaskFormer 0.61–1.01 across three decades; CLUE rises to 6.37.
6. At jet level MaskFormer's energy scale is 1.15 against CLUE's 1.62, with half the fake jets.
7. Focal's presence, not its weight, determines whether the mask head trains.
8. Threshold choice does not move F₁ (flat 0.338–0.359).

---

## 11. Numbers still to fill in

| Item | How |
|---|---|
| Entire pu0 column of every figure | finish run → dump store → `scripts.score` → re-run `scripts.make_thesis_figures` |
| pu0 headline table | as above |
| pu0 CLUE tuning (new checkpoint ⇒ new store) | `python -m scripts.tune_clue` |
| pu0 working point | `python -m scripts.scan_working_points` |
| Statistical uncertainties on the headline table | bootstrap by event; the machinery is in `src/evaluation/differential.py` |
| Energy-split shower profiles, if wanted | three-band version of `fig_shower_profile` |

**Reproduce every figure:** `python -m scripts.make_thesis_figures --events 500`
(add `--rebuild-anatomy --rebuild-jets` after any change to the store or the scoring).
