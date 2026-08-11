# The experiment configs

Every file here is either a **run that produced a reported number** or **one arm of a controlled
experiment the thesis reports**. None of them is scratch. If a config looks like clutter, this
table is the reason it is not: an examiner asking "you say you tried six interventions — where
are they?" should be able to point at the file.

Each is a jsonargparse overlay applied **on top of** `calo_clustering.yaml`, and some stack two
deep. The order on the command line matters — later files win, and `tasks` is a LIST, so any
overlay touching it must restate the whole list rather than merging into it.

## The runs behind the results

| config | what it is |
|---|---|
| `calo_clustering.yaml` | the base experiment. Every other file is a diff against this one. |
| `overlay_pu0_dice.yaml` | pu0 on ce-ai-1, dice-dominant mask objective. Data paths + objective. |
| `overlay_pu0_showers.yaml` | **pu0, current.** Shower-level truth, 400 queries. Stacks on `overlay_pu0_dice.yaml`. |
| `overlay_pu200_barrel.yaml` | pu200 restricted to \|eta\| < 0.88, so every target is contained. |
| `overlay_pu200_showers.yaml` | **pu200, current.** Shower-level truth, 1600 queries. Stacks on `overlay_pu200_barrel.yaml`. |

The two `_showers` overlays are the truth definition described in `config/experiment.yaml` under
`particle_selection`. Everything above them predates it and used per-Geant-particle truth.

## The mask-objective sweep

Three arms, 1500 steps each on the pu200 barrel config, scored by eff/purity on 10 test events.
Results in `external/sweep_mask_loss_results.txt`; the conclusion — that focal's *presence*
rather than its weight is what prevents collapse — is written up in the header of
`overlay_pu200_barrel.yaml`.

| config | arm |
|---|---|
| `sweep_a_dice_dominant.yaml` | dice 20 + focal 1 — the winner, and what both current runs use |
| `sweep_b_dice_only.yaml` | dice only — collapsed, sigmoid saturated at zero |
| `sweep_c_focal_control.yaml` | dice 5 + focal 20 — statistically tied with A |
| `overlay_sweep_short.yaml` | the shared short schedule the three arms run under |

A fourth arm, dice 1 + bce 1, also collapsed; it needs no file because it is the base config.

## The mask-head variants

Three attempts to fix the head under-claiming, run against a shared short schedule.

| config | intervention |
|---|---|
| `overlay_v1_coverage.yaml` | an energy-coverage term added to the mask loss |
| `overlay_v2_recall.yaml` | recall-biased BCE |
| `overlay_v3_propagation.yaml` | one message-passing step over the mask logits |
| `overlay_variants_short.yaml` | the shared schedule, applied last on top of each |

Handoff notes and outcomes: `docs/VARIANTS_HANDOVER.md`.

## The probe arms

Five single-variable probes asking *why* the masks are ordered wrongly, each changing exactly one
thing against an otherwise identical baseline.

| config | probe |
|---|---|
| `overlay_probe_maskattn.yaml` | masked attention off in the decoder |
| `overlay_probe_incidence.yaml` | the incidence head put back |
| `overlay_probe_exclusive.yaml` | mask target becomes the exclusive partition |
| `overlay_probe_posenc05.yaml` | position encoder correlation length 1 -> 0.5 |
| `overlay_probe_posenc02.yaml` | correlation length 1 -> 0.2 |

## Everything else

| config | why it exists |
|---|---|
| `overlay_long_schedule.yaml` | the step-starvation diagnosis: two runs at ~20,000 steps reached nearly the same loss despite 6.7x different data. Cited wherever a schedule is sized. |
| `overlay_metric_aligned.yaml` | energy-weighted mask loss, so the objective agrees with the reported metric. Restates `tasks` in full, so it reintroduces the incidence head if applied. |

## A trap worth knowing before editing any of these

`main.py` runs from the **hepattn checkout**, not from this directory. `ce_ai_1/env.sh` copies
these files into the checkout at launch, so editing a config here and starting a run picks up the
copy — this directory is the source of record and wins, automatically. Observed once as
`max_epochs` edited to 9 with `--print_config` still reporting 3. `../../verify_sync.sh` checks
the two trees agree.
