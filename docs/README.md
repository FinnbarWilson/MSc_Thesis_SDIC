# Documents

Status notes and thesis planning, moved here on 2026-08-11 from four different directories so
that the repository has one place to look for prose rather than four.

## Status notes — the provenance for the negative results

| file | what it records |
|---|---|
| `HIGH_ENERGY_STATUS.md` | the six training interventions and eleven post-processing methods tried against the high-energy failure. This is the evidence behind the negative results the thesis reports. |
| `PU200_STATUS.md` | the pu200 sizing measurements: throughput, memory, worker counts, and the schedule arithmetic that follows from them. Cited by name from several configs. |
| `VARIANTS_HANDOVER.md` | outcomes of the three mask-head variants (`overlay_v1`–`v3`). |

## Thesis planning

`METHODOLOGY_PLAN.md`, `RESULTS_PLAN.md`, `DISCUSSION_PLAN.md` — chapter plans, including which
figure and table belongs in which section, and which numbers are still pending.

## Repository

`REPO_CLEANUP.md` — the 2026-08-10 audit. Most of it has now been carried out; §2's structural
move of `hepattn_colliderml/` to a top-level `model/` has **not**, and the reason is recorded at
the top of that file.
