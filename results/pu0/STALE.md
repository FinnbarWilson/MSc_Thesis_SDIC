# These numbers are superseded. Do not quote them.

Everything in this directory was scored against the **per-Geant-particle truth definition**, in
which every secondary a shower produced inside the calorimeter was its own target particle. That
definition was replaced on 2026-08-11 — see `config/experiment.yaml` under `particle_selection`,
and `src/maskformer/hepattn_colliderml/configs/overlay_pu0_showers.yaml` for the measurements.

Under the old definition, on pu0:

- 85.7% of targets were non-primary Geant secondaries, 71.8% born inside the calorimeter
- 83% of targets sat in a shower split into several, median sibling separation dR 0.045
- only 31.7% of the calorimeter's energy belonged to any target at all

`oracle_geometric` in `reference_table.csv` shows the consequence directly: an idealised method
given the true particle count **and** the true shower axes reaches efficiency 0.61. That ceiling
is a property of the truth definition, not of any clustering method, so every efficiency and
purity figure here is measured against a target set that cannot be reconstructed.

## What replaces them

A pu0 training run under the new definition started 2026-08-11 12:16 UTC. When it finishes, the
sequence is: dump the two event stores, re-tune CLUE, re-scan the MaskFormer working point,
re-score. `scripts/run_pu200_pipeline.sh` is the pu200 version of that sequence and documents the
order; the CLUE parameters and the working point are both **measurements that must be re-derived**
against the new truth, not inherited from the files here.

`src/io/event_store.py` will refuse to open an old store against the new config — the truth
definition is in its `CONTRACT_KEYS` — so the two cannot be mixed by accident.

## Why they are still here

They are the only pu0 numbers that exist until the new run lands, and the thesis text currently
references them. Delete this directory wholesale once `results/pu0/` has been rewritten, along
with this file. `figures/pu0/` has the same status for the same reason.
