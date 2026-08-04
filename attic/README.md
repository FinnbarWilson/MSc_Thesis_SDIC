# Superseded code

Nothing here is imported, run, or tested. It is kept because the decisions inside it are
still live even though the implementations are not — deleting it would put them one `git
log` away instead of one `ls`.

**None of it currently imports.** All four files date from the design where each method
opened the raw ColliderML parquet and applied the shared cuts itself, and they call
`src.config.dataset_paths` and `src.config.split_bounds`, which no longer exist. That design
was replaced by the event store: the cuts are applied once by the dump and travel with the
data, so "both methods saw identical cells" is structural rather than two configs agreeing.
See the top-level `README.md`.

Anything revived from here has to be rewritten against `src/io/event_store.py`, not
repaired in place.

| | |
|---|---|
| `data/loader.py` | read the calorimeter parquet, applied `hit_selection` and the event selection |
| `data/truth.py` | built the reconstructable-particle set and the footprint-sharing description from `contrib_particle_ids` / `contrib_energies` |
| `jets.py` | anti-kt jets through FastJet, truth and reconstructed built identically, matched on delta-R |
| `evaluate_clue.py` | the old CLUE entry point: load, cluster, score, jets, one JSON per run |
| `run_truth_study.py` | the isolation and footprint-contamination characterisation |

Two of these are worth being specific about, because their replacements are only partial.

**Jets are deferred, not abandoned.** `jets.py` is the implementation; the *decisions* —
truth jets from generator four-vectors, neutrinos and muons excluded, per-subsystem
calibration mandatory because it does not cancel in the response ratio, matching by
`linear_sum_assignment` rather than greedily — are recorded under `jets:` in
`config/experiment.yaml` so they do not get relitigated. What blocks it is
`particle_min_pt`: 46% of the calorimeter energy comes from particles below the 0.5 GeV cut,
so MaskFormer jets would be missing that energy by construction while CLUE's would not.

**The truth study's live half is `local_density` in `src/evaluation/metrics.py`**, which
computes each particle's nearest-neighbour distance and neighbour count directly on the
store and puts them in the particle table as `dr_min` / `n_within`. That is what
`figures/performance_vs_density` reads. What has no replacement is
`footprint_contamination` — the description of how much of each particle's footprint is
shared with its neighbours, as opposed to how crowded its neighbourhood is.
