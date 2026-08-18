"""Where CLUE's residual splitting comes from, and the ceiling on fixing it by linking.

    python -m scripts.measure_split_origin
    python -m scripts.measure_split_origin --store eval --events 100

THE QUESTION. TICL runs CLUE3D and then a SEPARATE trackster-to-trackster linking stage, because
CLUE3D is deliberately tuned pure and splits one shower into several tracksters. This pipeline has
no equivalent stage inside a subsystem: `cluster_subsystem` runs CLUE's own 3D pass over layer
cluster centroids and whatever that pass declines to join stays separate, on the grounds that
`d_o_3d` is tuned against the reported metric (`clue.search`) rather than for purity, so the
split/merge trade has already been made at the right operating point. `link_across_subsystems`
exists for a different reason -- CLUE is run once per subsystem, so a shower crossing ECAL into
HCAL is cut in half before any algorithm gets a say -- and it repairs an artefact of that
partition, not CLUE's own fragmentation.

That argument is sound but it is an argument. This script measures it.

WHAT IS MEASURED.

1. WHERE THE SPLITS ARE. A target is split (`is_split_e`) when two or more clusters each hold at
   least `metrics.split_fraction` of its deposited energy. For each split target the DOMINANT
   SUBSYSTEM of every such fragment is taken, and the split is classified:

     intra-subsystem  all fragments in one subsystem   -- only an intra-subsystem stage can fix it
     cross-subsystem  fragments in two or more         -- `link_across_subsystems` is the stage

   This is the decomposition `link_radius_scan` cannot give: that scan varies one radius and
   watches a total move, so it conflates repairing a real split with merging two showers.

2. THE CEILING ON FIXING IT. An ORACLE linker, which is allowed to consult the truth and is
   therefore an upper bound no real linker can reach. Clusters are unioned when they are both
   >= `split_fraction` fragments of the SAME target, gated three ways:

     as-is         the configured `clue.link_radius`, i.e. today's pipeline
     + intra       oracle union of same-subsystem fragments      -- the stage TICL has and we do not
     + any         oracle union of all fragments regardless      -- a perfect linker of both kinds

   Each variant is rescored by `score_event`, so the columns are the reported metrics.

HOW TO READ IT. The `+ intra` row is the whole answer to "must I build an intra-subsystem linking
stage". It is what a PERFECT one would buy. If the f1 gain there is small, no real linker built on
geometry can be worth the complexity, and the current pipeline stands -- with a measured number
behind it rather than the reasoning above. If it is large, the stage is missing and the `+ any`
row bounds what both stages together are worth.

The oracle is an upper bound in one direction only, and that asymmetry is what makes it useful: a
small ceiling is conclusive, a large one only says the stage is worth ATTEMPTING.

Runs on the TUNE store by default. Whether to add a pipeline stage is a design decision and
design decisions are taken on the tuning window, for the same reason the CLUE parameters, the
linking radius and the MaskFormer thresholds are.

Numbers are printed and written to results/<dataset>/split_origin.csv.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.clue.pipeline import SUBSYSTEMS, cluster_event
from src.config import describe, results_dir, settings, store_expectations, store_path
from src.evaluation.matching import overlap_matrix
from src.evaluation.metrics import score_event
from src.io.event_store import EventStore


def cluster_subsystem_of(record, label: np.ndarray, n_clusters: int) -> np.ndarray:
    """Dominant subsystem of each cluster, by calibrated energy; -1 for empty clusters.

    Dominant rather than unique because `link_across_subsystems` may already have joined a
    cluster across a boundary, so "the subsystem a cluster is in" stops being well defined
    once linking is on. Taking the energy-dominant one keeps the classification total.
    """
    dominant = np.full(n_clusters, -1, dtype=np.int64)
    clustered = label >= 0
    if n_clusters == 0 or not clustered.any():
        return dominant

    n_sub = len(SUBSYSTEMS)
    flat = label[clustered] * n_sub + record.subsystem[clustered].astype(np.int64)
    energy = np.bincount(flat, weights=record.energy_calib[clustered], minlength=n_clusters * n_sub)
    dominant = energy.reshape(n_clusters, n_sub).argmax(axis=1)
    # A cluster holding no energy at all has no dominant subsystem; argmax would say 0.
    dominant[np.bincount(label[clustered], minlength=n_clusters) == 0] = -1
    return dominant


def fragment_matrix(record, label: np.ndarray, n_clusters: int, split_fraction: float) -> np.ndarray:
    """`[n_targets, n_clusters]` boolean: does this cluster hold >= `split_fraction` of this target?

    The same quantity :func:`~src.evaluation.matching.fragmentation` counts, kept as the matrix
    rather than its row sums because both the classification and the oracle need to know WHICH
    clusters a target was split into, not just how many.
    """
    n_truth = int(record.n_particles)
    if n_truth == 0 or n_clusters == 0:
        return np.zeros((max(n_truth, 0), max(n_clusters, 0)), dtype=bool)

    deposit = record.truth_deposit * record.calibration[record.subsystem]
    overlap = overlap_matrix(record.truth_label, label, deposit, n_truth, n_clusters)
    owned = record.truth_label >= 0
    total = np.bincount(record.truth_label[owned], weights=deposit[owned], minlength=n_truth)
    threshold = split_fraction * total[:, None]
    return overlap >= np.maximum(threshold, np.finfo(np.float64).tiny)


def classify_splits(fragments: np.ndarray, dominant: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per target, the number of fragments and how many distinct subsystems they span.

    Returns:
        ``(n_fragments, n_subsystems)``. A target with ``n_fragments > 1`` and
        ``n_subsystems == 1`` was split WITHIN one subsystem, which cross-subsystem linking
        cannot touch by construction.
    """
    n_targets = fragments.shape[0]
    n_fragments = fragments.sum(axis=1).astype(np.int32)
    n_subsystems = np.zeros(n_targets, dtype=np.int32)
    if fragments.size == 0:
        return n_fragments, n_subsystems

    for row in np.flatnonzero(n_fragments > 0):
        subs = dominant[fragments[row]]
        n_subsystems[row] = np.unique(subs[subs >= 0]).size
    return n_fragments, n_subsystems


def oracle_link(
    record,
    label: np.ndarray,
    n_clusters: int,
    split_fraction: float,
    same_subsystem_only: bool,
) -> tuple[np.ndarray, int]:
    """Union clusters that are fragments of the same target. AN UPPER BOUND, NOT AN ALGORITHM.

    This consults the truth, so it is not a linker anyone could run. It answers the only
    question worth asking before building one: what is the most a linking stage could possibly
    recover from this clustering?

    Args:
        same_subsystem_only: if True, only fragments whose dominant subsystems agree are
            joined -- the stage TICL has between CLUE3D and the PF interpretation, and the one
            this pipeline does not have. If False, any two fragments of one target are joined.

    Returns:
        ``(label, n_clusters)``, relabelled and compacted the way `cluster_event` leaves them.
    """
    if n_clusters == 0:
        return label, 0

    fragments = fragment_matrix(record, label, n_clusters, split_fraction)
    dominant = cluster_subsystem_of(record, label, n_clusters)

    parent = np.arange(n_clusters)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row in range(fragments.shape[0]):
        members = np.flatnonzero(fragments[row])
        if members.size < 2:
            continue
        if same_subsystem_only:
            # Join only within each subsystem, so a target split three ways across a boundary
            # has its same-side pieces joined and the boundary left alone.
            groups = [members[dominant[members] == s] for s in np.unique(dominant[members])]
        else:
            groups = [members]
        for group in groups:
            for other in group[1:]:
                ra, rb = find(int(group[0])), find(int(other))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(n_clusters)])
    out = label.copy()
    clustered = label >= 0
    out[clustered] = roots[label[clustered]]

    if not clustered.any():
        return out.astype(np.int32), 0
    used, compact = np.unique(out[clustered], return_inverse=True)
    relabelled = np.full(label.size, -1, dtype=np.int32)
    relabelled[clustered] = compact.astype(np.int32)
    return relabelled, int(used.size)


def summarise(particles: pd.DataFrame, clusters: pd.DataFrame, variant: str, wp: float) -> dict:
    """The reported metrics for one variant, matching `scan_link_radius.summarise`."""
    return {
        "variant": variant,
        "efficiency": float((particles["eff_e"] >= wp).mean()) if len(particles) else 0.0,
        "purity": float((clusters["pur_e"] >= wp).mean()) if len(clusters) else 0.0,
        "mean_eff": float(particles["eff_e"].mean()) if len(particles) else 0.0,
        "mean_pur": float(clusters["pur_e"].mean()) if len(clusters) else 0.0,
        "split_rate": float(particles["is_split_e"].mean()) if len(particles) else 0.0,
        "merge_rate": float(clusters["is_merge_e"].mean()) if len(clusters) else 0.0,
        "frag_frac": float(particles["frag_frac"].mean()) if len(particles) else 0.0,
        "clusters_per_event": float(clusters.groupby("sample_id").size().mean()) if len(clusters) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="tune_store", help="tune_store (default) or store")
    parser.add_argument("--events", type=int, default=0, help="0 = the whole window")
    args = parser.parse_args()

    cfg = settings()
    print(describe())

    kind = {"tune": "tune_store", "eval": "store"}.get(args.store, args.store)
    store = EventStore(store_path(kind), expect=store_expectations())
    n = len(store) if args.events <= 0 else min(args.events, len(store))

    params_path = results_dir() / "clue_parameters.json"
    if not params_path.exists():
        msg = f"{params_path} does not exist. Run `python -m scripts.tune_clue` first."
        raise SystemExit(msg)
    tuned = json.loads(params_path.read_text())
    params = {name: entry["parameters"] for name, entry in tuned["subsystems"].items()}

    split_fraction = cfg["metrics"]["split_fraction"]
    wp = cfg["metrics"]["working_points"][0]
    link_radius = cfg["clue"].get("link_radius", 0.0)
    print(f"\n{n} events from {store.root}")
    print(f"  link_radius {link_radius}   split_fraction {split_fraction}   working point {wp}\n")

    rows: dict[str, list] = {"as-is": [], "+ intra (oracle)": [], "+ any (oracle)": []}
    cluster_rows: dict[str, list] = {k: [] for k in rows}
    census = []

    for i in range(n):
        record = store[i]
        label, n_clusters = cluster_event(
            record,
            params,
            subsystems=tuple(cfg["detectors"]),
            coords=cfg["clue"]["coords"],
            backend=cfg["clue"]["backend"],
            min_cluster_hits=cfg["metrics"]["min_cluster_hits"],
            link_radius=link_radius,
        )

        variants = {
            "as-is": (label, n_clusters),
            "+ intra (oracle)": oracle_link(record, label, n_clusters, split_fraction, same_subsystem_only=True),
            "+ any (oracle)": oracle_link(record, label, n_clusters, split_fraction, same_subsystem_only=False),
        }
        for name, (lab, n_c) in variants.items():
            p, c, _ = score_event(
                record,
                lab,
                n_c,
                algo="clue",
                split_fraction=split_fraction,
                min_overlap=cfg["metrics"]["min_overlap"],
                min_overlap_frac=cfg["metrics"]["min_overlap_frac"],
            )
            rows[name].append(p)
            cluster_rows[name].append(c)

        # The census is taken on the as-is clustering: it describes what today's pipeline leaves
        # behind, which is the thing a new stage would be built to attack.
        fragments = fragment_matrix(record, label, n_clusters, split_fraction)
        dominant = cluster_subsystem_of(record, label, n_clusters)
        n_frag, n_sub = classify_splits(fragments, dominant)
        census.append(pd.DataFrame({"n_frag": n_frag, "n_sub": n_sub}))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n} events")

    table = pd.DataFrame([
        summarise(pd.concat(rows[name]), pd.concat(cluster_rows[name]), name, wp) for name in rows
    ])

    cen = pd.concat(census, ignore_index=True)
    split = cen["n_frag"] > 1
    n_targets = len(cen)
    intra = split & (cen["n_sub"] <= 1)
    cross = split & (cen["n_sub"] > 1)

    print("\n1. WHERE THE SPLITS ARE  (as-is clustering, targets pooled over events)")
    print(f"   targets                          {n_targets:,}")
    print(f"   split                            {split.mean():.4f}  ({int(split.sum()):,})")
    print(f"     all fragments in 1 subsystem   {intra.mean():.4f}  ({int(intra.sum()):,})"
          f"   <- only an intra-subsystem stage can fix these")
    print(f"     fragments across subsystems    {cross.mean():.4f}  ({int(cross.sum()):,})"
          f"   <- link_across_subsystems' remit")
    if int(split.sum()):
        print(f"   share of splits that are intra   {intra.sum() / split.sum():.4f}")

    print("\n2. CEILING ON FIXING IT  (oracle linkers -- upper bounds, not algorithms)")
    show = table.copy()
    base = show.loc[show["variant"] == "as-is"].iloc[0]
    show["f1"] = 2 * show["efficiency"] * show["purity"] / (show["efficiency"] + show["purity"]).replace(0, np.nan)
    show["d_f1"] = show["f1"] - (2 * base["efficiency"] * base["purity"] / (base["efficiency"] + base["purity"]))
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = results_dir() / "split_origin.csv"
    show.assign(
        targets=n_targets,
        split_frac=split.mean(),
        split_intra_frac=intra.mean(),
        split_cross_frac=cross.mean(),
        link_radius=link_radius,
        events=n,
    ).to_csv(out, index=False, float_format="%.6g")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
