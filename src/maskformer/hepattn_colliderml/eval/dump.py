"""Dump cells, truth and MaskFormer predictions into a portable event store.

This is the only GPU-dependent step of the CLUE comparison. It runs one forward pass per
event and writes a compact, plain-numpy record; everything downstream -- running CLUE,
matching, scoring, plotting -- happens in the thesis repository against that record, with
no hepattn, no torch and no dataset access.

Two things the obvious route gets wrong and this one avoids:

*   `main.py test` with `PredictionWriter` writes `flow_calohit_valid_prob` as a dense
    float32 `[1, 1000, n_hits]` tensor: ~88 MB per event, ~44 GB for 500 events, onto a
    filesystem that is 97% full. Here the masks go out as a sparse CSR of uint8 logit codes
    above a loose threshold -- roughly three orders of magnitude smaller, and it still
    supports re-deriving *any* working point offline.
*   Scoring against the metric in `model.py` would need `MaskFormer.loss()`, whose Hungarian
    permutation has no analogue for CLUE. Nothing here matches anything; the store holds raw
    predictions and the shared, model-agnostic matcher runs later on both algorithms.

Both prediction heads are written, which format version 1 did not do. The mask head answers
"is this cell claimed at all", the incidence head answers "whose is it, and in what
proportion". Only the first was ever exported, so every reported number resolved contested
cells with a quantity that was never trained to divide one -- see `eval/format.py`.

Usage (inside the container, from the experiment directory):

    python -m hepattn.experiments.colliderml.eval.dump <ckpt> \
        --start-event 20250 --num-events 500 --out ~/eventstore
"""

import argparse
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hepattn.experiments.colliderml.data import ColliderMLDataset
from hepattn.experiments.colliderml.eval import format as fmt
from hepattn.experiments.colliderml.eval import geometry as geo
from hepattn.experiments.colliderml.model import ColliderMLModel
from hepattn.models.task import IncidenceRegressionTask

PARTICLE_CLASS_FLAGS = (
    ("particle_is_photon", "photon"),
    ("particle_is_electron", "electron"),
    ("particle_is_muon", "muon"),
    ("particle_is_tau", "tau"),
    ("particle_is_neutrino", "neutrino"),
    ("particle_is_charged_hadron", "charged_hadron"),
    ("particle_is_neutral_hadron", "neutral_hadron"),
    ("particle_is_other", "other"),
)

# Mask entries below this logit are not stored. log(0.02 / 0.98).
STORE_LOGIT_FLOOR = float(np.log(fmt.STORE_MASK_THRESHOLD / (1.0 - fmt.STORE_MASK_THRESHOLD)))


def build_dataset(run_config: Path, start_event: int, num_events: int) -> ColliderMLDataset:
    """Rebuild the checkpoint's own data configuration over a different event window.

    Deriving the dataset from the run's `config.yaml` is what makes the comparison fair
    without anyone having to remember to keep two configs in step: the hit set CLUE will
    later cluster is, by construction, the hit set this checkpoint was evaluated on.

    A `ColliderMLDataset` is built directly rather than going through
    `ColliderMLDataModule`, whose `setup("test")` also constructs the *training* dataset and
    re-scans all 1000 parquet shards for nothing.
    """
    config = yaml.safe_load(run_config.read_text())["data"]
    accepted = set(inspect.signature(ColliderMLDataset.__init__).parameters)
    kwargs = {k: v for k, v in config.items() if k in accepted}
    kwargs.update(dirpath=config["test_dir"], start_event=start_event, num_events=num_events)
    return ColliderMLDataset(**kwargs)


def subsystem_codes(detector: np.ndarray) -> np.ndarray:
    """Map raw detector ids onto the four calorimeter subsystems."""
    codes = np.full(detector.shape, 255, dtype=np.uint8)
    for name, ids in ColliderMLDataset.CALO_SUBSYSTEM_DETECTOR_IDS.items():
        codes[np.isin(detector, ids)] = fmt.SUBSYSTEM_CODE[name]
    if (codes == 255).any():
        unknown = np.unique(detector[codes == 255])
        msg = f"calo hits with detector ids {unknown.tolist()} belong to no known subsystem"
        raise ValueError(msg)
    return codes


def calibrate_layers(dataset: ColliderMLDataset, num_events: int) -> dict[str, list[float]]:
    """Derive the layer geometry once, pooling several events.

    Pooling matters: HCal barrel sees only a few hundred hits per event spread over 36
    layers, so any single event leaves most of them unlit and a per-event derivation
    undercounts them (26 instead of 36). The layer counts are asserted afterwards, so a
    dataset change fails here rather than silently renumbering every layer in the store.
    """
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in geo.SUBSYSTEMS}
    for index in range(min(num_events, len(dataset))):
        inputs, _ = dataset[index]
        x = inputs["calohit_x"][0].numpy()
        y = inputs["calohit_y"][0].numpy()
        z = inputs["calohit_z"][0].numpy()
        codes = subsystem_codes(inputs["calohit_detector"][0].numpy())
        for name in geo.SUBSYSTEMS:
            selected = codes == fmt.SUBSYSTEM_CODE[name]
            if selected.any():
                pooled[name].append(geo.layer_depth(name, x[selected], y[selected], z[selected]))

    # Only subsystems that actually have cells. A barrel-only sample (configs/pu200.yaml
    # cuts |eta| < 0.88) has no endcap cells at all, so `ece` and `hce` are legitimately absent
    # and the counts are checked against what is present rather than against all four.
    populated = [name for name, depths in pooled.items() if depths]
    centres = {name: geo.calibrate_layer_centres(np.concatenate(pooled[name])) for name in populated}
    geo.check_layer_counts(centres, populated=populated)
    return {name: values.tolist() for name, values in centres.items()}


def _csr_from_dense(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack (row, col, value) triples into a row-major CSR with columns ascending."""
    order = np.lexsort((cols, rows))
    rows, cols, values = rows[order], cols[order], values[order]
    indptr = np.zeros(n_rows + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.bincount(rows, minlength=n_rows))
    return indptr, cols.astype(np.int32), values


def top_k_incidence(incidence: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a dense `[n_kept_queries, n_hits]` incidence block to the top k shares per cell.

    The incidence head softmaxes over queries, so every entry is nonzero and the dense block
    is as large as the probability tensor the store exists to avoid (~88 MB/event). Almost
    all of that is numerically irrelevant: truth divides a cell 1.22 ways, so beyond the
    first few queries the shares are rounding noise. Keeping the top k is what makes the head
    storable at all.

    Args:
        incidence: `[n_kept_queries, n_hits]` shares, already restricted to kept queries.
        k: how many (query, share) pairs to keep per cell.

    Returns:
        ``(query, share)``, both `[n_hits, k]`, descending in share along the second axis.
        Query is an index into the kept queries, or -1 where a cell has fewer than k of them.
    """
    n_queries, n_hits = incidence.shape
    query = np.full((n_hits, k), -1, dtype=np.int16)
    share = np.zeros((n_hits, k), dtype=np.float16)
    if n_queries == 0:
        return query, share

    take = min(k, n_queries)
    # argpartition is O(n) against argsort's O(n log n) and n here is the query count on
    # every one of ~24k cells, so this is worth not being a full sort.
    top = np.argpartition(-incidence, take - 1, axis=0)[:take]
    values = np.take_along_axis(incidence, top, axis=0)
    order = np.argsort(-values, axis=0)

    query[:, :take] = np.take_along_axis(top, order, axis=0).T.astype(np.int16)
    share[:, :take] = np.take_along_axis(values, order, axis=0).T.astype(np.float16)
    return query, share


def extract_event(
    inputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mask_logit: torch.Tensor,
    valid_prob: torch.Tensor,
    layer_centres: dict[str, list[float]],
    calibration: np.ndarray,
    mf_incidence: torch.Tensor | None = None,
    store_mask_threshold: float = fmt.STORE_MASK_THRESHOLD,
    max_hits_per_query: int = 0,
    incidence_top_k: int = fmt.INCIDENCE_TOP_K,
) -> dict[str, np.ndarray]:
    """Turn one event's tensors into the flat arrays the store holds.

    Args:
        inputs: loader inputs, each with a leading batch dimension of 1.
        targets: loader targets, likewise.
        mask_logit: `[num_queries, n_hits]` raw mask logits, on CPU, float32.
        valid_prob: `[num_queries]` object-head probabilities, on CPU, float32.
        layer_centres: frozen layer geometry per subsystem.
        calibration: sampling calibration indexed by subsystem code.
        mf_incidence: `[num_queries, n_hits]` PREDICTED incidence-head shares, on CPU,
            float32, or None for a checkpoint trained without an `IncidenceRegressionTask`.
            Named with the `mf_` prefix to keep it distinct from the truth `particle_incidence`
            unpacked below -- they are the same quantity from opposite sides, and the two must
            not be confused. Stored as the top `incidence_top_k` per cell; None writes width-0
            arrays and sets `maskformer.has_incidence` false in the metadata.
        incidence_top_k: how many (query, share) pairs to keep per cell.
        store_mask_threshold: mask probability below which entries are not stored; this sets
            the floor of any working point a later scan can reach.
        max_hits_per_query: cap on stored hits per query, 0 for uncapped. A safety valve
            against one uncommitted query with a flat logit distribution dominating the file.
    """
    x = inputs["calohit_x"][0].numpy()
    y = inputs["calohit_y"][0].numpy()
    z = inputs["calohit_z"][0].numpy()
    phi = inputs["calohit_phi"][0].numpy()
    energy = inputs["calohit_total_energy"][0].numpy()
    detector = inputs["calohit_detector"][0].numpy().astype(np.uint8)
    n_hits = int(x.size)

    subsystem = subsystem_codes(detector)
    layer = np.zeros(n_hits, dtype=np.uint8)
    assigned = np.zeros(n_hits, dtype=bool)
    for name, centres in layer_centres.items():
        selected = subsystem == fmt.SUBSYSTEM_CODE[name]
        if selected.any():
            layer[selected] = geo.assign_layers(name, x[selected], y[selected], z[selected], centres)
            assigned |= selected

    # Every cell must belong to a CALIBRATED subsystem. `layer` starts at zeros and is only
    # written for subsystems present in `layer_centres`, so a cell in an uncalibrated one would
    # silently be recorded as layer 0 -- a wrong value that no downstream check would catch.
    # This is reachable now that calibration accepts a subset of subsystems: the calibration scan
    # reads only the first `--layer-calib-events`, so a subsystem that is empty there but
    # populated later in the window would land here.
    if not assigned.all():
        missing = sorted({fmt.SUBSYSTEM_ORDER[c] for c in np.unique(subsystem[~assigned])})
        msg = (
            f"{int((~assigned).sum())} cells belong to subsystems with no layer calibration ({missing}). "
            "Raise --layer-calib-events so the calibration scan sees them, or exclude them from the sample."
        )
        raise ValueError(msg)

    # Geometry order: neighbouring cells end up adjacent, which is what makes the store and
    # every downstream label array compress.
    order = fmt.store_order(subsystem, layer, phi)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)

    x, y, z, energy = x[order], y[order], z[order], energy[order]
    detector, subsystem, layer = detector[order], subsystem[order], layer[order]

    # --- truth ---------------------------------------------------------------------
    # particle_incidence has a row for every particle surviving the kinematic cuts, but
    # only `particle_valid` rows are targets (the min-calohits cut is applied later).
    # Restricting to targets first is what makes the partition the efficiency denominator.
    particle_valid = targets["particle_valid"][0].numpy()
    valid_rows = np.flatnonzero(particle_valid)
    n_particles = int(valid_rows.size)

    incidence = targets["particle_incidence"][0].numpy()[valid_rows][:, order]
    owner = incidence.argmax(axis=0) if n_particles else np.zeros(n_hits, dtype=np.int64)
    owned = incidence.max(axis=0) > 0 if n_particles else np.zeros(n_hits, dtype=bool)
    cell_truth_label = np.where(owned, owner, -1).astype(np.int32)

    rows, cols = np.nonzero(incidence)
    truth_indptr, truth_indices, truth_incidence = _csr_from_dense(rows, cols, incidence[rows, cols], n_particles)

    def particle_field(key: str, dtype: np.dtype) -> np.ndarray:
        return targets[key][0].numpy()[valid_rows].astype(dtype)

    particle_class = np.full(n_particles, fmt.PARTICLE_CLASS_CODES["other"], dtype=np.uint8)
    for key, label in PARTICLE_CLASS_FLAGS:
        particle_class[particle_field(key, np.bool_)] = fmt.PARTICLE_CLASS_CODES[label]

    # --- MaskFormer ----------------------------------------------------------------
    kept_queries = np.flatnonzero(valid_prob.numpy() >= fmt.STORE_OBJECT_THRESHOLD)
    kept_logits = mask_logit.numpy()[kept_queries][:, order]
    floor = float(np.log(store_mask_threshold / (1.0 - store_mask_threshold)))
    mf_rows, mf_cols = np.nonzero(kept_logits >= floor)
    mf_values = kept_logits[mf_rows, mf_cols]

    # Safety valve. A query that never commits leaves a broad, flat logit distribution and
    # can contribute tens of thousands of near-threshold hits that no working point will
    # ever select. Capping keeps one pathological query from dominating the file; the count
    # of truncated queries is reported so a cap that actually bites is visible rather than
    # silent.
    if max_hits_per_query > 0:
        keep = np.ones(mf_rows.size, dtype=bool)
        for row in np.flatnonzero(np.bincount(mf_rows, minlength=kept_queries.size) > max_hits_per_query):
            in_row = np.flatnonzero(mf_rows == row)
            keep[in_row[np.argsort(mf_values[in_row])[:-max_hits_per_query]]] = False
        mf_rows, mf_cols, mf_values = mf_rows[keep], mf_cols[keep], mf_values[keep]

    mf_indptr, mf_indices, mf_codes = _csr_from_dense(mf_rows, mf_cols, fmt.quantise_logit(mf_values), int(kept_queries.size))

    # The incidence head, restricted to the same kept queries and reordered onto store order,
    # so its query axis lines up with `mf_valid_prob` and its cell axis with everything else.
    if mf_incidence is not None:
        kept_incidence = mf_incidence.numpy()[kept_queries][:, order]
        mf_incidence_query, mf_incidence_share = top_k_incidence(kept_incidence, incidence_top_k)
    else:
        mf_incidence_query = np.full((n_hits, 0), -1, dtype=np.int16)
        mf_incidence_share = np.zeros((n_hits, 0), dtype=np.float16)

    calibrated = energy * calibration[subsystem]
    indptr_key = "particle_calohit_indptr"
    n_untruncated = int(targets[indptr_key].shape[-1] - 1) if indptr_key in targets else n_particles

    return {
        "cell_x": x.astype(np.float32),
        "cell_y": y.astype(np.float32),
        "cell_z": z.astype(np.float32),
        "cell_energy": energy.astype(np.float32),
        "cell_detector": detector,
        "cell_subsystem": subsystem,
        "cell_layer": layer,
        "cell_truth_label": cell_truth_label,
        "truth_indptr": truth_indptr,
        "truth_indices": truth_indices,
        "truth_incidence": truth_incidence.astype(np.float32),
        "particle_id": particle_field("particle_particle_id", np.uint64),
        "particle_px": particle_field("particle_px", np.float32),
        "particle_py": particle_field("particle_py", np.float32),
        "particle_pz": particle_field("particle_pz", np.float32),
        "particle_energy": particle_field("particle_energy", np.float32),
        "particle_pt": particle_field("particle_pt", np.float32),
        "particle_eta": particle_field("particle_eta", np.float32),
        "particle_phi": particle_field("particle_phi", np.float32),
        "particle_pdg_id": particle_field("particle_pdg_id", np.int32),
        "particle_class": particle_class,
        "particle_num_calohits": particle_field("particle_num_calohits", np.int32),
        "particle_energy_calo_sum": particle_field("particle_energy_calo_sum", np.float32),
        "mf_query_index": kept_queries.astype(np.int16),
        "mf_valid_prob": valid_prob.numpy()[kept_queries].astype(np.float32),
        "mf_indptr": mf_indptr,
        "mf_indices": mf_indices,
        "mf_logit_u8": mf_codes.astype(np.uint8),
        "mf_incidence_query": mf_incidence_query,
        "mf_incidence_share": mf_incidence_share,
        "n_hits": np.array(n_hits, dtype=np.int32),
        "n_particles": np.array(n_particles, dtype=np.int32),
        "n_particles_untruncated": np.array(n_untruncated, dtype=np.int32),
        "truncated": np.array(n_untruncated > particle_valid.size, dtype=np.bool_),
        "event_energy_raw": np.array(energy.sum(), dtype=np.float32),
        "event_energy_calib": np.array(calibrated.sum(), dtype=np.float32),
        "event_energy_on_target_calib": np.array(calibrated[cell_truth_label >= 0].sum(), dtype=np.float32),
    }


def git_provenance(repo: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return "unknown"

    return {
        "repo": "hepattn",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--porcelain")),
        "module": __name__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt", type=Path, help="a trained CLUSTERING checkpoint")
    parser.add_argument("--config", type=Path, default=None, help="defaults to config.yaml beside the checkpoint")
    parser.add_argument("--start-event", type=int, required=True)
    parser.add_argument("--num-events", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory to hold the store")
    parser.add_argument("--name", default=None, help="store directory name; derived from the window by default")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--layer-calib-events", type=int, default=25)
    parser.add_argument(
        "--store-mask-threshold",
        type=float,
        default=fmt.STORE_MASK_THRESHOLD,
        help="mask probability below which entries are not stored; sets the floor of any later working-point scan",
    )
    parser.add_argument("--max-hits-per-query", type=int, default=0, help="cap stored hits per query (0 = uncapped)")
    parser.add_argument(
        "--incidence-top-k",
        type=int,
        default=fmt.INCIDENCE_TOP_K,
        help="(query, share) pairs kept per cell from the incidence head; 1 suffices for the exclusive metric",
    )
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_config = args.config or args.ckpt.parent.parent / "config.yaml"
    dataset = build_dataset(run_config, args.start_event, args.num_events)

    print(f"calibrating layer geometry over {args.layer_calib_events} events ...", flush=True)
    layer_centres = calibrate_layers(dataset, args.layer_calib_events)
    for name, centres in layer_centres.items():
        pitch = np.diff(np.asarray(centres))
        print(f"  {name:>4}: {len(centres):>3} layers, pitch {pitch.min() * 1e3:.2f}-{pitch.max() * 1e3:.2f} mm")

    calibration = np.array([ColliderMLDataset.CALO_SUBSYSTEM_CALIBRATION[n] for n in fmt.SUBSYSTEM_ORDER], dtype=np.float32)

    model = ColliderMLModel.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval().to(args.device)
    object_task = next(t for t in model.model.tasks if t.name == "flow_valid")

    # The incidence head is optional only because checkpoints predating it exist. It is found
    # by class rather than by name so a renamed task still resolves, and its output key is
    # read off the task instead of being spelled out here.
    incidence_task = next((t for t in model.model.tasks if isinstance(t, IncidenceRegressionTask)), None)
    if incidence_task is None:
        print("  ! this checkpoint has no IncidenceRegressionTask; storing masks only", flush=True)
    else:
        print(f"  incidence head {incidence_task.name!r} -> top {args.incidence_top_k} shares per cell", flush=True)

    trained = yaml.safe_load(run_config.read_text())["data"]
    meta = fmt.build_metadata(
        dataset={
            "dirpath": str(dataset.dirpath),
            "dataset_prefix": trained.get("dataset_prefix"),
            "event_type": trained.get("event_type", "ttbar"),
            "calo_only": trained.get("calo_only", True),
        },
        event_window={"start_event": args.start_event, "num_events": args.num_events},
        hit_selection={"calohit_min_energy": dataset.calohit_min_energy},
        particle_selection={
            "particle_min_pt": dataset.particle_min_pt,
            "particle_max_abs_eta": dataset.particle_max_abs_eta,
            "particle_min_num_calohits": dataset.particle_min_num_calohits,
            "event_max_num_particles": trained.get("event_max_num_particles"),
            # Which particles ARE the targets, as opposed to which of them survive the cuts above.
            # Without this the store cannot tell a per-Geant-particle truth from a shower-level
            # one -- the three cuts are identical either way, while the target set differs by 3x
            # and the energy it accounts for by 2.7x. Two stores dumped under different
            # definitions would then compare as equal, which is the exact failure
            # config/experiment.yaml exists to prevent.
            "particle_collapse_shower_secondaries": dataset.particle_collapse_shower_secondaries,
            "calo_entry_radius": dataset.calo_entry_radius,
            "calo_entry_abs_z": dataset.calo_entry_abs_z,
        },
        detector={
            "subsystem_order": list(fmt.SUBSYSTEM_ORDER),
            "subsystem_detector_ids": {k: v.tolist() for k, v in ColliderMLDataset.CALO_SUBSYSTEM_DETECTOR_IDS.items()},
            "subsystem_calibration": dict(ColliderMLDataset.CALO_SUBSYSTEM_CALIBRATION),
            "layer_rule": {
                "barrel": "r * cos(mod(phi + pi/N, 2pi/N) - pi/N), N = 16",
                "endcap": "abs(z)",
                "stave_symmetry": geo.STAVE_SYMMETRY,
                "gap_tolerance_m": geo.LAYER_GAP_TOLERANCE_M,
            },
            "layer_centres_m": layer_centres,
        },
        maskformer={
            "checkpoint": str(args.ckpt),
            "run_config": str(run_config),
            "trained_event_window": [trained.get("train_start_event", 0), trained.get("train_start_event", 0) + trained.get("num_train", 0)],
            "nominal_mask_threshold": fmt.NOMINAL_MASK_THRESHOLD,
            "nominal_object_threshold": fmt.NOMINAL_OBJECT_THRESHOLD,
            "store_mask_threshold": fmt.STORE_MASK_THRESHOLD,
            "store_object_threshold": fmt.STORE_OBJECT_THRESHOLD,
            "autocast_dtype": "bfloat16",
            "has_incidence": incidence_task is not None,
            "incidence_top_k": args.incidence_top_k if incidence_task is not None else 0,
            "incidence_normalisation": (
                "softmax over queries per cell, restricted to the kept queries and NOT "
                "renormalised over the stored k. Trained by KL divergence against "
                "particle_incidence, so a share is a predicted fraction of the cell's energy "
                "-- unlike a mask probability, which is an independent per-(query, cell) "
                "sigmoid and is constrained to sum to nothing."
            ),
        },
        producer=git_provenance(Path(__file__).resolve().parents[5]),
    )

    name = args.name or f"{trained.get('dataset_prefix', 'store')}_{args.start_event}_{args.start_event + args.num_events}_v{fmt.FORMAT_VERSION}"
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    buffer: list[dict[str, np.ndarray]] = []
    buffered_ids: list[int] = []
    chunk_index = 0
    n_pairs_seen: list[int] = []

    # bf16 autocast is mandatory, not an optimisation: outside Lightning the model runs in
    # fp32 and FlashAttention refuses ("only support fp16 and bf16 data type").
    with torch.no_grad(), torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16):
        for index, (inputs, targets) in enumerate(loader):
            sample_id = dataset.sample_ids[index]
            device_inputs = {k: v.to(args.device, non_blocking=True) for k, v in inputs.items()}

            outputs = model.model(device_inputs)
            final = outputs["final"]
            mask_logit = final["flow_calohit_assignment"]["flow_calohit_logit"][0].float()
            # The task's own predict() is used rather than a hand-rolled sigmoid, so validity
            # is 1 - P(null) for the binary and the multi-class head alike and this does not
            # silently depend on num_classes.
            valid_prob = object_task.predict(final["flow_valid"])["flow_valid_prob"][0].float()

            mf_incidence = None
            if incidence_task is not None:
                mf_incidence = final[incidence_task.name][incidence_task.incidence_key][0].float()

            record = extract_event(
                inputs,
                targets,
                mask_logit.cpu(),
                valid_prob.cpu(),
                layer_centres,
                calibration,
                mf_incidence=None if mf_incidence is None else mf_incidence.cpu(),
                store_mask_threshold=args.store_mask_threshold,
                max_hits_per_query=args.max_hits_per_query,
                incidence_top_k=args.incidence_top_k,
            )
            n_pairs_seen.append(int(record["mf_logit_u8"].size))

            # How sparse the masks actually are decides whether the working-point CSR is
            # viable at all, and it cannot be known without a trained model. Report it for
            # the first few events so a bad choice of --store-mask-threshold is caught on a
            # smoke run rather than after 500 events.
            if index < 3:
                probs = mask_logit.sigmoid()
                counts = {p: int((probs >= p).sum()) for p in (0.02, 0.05, 0.1, 0.2, 0.5)}
                print(f"  ev {sample_id}: mask entries above " + ", ".join(f"p={p}: {n}" for p, n in counts.items()), flush=True)
                # How much of each cell the stored top-k actually accounts for. If this is not
                # close to 1 the head is spreading a cell over more queries than k can hold and
                # the exclusive argmax, while still correct, is a weak summary of it.
                if mf_incidence is not None:
                    captured = float(record["mf_incidence_share"].astype(np.float32).sum(axis=1).mean())
                    print(f"           incidence: top-{args.incidence_top_k} captures {captured:.3f} of a cell on average", flush=True)

            buffer.append(record)
            buffered_ids.append(int(sample_id))
            if len(buffer) == args.chunk_size:
                fmt.write_chunk(out_dir / fmt.chunk_filename(chunk_index), buffer, meta, buffered_ids)
                print(f"  chunk {chunk_index}: {len(buffer)} events -> {out_dir / fmt.chunk_filename(chunk_index)}", flush=True)
                buffer, buffered_ids = [], []
                chunk_index += 1

    if buffer:
        fmt.write_chunk(out_dir / fmt.chunk_filename(chunk_index), buffer, meta, buffered_ids)
        print(f"  chunk {chunk_index}: {len(buffer)} events", flush=True)

    total = sum(f.stat().st_size for f in out_dir.glob("*.npz"))
    print(f"\n{args.num_events} events -> {out_dir}")
    print(f"  {total / 1e6:.1f} MB total, {total / max(args.num_events, 1) / 1e3:.0f} kB/event")
    print(f"  mask CSR entries per event: mean {np.mean(n_pairs_seen):.0f}, max {np.max(n_pairs_seen):.0f}")


if __name__ == "__main__":
    main()
