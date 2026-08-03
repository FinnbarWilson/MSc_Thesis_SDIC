"""The event-store format: the on-disk contract between the model and the analysis.

This file is the single source of truth for the layout. The thesis repository holds a
hand-written *mirror* of it (a reader), deliberately not an import: the whole point of the
format is that scoring and plotting need numpy and nothing else -- no hepattn, no torch, no
GPU, no access to the 991 GB dataset. Keeping this module torch-free means the writer and
the reader can be audited side by side.

What a store contains, per event: the calorimeter cells the model actually saw, the truth
partition over those cells, the truth particle table, and the MaskFormer output. CLUE is
then run over the *same* cells, in the other repository, which is what makes "both
algorithms see identical input" a structural property rather than something to remember.

The MaskFormer output is TWO heads, not one, and reading only the first was a measurement
error rather than a simplification. Format version 2 adds the incidence head; see the block
comment on `mf_incidence_query` for why the two are not interchangeable.

Two design choices are worth explaining.

**Per-event keys, not concatenated arrays.** Every array is stored as `e{sample_id}__{name}`.
An `.npz` is a zip, so reading one event decompresses only that event's members; a
concatenated layout would force the whole chunk through zlib to get at one event, and would
need offset bookkeeping in two repositories at once.

**Cells are stored in geometry order**, `lexsort((phi, layer, subsystem))`, not in the order
the loader produced them. Sorting costs nothing and buys ~26% on disk, because neighbouring
cells then have similar coordinates and identical subsystem/layer bytes; it also makes the
cluster-label arrays -- which are spatially coherent by construction -- compress far better.
Hit indices inside the CSR blocks refer to this order.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2

# Number of (query, share) pairs kept per cell from the incidence head. See the block comment
# on the incidence arrays in ARRAY_DTYPES for what they are; k decides how much of each cell's
# division survives into the store.
#
# 16, MEASURED rather than reasoned from the truth multiplicity. The first guess was 4, on the
# grounds that truth divides a cell 1.22 ways and the mask head 2.04, so four slots ought to
# hold the whole division. That was wrong: the incidence head is a softmax over 1000 queries
# and is far more diffuse than either. Cumulative share captured, over 3 events, restricted to
# the cells the mask head claims (the only ones the soft study ever divides):
#
#     k =  1  ->  0.554        k = 12  ->  0.952
#     k =  2  ->  0.733        k = 16  ->  0.964
#     k =  4  ->  0.861        k = 32  ->  0.983
#     k =  8  ->  0.928        k = 64  ->  0.993
#
# At k = 4 the measured claims-per-cell came back as exactly 4.000 -- pinned to k, i.e. the
# number was reporting the truncation rather than the model. 16 keeps 96% and costs ~1.6 MB per
# event; going to 32 buys two points for another 1.3 MB.
#
# k = 1 remains sufficient for the exclusive head-to-head, which only needs the argmax and is
# unaffected by any of this.
INCIDENCE_TOP_K = 16

# Order of the subsystem codes used by `cell_subsystem`; also the order of the calibration
# array the reader builds. Matches ColliderMLDataset.CALO_SUBSYSTEMS.
SUBSYSTEM_ORDER: tuple[str, ...] = ("ecb", "ece", "hcb", "hce")
SUBSYSTEM_CODE: Mapping[str, int] = {name: i for i, name in enumerate(SUBSYSTEM_ORDER)}

# Mask logits are stored as uint8 rather than float16. The decisions taken downstream are
# all thresholds on the probability, and a code that is linear in the *logit* is uniform in
# the quantity the model emits -- unlike linear-in-p, which wastes resolution in the middle
# and loses it exactly where the sigmoid saturates. The step is 16/255 = 0.063 in logit,
# i.e. under 1.6% relative on p near 0.5, far finer than any threshold that will be scanned.
LOGIT_MIN = -8.0
LOGIT_MAX = 8.0
LOGIT_LEVELS = 256

# Loose thresholds at which mask entries are *stored*, well below any working point that
# will be scanned. This is what makes post-hoc working-point curves possible without a
# second GPU pass and without the 88 MB/event dense probability tensor.
STORE_MASK_THRESHOLD = 0.02
STORE_OBJECT_THRESHOLD = 0.02

# Working point reported as nominal. Both are post-hoc and both are measured: the mask
# threshold is ObjectHitMaskTask.pred_threshold, and the object threshold was chosen from a
# sweep that showed 0.2 buys ~25% relative efficiency over the argmax default at flat purity.
NOMINAL_MASK_THRESHOLD = 0.5
NOMINAL_OBJECT_THRESHOLD = 0.2

PARTICLE_CLASS_CODES: Mapping[str, int] = {
    "photon": 0,
    "electron": 1,
    "muon": 2,
    "tau": 3,
    "neutrino": 4,
    "charged_hadron": 5,
    "neutral_hadron": 6,
    "other": 7,
}

# Every per-event array, with the dtype it must have on disk. Enforced on write so the
# reader can trust it, and mirrored in the reader so a hand-edited store fails loudly.
ARRAY_DTYPES: Mapping[str, np.dtype] = {
    # cells, in store order
    "cell_x": np.dtype(np.float32),
    "cell_y": np.dtype(np.float32),
    "cell_z": np.dtype(np.float32),
    "cell_energy": np.dtype(np.float32),
    "cell_detector": np.dtype(np.uint8),
    "cell_subsystem": np.dtype(np.uint8),
    "cell_layer": np.dtype(np.uint8),
    "cell_truth_label": np.dtype(np.int32),
    # multi-owner truth, particle-major CSR over cells
    "truth_indptr": np.dtype(np.int32),
    "truth_indices": np.dtype(np.int32),
    "truth_incidence": np.dtype(np.float32),
    # truth particles
    "particle_id": np.dtype(np.uint64),
    "particle_px": np.dtype(np.float32),
    "particle_py": np.dtype(np.float32),
    "particle_pz": np.dtype(np.float32),
    "particle_energy": np.dtype(np.float32),
    "particle_pt": np.dtype(np.float32),
    "particle_eta": np.dtype(np.float32),
    "particle_phi": np.dtype(np.float32),
    "particle_pdg_id": np.dtype(np.int32),
    "particle_class": np.dtype(np.uint8),
    "particle_num_calohits": np.dtype(np.int32),
    "particle_energy_calo_sum": np.dtype(np.float32),
    # MaskFormer, query-major CSR over cells
    "mf_query_index": np.dtype(np.int16),
    "mf_valid_prob": np.dtype(np.float32),
    "mf_indptr": np.dtype(np.int32),
    "mf_indices": np.dtype(np.int32),
    "mf_logit_u8": np.dtype(np.uint8),
    # MaskFormer incidence head, cell-major and dense in k: `[n_hits, INCIDENCE_TOP_K]`.
    #
    # This is a DIFFERENT quantity from the masks above and the distinction is the reason it
    # is stored at all. The mask head emits an independent sigmoid per (query, cell): nothing
    # in its loss constrains one cell's claims to sum to anything, so a mask probability is
    # not an energy fraction and normalising a set of them does not make one. The incidence
    # head emits a softmax OVER QUERIES per cell and is trained by KL divergence against
    # `particle_incidence`, i.e. against I_ia = E_ia / E_i, the share of each cell's energy
    # belonging to each particle. It is the head that was taught what a cell's division adds
    # up to, and until format version 2 it was computed on every forward pass and discarded.
    #
    # Stored as the top k shares per cell, descending, with the query axis indexed into the
    # KEPT queries (the same rows `mf_indptr` walks, not the model's raw 0..999), so a reader
    # can gate on `mf_valid_prob` without a second lookup table. Padding is -1 in the index
    # and 0.0 in the value, which happens when an event keeps fewer than k queries.
    #
    # The values are the raw softmax over all queries restricted to the kept ones, NOT
    # renormalised over k. Renormalising here would bake a value of k into the store and
    # silently change what a share means; a reader that wants a per-cell partition should
    # divide by the row sum itself.
    "mf_incidence_query": np.dtype(np.int16),
    "mf_incidence_share": np.dtype(np.float16),
}

# Per-event scalars, stored as 0-d arrays.
SCALAR_DTYPES: Mapping[str, np.dtype] = {
    "n_hits": np.dtype(np.int32),
    "n_particles": np.dtype(np.int32),
    "n_particles_untruncated": np.dtype(np.int32),
    "truncated": np.dtype(np.bool_),
    "event_energy_raw": np.dtype(np.float32),
    "event_energy_calib": np.dtype(np.float32),
    "event_energy_on_target_calib": np.dtype(np.float32),
}


def event_key(sample_id: int, name: str) -> str:
    """Key under which `name` is stored for one event."""
    return f"e{int(sample_id):06d}__{name}"


def chunk_filename(index: int) -> str:
    return f"chunk_{index:04d}.npz"


def quantise_logit(logit: np.ndarray) -> np.ndarray:
    """Encode mask logits as uint8 on a linear logit scale."""
    scaled = (np.asarray(logit, dtype=np.float64) - LOGIT_MIN) / (LOGIT_MAX - LOGIT_MIN) * (LOGIT_LEVELS - 1)
    return np.clip(np.round(scaled), 0, LOGIT_LEVELS - 1).astype(np.uint8)


def dequantise_logit(code: np.ndarray) -> np.ndarray:
    """Decode uint8 codes back to logits (the centre of each code's bin)."""
    return LOGIT_MIN + np.asarray(code, dtype=np.float64) * (LOGIT_MAX - LOGIT_MIN) / (LOGIT_LEVELS - 1)


def logit_code_for_threshold(probability: float) -> int:
    """Smallest uint8 code whose decoded probability is at least `probability`.

    Lets a working point be applied as an integer comparison on the stored codes,
    `mf_logit_u8 >= logit_code_for_threshold(p)`, with no float round-trip. The comparison
    is exact up to the half-step quantisation error of +/-0.031 in logit.
    """
    if not 0.0 < probability < 1.0:
        msg = f"threshold must be in (0, 1), got {probability}"
        raise ValueError(msg)
    logit = float(np.log(probability / (1.0 - probability)))
    return int(np.ceil((logit - LOGIT_MIN) / (LOGIT_MAX - LOGIT_MIN) * (LOGIT_LEVELS - 1)))


def store_order(subsystem: np.ndarray, layer: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Permutation putting cells into the canonical geometry order.

    Returns:
        Index array such that `array[store_order(...)]` is in store order.
    """
    return np.lexsort((phi, layer, subsystem))


def build_metadata(
    *,
    dataset: Mapping[str, object],
    event_window: Mapping[str, int],
    hit_selection: Mapping[str, float],
    particle_selection: Mapping[str, object],
    detector: Mapping[str, object],
    maskformer: Mapping[str, object],
    producer: Mapping[str, object],
) -> dict:
    """Assemble the metadata block embedded in every chunk.

    Everything the analysis side needs in order to avoid hardcoding a physical constant
    lives here: the cuts that defined the hit and particle sets, the subsystem detector-id
    groups and sampling calibrations, the layer geometry, and the checkpoint provenance.
    The reader validates these against the thesis config rather than restating them.
    """
    return {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": dict(producer),
        "units": {"length": "m", "energy": "GeV", "angle": "rad"},
        "coordinates": {
            "xyz_scale_applied_to_source": 1e-3,
            "phi_convention": "atan2(y, x) in (-pi, pi]",
            "cell_order": "np.lexsort((phi, cell_layer, cell_subsystem))",
        },
        "dataset": dict(dataset),
        "event_window": dict(event_window),
        "hit_selection": dict(hit_selection),
        "particle_selection": dict(particle_selection),
        "detector": dict(detector),
        "truth": {
            "exclusive_rule": "argmax_a particle_incidence[a, i]; ties -> lowest particle row",
            "incidence_normalisation": (
                "particle_incidence columns are normalised over TARGET particles only, not over all "
                "contributors, so they sum to 1 wherever any target deposited. The exclusive argmax is "
                "unaffected (the normalisation is a per-cell constant), but an energy denominator built "
                "from it over-attributes: use particle_energy_calo_sum for the true deposit."
            ),
            "unowned_label": -1,
            "true_deposit_field": "particle_energy_calo_sum",
        },
        "maskformer": dict(maskformer),
        "particle_class_codes": dict(PARTICLE_CLASS_CODES),
        "encoding": {
            "prob_encoding": {
                "kind": "logit_uint8",
                "logit_min": LOGIT_MIN,
                "logit_max": LOGIT_MAX,
                "levels": LOGIT_LEVELS,
            },
        },
    }


def write_chunk(path: Path, events: Sequence[Mapping[str, np.ndarray]], meta: Mapping[str, object], sample_ids: Sequence[int]) -> None:
    """Write one chunk of events, with the metadata embedded.

    Args:
        path: destination `.npz`.
        events: one mapping of array name -> array per event, keys drawn from
            `ARRAY_DTYPES` and `SCALAR_DTYPES`.
        meta: the metadata from `build_metadata`; the chunk block is added here.
        sample_ids: event ids, parallel to `events`.

    Raises:
        ValueError: if an array is missing, unknown, or has the wrong dtype. Enforced on
            write so the reader in the other repository can rely on it.
    """
    payload: dict[str, np.ndarray] = {}
    required = set(ARRAY_DTYPES) | set(SCALAR_DTYPES)

    for sample_id, event in zip(sample_ids, events, strict=True):
        missing = required - set(event)
        if missing:
            msg = f"event {sample_id} is missing arrays: {sorted(missing)}"
            raise ValueError(msg)
        unknown = set(event) - required
        if unknown:
            msg = f"event {sample_id} has unknown arrays: {sorted(unknown)}"
            raise ValueError(msg)

        for name, array in event.items():
            expected = ARRAY_DTYPES.get(name) or SCALAR_DTYPES[name]
            array = np.asarray(array)
            if array.dtype != expected:
                msg = f"event {sample_id}, array {name!r}: dtype is {array.dtype}, expected {expected}"
                raise ValueError(msg)
            payload[event_key(sample_id, name)] = array

    chunk_meta = dict(meta)
    chunk_meta["chunk"] = {"sample_ids": [int(s) for s in sample_ids], "n_events": len(sample_ids)}
    payload["meta_json"] = np.array(json.dumps(chunk_meta, sort_keys=True))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
