from collections import OrderedDict
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import ClassVar

import awkward as ak
import h5py
import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# Use file-system sharing for tensors passed from dataloader workers to the main process. The
# default file-descriptor strategy runs out of FDs with many workers and large per-batch tensors
# (dense 1200 x ~29k calo masks here), crashing with "RuntimeError: received 0 items of ancdata".
# See https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/2 (also done in
# the cld/ and pixel/ experiments).
torch.multiprocessing.set_sharing_strategy("file_system")


class ColliderMLDataset(Dataset):
    CALO_SUBSYSTEMS = ("ecb", "ece", "hcb", "hce")
    CALO_SUBSYSTEM_DETECTOR_IDS: ClassVar[dict[str, np.ndarray]] = {
        "ecb": np.array([10], dtype=np.int64),
        "ece": np.array([9, 11], dtype=np.int64),
        "hcb": np.array([13], dtype=np.int64),
        "hce": np.array([12, 14], dtype=np.int64),
    }
    CALO_ECAL_DETECTOR_IDS = np.array([9, 10, 11], dtype=np.int64)
    CALO_HCAL_DETECTOR_IDS = np.array([12, 13, 14], dtype=np.int64)
    CALO_SUBSYSTEM_CALIBRATION: ClassVar[dict[str, float]] = {
        "ecb": 37.5,
        "ece": 38.7,
        "hcb": 45.0,
        "hce": 46.9,
    }
    CALO_GROUP_DETECTOR_IDS: ClassVar[dict[str, np.ndarray]] = {
        "ecalhits": CALO_ECAL_DETECTOR_IDS,
        "hcalhits": CALO_HCAL_DETECTOR_IDS,
    }
    PARTICLE_HIT_CUT_CLASS_TO_MASK: ClassVar[dict[str, str]] = {
        "all": "particle_valid",
        "charged_hadron": "particle_is_charged_hadron",
        "neutral_hadron": "particle_is_neutral_hadron",
        "electron": "particle_is_electron",
        "photon": "particle_is_photon",
        "muon": "particle_is_muon",
        "tau": "particle_is_tau",
        "neutrino": "particle_is_neutrino",
        "other": "particle_is_other",
    }
    PARTICLE_HIT_CUT_KEYS = (
        "min_num_sihit",
        "min_num_ecal",
        "min_num_hcal",
    )
    PARTICLE_HIT_CUT_DEFAULTS: ClassVar[dict[str, int]] = {
        "min_num_sihit": 0,
        "min_num_ecal": 0,
        "min_num_hcal": 0,
    }

    def __init__(
        self,
        dirpath: str,
        num_events: int = -1,
        start_event: int = 0,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        particle_hit_cuts: dict[str, dict[str, int]] | None = None,
        particle_include_charged: bool = True,
        particle_include_neutral: bool = True,
        particle_min_num_calohits: int = 0,
        # Merge Geant secondaries produced INSIDE the calorimeter back onto the particle that
        # entered it, so one shower is one target. See _build_calo_entry_ancestors for the
        # measurement that motivates it. False reproduces the historical per-particle truth.
        particle_collapse_shower_secondaries: bool = False,
        # The calorimeter front face, in the dataset's mm. Defaults are the measured ECAL barrel
        # inner radius and ECAL endcap front |z| of the ColliderML geometry; only used when
        # particle_collapse_shower_secondaries is on.
        calo_entry_radius: float = 1252.0,
        calo_entry_abs_z: float = 3202.0,
        event_max_num_particles: int | None = None,
        calohit_min_energy: float = 0.0,
        # Drop calo cells outside |eta|, i.e. restrict the event to a detector region. 0 disables
        # it, which is the pu0 behaviour. Added for pileup-200: see _apply_calohit_eta_cut.
        calohit_max_abs_eta: float = 0.0,
        calohit_truth_filter: bool = False,
        calohit_loss_weight_power: float = 0.0,
        calohit_loss_weight_clip: float = 10.0,
        particle_calohit_exclusive: bool = False,
        hit_eval_path: str | None = None,
        hit_filter_threshold: float = 0.1,
        hit_filter_as_feature: bool = False,
        return_calohits: bool = True,
        return_tracks: bool = True,
        event_type: str = "ttbar",
        build_calohit_associations: bool = True,
        dataset_prefix: str | None = None,
        calo_only: bool = False,
        # Decoded parquet row groups held PER DATALOADER WORKER. Host RSS scales as
        # num_workers x this x decoded shard size, which is what ties worker count to memory.
        #
        # DEFAULT 8 IS THE HISTORICAL VALUE AND IS DELIBERATELY UNCHANGED, so pu0 and every existing
        # config behave exactly as before. pu200 lowers it in configs/pu200.yaml,
        # where a decoded shard is 4.0 GB against pu0's ~1.5 GB and 24 workers at 8 slots would
        # need ~690 GB. Measured there: 24 workers x 3 slots = 518 GB resident.
        #
        # Lowering it trades per-worker cache hit rate for the ability to run more workers. That is
        # the right trade only when the job is dataloader-bound AND shards are large; do not lower
        # it for pu0, where the same 8 slots cost far less and the run is not memory-constrained.
        row_group_cache_size: int = 8,
        debug: bool = False,
    ):
        super().__init__()

        self.dirpath = Path(dirpath)
        # Directory layout is "<prefix>_<collection>". When no prefix is given we fall back
        # to the legacy pu200 layout ("<event_type>_pu200_<collection>") for backwards
        # compatibility; pu0 datasets pass e.g. dataset_prefix="ttbar_pu0".
        prefix = dataset_prefix if dataset_prefix is not None else f"{event_type}_pu200"
        self.collection_dirs = {name: self.dirpath / f"{prefix}_{name}" for name in ("particles", "tracker_hits", "calo_hits", "tracks")}

        # In calo-only mode there are no tracker hits or tracks (e.g. ColliderML pu0); the model
        # constituents come from calorimeter hits instead of silicon hits.
        self.calo_only = calo_only

        self.particle_hit_cuts = self._normalize_particle_hit_cuts(particle_hit_cuts)
        self._requires_calohits_for_hit_cuts = any(cuts["min_num_ecal"] > 0 or cuts["min_num_hcal"] > 0 for cuts in self.particle_hit_cuts.values())

        required_collections = {"particles"}
        if not calo_only:
            required_collections.add("tracker_hits")
        if return_calohits or self._requires_calohits_for_hit_cuts:
            required_collections.add("calo_hits")
        if return_tracks:
            required_collections.add("tracks")

        missing_dirs = [name for name in sorted(required_collections) if not self.collection_dirs[name].is_dir()]
        if missing_dirs:
            msg = f"Missing required dataset directories for '{event_type}': {missing_dirs}. Expected these under {self.dirpath}."
            raise ValueError(msg)

        # Use particle shards as the reference and keep only shard names that are
        # available in every required collection.
        shared_shard_names = self._get_shared_shard_names(required_collections)
        self.particle_shard_paths = [self.collection_dirs["particles"] / name for name in shared_shard_names]
        if not self.particle_shard_paths:
            msg = f"No shared parquet shards found for '{event_type}' under {self.dirpath}"
            raise ValueError(msg)

        # Cache parquet row-group metadata and a few decoded row-groups.
        # Must be initialised before counting events in shards.
        self._row_group_starts: dict[str, np.ndarray] = {}
        self._row_group_cache: OrderedDict[tuple[str, int], ak.Array] = OrderedDict()
        # Set from the constructor rather than hardcoded, so pu0 and pu200 can differ without one
        # editing a file the other reads. See the argument's own comment for the sizing.
        self._row_group_cache_size = row_group_cache_size

        # Build a dense sample index: each sample_id maps to (shard index, row/event index in shard).
        self.sample_index = []
        for shard_idx, particle_path in enumerate(self.particle_shard_paths):
            num_events_in_shard = self._get_num_events_in_shard(particle_path)
            self.sample_index.extend((shard_idx, event_idx) for event_idx in range(num_events_in_shard))

        num_events_available = len(self.sample_index)

        if num_events_available == 0:
            msg = f"No events found in {dirpath}"
            raise ValueError(msg)
        # start_event lets train/val/test carve out disjoint, non-overlapping windows of the same
        # directory (e.g. train = [0, num_train), val = [num_train, num_train + num_val)). Without
        # this, both sets start at event 0 and the validation metrics are contaminated by training
        # events. See ColliderMLDataModule for how the offsets are wired up.
        if start_event < 0 or start_event > num_events_available:
            msg = f"start_event {start_event} is out of range [0, {num_events_available}] for directory {dirpath}."
            raise ValueError(msg)
        if num_events < 0:
            num_events = num_events_available - start_event
        if start_event + num_events > num_events_available:
            msg = (
                f"Requested events [{start_event}, {start_event + num_events}) but only "
                f"{num_events_available} are available in the directory {dirpath}."
            )
            raise ValueError(msg)

        print(f"Found {num_events_available} available events, using {num_events} events starting at event {start_event}.")

        # Sample ID is an integer that can uniquely identify each event/sample, used for picking out events during eval etc
        self.num_events = num_events
        self.sample_ids = list(range(start_event, start_event + num_events))

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_include_charged = particle_include_charged
        self.particle_include_neutral = particle_include_neutral
        # For calo clustering: keep only particles that actually deposit calo energy as targets
        # (>0 excludes neutrinos and other particles that leave no calo hits). 0 disables the cut.
        self.particle_min_num_calohits = int(particle_min_num_calohits)

        # Truth definition. OFF reproduces the historical behaviour: every Geant particle that
        # deposited in the calorimeter is its own target, including the secondaries a shower
        # creates as it develops. Measured on pu0 shards 0/17/55, that definition makes 85.7% of
        # targets non-primary, 71.8% of them born inside the calorimeter volume, and leaves only
        # 31.7% of the calorimeter's energy owned by any target at all -- 83% of targets sit in a
        # shower that was split into several, with a median sibling separation of dR 0.045, well
        # inside one Molière radius. Those fragments are not separable by any algorithm, so they
        # put a ceiling on efficiency and purity that has nothing to do with the method.
        #
        # ON collapses each such fragment onto the ancestor that ENTERED the calorimeter. Not onto
        # the generator-level root: a pi0 decays to two photons at the primary vertex and those are
        # two genuinely separate showers, which collapsing to the root would wrongly fuse (measured
        # 91 targets/event at 137 cells each). Stopping at the calo face keeps them apart while
        # still merging what one shower produced, and lands at 86.4% energy coverage.
        self.particle_collapse_shower_secondaries = bool(particle_collapse_shower_secondaries)
        self.calo_entry_radius = float(calo_entry_radius)
        self.calo_entry_abs_z = float(calo_entry_abs_z)
        # Set per event by _build_calo_entry_ancestors and consumed by the contrib-id lookups.
        # None means "no remapping", which is what the OFF path leaves it as.
        self._contrib_id_remap: tuple[np.ndarray, np.ndarray] | None = None

        # Fixed number of particle slots per event. MaskFormer requires the target object dimension
        # to equal num_queries, so set this equal to num_queries in the config. None -> pad to the
        # per-event particle count only (not usable with the MaskFormer matching loss).
        self.event_max_num_particles = event_max_num_particles
        # Zero-suppression: drop calo cells with total_energy below this threshold (mostly noise).
        # 0 disables it. Applied to inputs and the truth mask consistently in _add_calohits.
        self.calohit_min_energy = float(calohit_min_energy)
        self.calohit_max_abs_eta = float(calohit_max_abs_eta)
        # Diagnostic truth hit filter — see _add_calohits. Drops hits not on any valid target
        # particle, emulating a perfect trained filter (the trackml two-stage recipe).
        self.calohit_truth_filter = bool(calohit_truth_filter)

        # Per-cell weight on the mask loss, exposed as the `calohit_loss_weight` target and
        # consumed by ObjectHitMaskTask.constituent_weight_field. 0 disables it (every cell
        # worth the same, the historical behaviour); see _build_calohit_loss_weight for why
        # the default when enabled is 0.5 rather than 1.0.
        self.calohit_loss_weight_power = float(calohit_loss_weight_power)
        self.calohit_loss_weight_clip = float(calohit_loss_weight_clip)

        # Build `particle_calohit_valid` as an EXCLUSIVE partition (each cell True for the one
        # particle depositing the most energy in it) rather than the multi-owner default (True
        # for every contributor).
        #
        # This exists because training and evaluation disagreed. The reported metric scores an
        # exclusive partition -- it has to, since CLUE produces one and cannot do otherwise --
        # while the mask head was trained on the multi-owner mask, i.e. explicitly taught to
        # claim cells it does not dominate and then marked down for doing so. Exclusive truth
        # keeps ~83% of (particle, cell) associations and ~94% of each particle's own energy,
        # so the information given up is small and the train/eval mismatch it removes is not.
        #
        # Note this changes only the MASK target. `particle_incidence` is untouched and stays
        # fractional, which is the point: the incidence head is where shared cells are
        # represented, and it is supervised on the real division.
        self.particle_calohit_exclusive = bool(particle_calohit_exclusive)

        # Stage-2 hit filtering: path to the h5 written by a trained hit filter
        # (configs/calo_hit_filter.yaml, run in `test` mode). Hits whose predicted probability is
        # below hit_filter_threshold are dropped, mirroring trackml/data.py. This is the production
        # counterpart of calohit_truth_filter, which uses truth and is only a diagnostic.
        self.hit_eval_path = hit_eval_path
        self.hit_filter_threshold = float(hit_filter_threshold)
        # Expose the filter's per-hit probability to the clustering model as an input feature
        # (`calohit_filter_prob`) instead of only using it to delete hits.
        #
        # Worth it because our filter is a WEAK discriminator: measured AUC 0.80, because the hits
        # it rejects are not detector noise but real deposits from particles just below the 0.5 GeV
        # pT cut — a 0.45 GeV shower looks like a 0.55 GeV one, so the boundary is genuinely fuzzy.
        # Hard thresholding a weak score throws information away irreversibly and caps efficiency;
        # handing the score to the model lets it treat "probably from something soft" as evidence
        # and still recover the hit. Set the threshold low (or 0) to use this as pure soft filtering.
        self.hit_filter_as_feature = bool(hit_filter_as_feature)
        self._last_filter_probs: torch.Tensor | None = None
        if self.hit_eval_path is not None:
            if self.calohit_truth_filter:
                msg = "Set at most one of calohit_truth_filter and hit_eval_path — they are two ways of doing the same filtering."
                raise ValueError(msg)
            if not Path(self.hit_eval_path).is_file():
                msg = f"hit_eval_path does not exist: {self.hit_eval_path}"
                raise ValueError(msg)
            print(f"Filtering calo hits with {self.hit_eval_path} at threshold {self.hit_filter_threshold}")

        # Whether to return calo/ACTS track collections
        self.return_calohits = return_calohits
        self.return_tracks = return_tracks
        self.event_type = event_type
        self.build_calohit_associations = build_calohit_associations
        self.debug = debug

    @staticmethod
    def _coerce_non_negative_int(name: str, value) -> int:
        if isinstance(value, bool):
            msg = f"Cut '{name}' must be numeric, got bool."
            raise TypeError(msg)
        as_float = float(value)
        if as_float < 0:
            msg = f"Cut '{name}' must be non-negative, got {value}."
            raise ValueError(msg)
        as_int = int(as_float)
        if as_int != as_float:
            msg = f"Cut '{name}' must be an integer-like value, got {value}."
            raise ValueError(msg)
        return as_int

    @classmethod
    def _normalize_particle_hit_cuts(cls, particle_hit_cuts: dict[str, dict[str, int]] | None) -> dict[str, dict[str, int]]:
        if particle_hit_cuts is None:
            return {}
        if not isinstance(particle_hit_cuts, dict):
            msg = "particle_hit_cuts must be a dict like {'electron': {'min_num_sihit': 6, ...}}."
            raise TypeError(msg)

        normalized: dict[str, dict[str, int]] = {}
        for particle_type, cut_cfg in particle_hit_cuts.items():
            class_key = str(particle_type).strip().lower()
            if class_key not in cls.PARTICLE_HIT_CUT_CLASS_TO_MASK:
                valid = ", ".join(sorted(set(cls.PARTICLE_HIT_CUT_CLASS_TO_MASK)))
                msg = f"Unknown particle_hit_cuts class '{particle_type}'. Valid classes: {valid}"
                raise ValueError(msg)
            if not isinstance(cut_cfg, dict):
                msg = f"particle_hit_cuts['{particle_type}'] must be a dict."
                raise TypeError(msg)

            mask_key = cls.PARTICLE_HIT_CUT_CLASS_TO_MASK[class_key]
            if mask_key not in normalized:
                normalized[mask_key] = cls.PARTICLE_HIT_CUT_DEFAULTS.copy()

            for cut_name, cut_value in cut_cfg.items():
                cut_key = str(cut_name).strip().lower()
                if cut_key not in cls.PARTICLE_HIT_CUT_KEYS:
                    valid = ", ".join(sorted(cls.PARTICLE_HIT_CUT_KEYS))
                    msg = f"Unknown cut key '{cut_name}' for '{particle_type}'. Valid keys: {valid}"
                    raise ValueError(msg)
                cut_threshold = cls._coerce_non_negative_int(cut_name, cut_value)
                normalized[mask_key][cut_key] = max(normalized[mask_key][cut_key], cut_threshold)

        return {mask_key: cfg for mask_key, cfg in normalized.items() if any(cfg[cut_key] > 0 for cut_key in cls.PARTICLE_HIT_CUT_KEYS)}

    def __len__(self):
        return int(self.num_events)

    def _get_shared_shard_names(self, required_collections: set[str]) -> list[str]:
        shared_names = {path.name for path in self.collection_dirs["particles"].glob("*.parquet")}
        for collection in required_collections:
            if collection == "particles":
                continue
            shared_names &= {path.name for path in self.collection_dirs[collection].glob("*.parquet")}
        return sorted(shared_names)

    def _get_row_group_starts(self, path: Path) -> np.ndarray:
        path_key = str(path)
        if path_key in self._row_group_starts:
            return self._row_group_starts[path_key]

        parquet_file = pq.ParquetFile(path)
        num_row_groups = parquet_file.metadata.num_row_groups
        row_group_sizes = np.fromiter(
            (parquet_file.metadata.row_group(i).num_rows for i in range(num_row_groups)),
            dtype=np.int64,
            count=num_row_groups,
        )

        row_group_starts = np.zeros(num_row_groups + 1, dtype=np.int64)
        row_group_starts[1:] = np.cumsum(row_group_sizes)
        self._row_group_starts[path_key] = row_group_starts
        return row_group_starts

    def _get_num_events_in_shard(self, path: Path) -> int:
        row_group_starts = self._get_row_group_starts(path)
        return int(row_group_starts[-1])

    def _get_collection_path(self, collection: str, particle_path: Path) -> Path:
        return self.collection_dirs[collection] / particle_path.name

    def _get_row_group_lookup(self, path: Path, event_idx: int) -> tuple[int, int]:
        row_group_starts = self._get_row_group_starts(path)
        num_rows = int(row_group_starts[-1])
        if event_idx < 0 or event_idx >= num_rows:
            msg = f"Event index {event_idx} is out of range [0, {num_rows - 1}] for {path}"
            raise ValueError(msg)

        row_group_idx = int(np.searchsorted(row_group_starts, event_idx, side="right") - 1)
        row_in_group_idx = int(event_idx - row_group_starts[row_group_idx])
        return row_group_idx, row_in_group_idx

    def _get_row_group_array(self, path: Path, row_group_idx: int) -> ak.Array:
        key = (str(path), row_group_idx)
        if key in self._row_group_cache:
            self._row_group_cache.move_to_end(key)
            return self._row_group_cache[key]

        row_group_array = ak.from_parquet(path, row_groups=[row_group_idx])
        self._row_group_cache[key] = row_group_array
        self._row_group_cache.move_to_end(key)

        if len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)

        return row_group_array

    def _read_event_from_file(self, path: Path, event_idx: int) -> ak.Record:
        row_group_idx, row_in_group_idx = self._get_row_group_lookup(path, event_idx)
        row_group_array = self._get_row_group_array(path, row_group_idx)
        return row_group_array[row_in_group_idx]

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[ColliderMLDataset] {message}", flush=True)

    @staticmethod
    def _record_to_tensors(
        record: ak.Record,
        prefix: str,
        *,
        int_fields: set[str] | None = None,
        skip_fields: set[str] | None = None,
        default_dtype: torch.dtype | None = torch.float32,
    ) -> dict[str, torch.Tensor]:
        int_fields = set() if int_fields is None else set(int_fields)
        skip_fields = set() if skip_fields is None else set(skip_fields)
        tensors: dict[str, torch.Tensor] = {}

        for field in record.fields:
            if field in skip_fields:
                continue

            tensor = ak.to_torch(record[field])
            if field in int_fields:
                tensor = tensor.to(torch.int64)
            elif default_dtype is not None:
                tensor = tensor.to(default_dtype)
            tensors[f"{prefix}_{field}"] = tensor

        return tensors

    @staticmethod
    def _scale_xyz_inplace(tensors: dict[str, torch.Tensor], prefix: str, scale: float) -> None:
        for axis in ("x", "y", "z"):
            key = f"{prefix}_{axis}"
            if key in tensors:
                tensors[key] = tensors[key] * scale

    @staticmethod
    def _add_cylindrical_coords_inplace(tensors: dict[str, torch.Tensor], prefix: str) -> None:
        x = tensors[f"{prefix}_x"]
        y = tensors[f"{prefix}_y"]
        z = tensors[f"{prefix}_z"]
        r = torch.sqrt(x**2 + y**2)
        s = torch.sqrt(r**2 + z**2).clamp_min(1e-12)
        cos_theta = (z / s).clamp(-1.0, 1.0)

        tensors[f"{prefix}_r"] = r
        tensors[f"{prefix}_s"] = s
        tensors[f"{prefix}_eta"] = torch.arctanh(z / s)
        tensors[f"{prefix}_theta"] = torch.arccos(cos_theta)
        tensors[f"{prefix}_phi"] = torch.arctan2(y, x)

    @staticmethod
    def _valid_mask_like(values: torch.Tensor) -> torch.Tensor:
        return torch.full_like(values, True, dtype=torch.bool)

    @staticmethod
    def _apply_mask_to_prefixed_keys(
        tensors: dict[str, torch.Tensor],
        prefix: str,
        mask: torch.Tensor,
        *,
        skip_prefixes: tuple[str, ...] = (),
    ) -> None:
        for key in list(tensors.keys()):
            if not key.startswith(prefix):
                continue
            if any(key.startswith(skip) for skip in skip_prefixes):
                continue
            tensors[key] = tensors[key][mask]

    @staticmethod
    def _lookup_row_indices(reference_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if reference_ids.size == 0 or query_ids.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(query_ids.size, dtype=bool)

        sort_idx = np.argsort(reference_ids)
        sorted_reference_ids = reference_ids[sort_idx]

        search_idx = np.searchsorted(sorted_reference_ids, query_ids)
        in_bounds = search_idx < sorted_reference_ids.size
        matched = np.zeros(query_ids.shape[0], dtype=bool)
        matched[in_bounds] = sorted_reference_ids[search_idx[in_bounds]] == query_ids[in_bounds]

        if not np.any(matched):
            return np.zeros(0, dtype=np.int64), matched

        row_indices = sort_idx[search_idx[matched]]
        return row_indices, matched

    def _build_calo_entry_ancestors(self, particles: ak.Record) -> np.ndarray:
        """Row index of each particle's calorimeter-entering ancestor.

        Climbs `parent_id` for as long as the current particle was BORN inside the calorimeter,
        stopping at the first ancestor produced before the front face -- that is the particle that
        entered the calorimeter and whose shower everything above it belongs to. A particle born
        outside the calorimeter is its own ancestor, so tracker conversions still give one target
        per outgoing leg.

        The chain also stops when `parent_id` names a particle absent from this event's table,
        which is how generator-level roots terminate.

        Also stashes the id -> ancestor-id map in `self._contrib_id_remap`, because the calo hits
        record its contributions by particle id and every one of them has to be re-pointed at the
        ancestor before it is looked up.

        The front face is a measurement, not a tuned parameter: the ColliderML calo-hit cloud puts
        the ECAL barrel at r >= 1252 mm and the ECAL endcap at |z| >= 3202 mm. Moving the boundary
        inwards barely moves the answer (r > 1000 / |z| > 2800 gives 156 targets/event against 176
        here); only pushing it INSIDE the ECAL breaks it, since secondaries made in the first
        layers then stop counting as shower products (r > 1400 gives 401).
        """
        particle_ids = ak.to_numpy(particles["particle_id"]).astype(np.int64, copy=False)
        parent_ids = ak.to_numpy(particles["parent_id"]).astype(np.int64, copy=False)
        num_particles = particle_ids.size
        if num_particles == 0:
            self._contrib_id_remap = None
            return np.zeros(0, dtype=np.int64)

        vx, vy, vz = (ak.to_numpy(particles[axis]).astype(np.float64, copy=False) for axis in ("vx", "vy", "vz"))
        born_in_calo = (np.hypot(vx, vy) >= self.calo_entry_radius) | (np.abs(vz) >= self.calo_entry_abs_z)

        parent_rows, has_parent = self._lookup_row_indices(particle_ids, parent_ids)
        # _lookup_row_indices returns rows only for the matched entries; scatter them back so the
        # array can be indexed by row, and leave unmatched rows pointing at themselves.
        parent_row_of = np.arange(num_particles, dtype=np.int64)
        parent_row_of[has_parent] = parent_rows

        # Pointer doubling rather than a per-particle while loop: chains reach depth 8 here and an
        # event holds thousands of particles, so this runs in the dataloader budget. The iteration
        # cap also makes a corrupt cyclic chain terminate instead of hanging a worker.
        ancestors = np.arange(num_particles, dtype=np.int64)
        for _ in range(64):
            climbing = born_in_calo[ancestors] & has_parent[ancestors]
            if not climbing.any():
                break
            nxt = ancestors.copy()
            nxt[climbing] = parent_row_of[ancestors[climbing]]
            if np.array_equal(nxt, ancestors):
                break
            ancestors = nxt

        sort_idx = np.argsort(particle_ids)
        self._contrib_id_remap = (particle_ids[sort_idx], particle_ids[ancestors][sort_idx])
        return ancestors

    def _map_contrib_particle_ids(self, flat_contrib_particle_ids: np.ndarray) -> np.ndarray:
        """Re-point each calo-hit contribution at its shower's calo-entering particle.

        A no-op unless particle_collapse_shower_secondaries is on. Contributions from ids absent
        from the particle table are left alone; they fail the target lookup afterwards either way.
        """
        if self._contrib_id_remap is None:
            return flat_contrib_particle_ids

        keys, values = self._contrib_id_remap
        if keys.size == 0 or flat_contrib_particle_ids.size == 0:
            return flat_contrib_particle_ids

        search_idx = np.searchsorted(keys, flat_contrib_particle_ids)
        in_bounds = search_idx < keys.size
        found = np.zeros(flat_contrib_particle_ids.shape, dtype=bool)
        found[in_bounds] = keys[search_idx[in_bounds]] == flat_contrib_particle_ids[in_bounds]

        mapped = flat_contrib_particle_ids.copy()
        mapped[found] = values[search_idx[found]]
        return mapped

    @staticmethod
    def _build_csr_components(
        num_rows: int,
        num_cols: int,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        col_indices = np.asarray(col_indices, dtype=np.int64)

        if row_indices.size == 0:
            indptr = np.zeros(num_rows + 1, dtype=np.int64)
            indices = np.zeros(0, dtype=np.int64)
            shape = np.array([num_rows, num_cols], dtype=np.int64)
            return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(shape)

        # Build canonical CSR with optimized sparse backend:
        # deduplicates duplicate entries and stores row-sorted indices.
        values = np.ones(row_indices.shape[0], dtype=np.uint8)
        csr = sp.coo_matrix((values, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.uint8).tocsr()
        csr.sum_duplicates()
        indptr = np.array(csr.indptr, dtype=np.int64, copy=True)
        indices = np.array(csr.indices, dtype=np.int64, copy=True)
        shape = np.array([num_rows, num_cols], dtype=np.int64)
        return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(shape)

    @staticmethod
    def _empty_csr_components(num_rows: int, num_cols: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty = np.zeros(0, dtype=np.int64)
        return ColliderMLDataset._build_csr_components(num_rows, num_cols, empty, empty)

    @staticmethod
    def _float32_bincount(
        indices: np.ndarray,
        *,
        minlength: int,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        counts = np.bincount(indices, weights=weights, minlength=minlength)
        return counts.astype(np.float32, copy=False)

    @staticmethod
    def _get_calohit_detector_ids(calohits: ak.Record) -> np.ndarray:
        return np.asarray(ak.to_numpy(calohits["detector"]), dtype=np.int64).reshape(-1)

    def _build_particle_sihit_csr(
        self,
        particle_ids: torch.Tensor,
        sihit_particle_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        num_sihits = int(sihit_particle_ids.numel())
        if num_particles == 0 or num_sihits == 0:
            return self._empty_csr_components(num_particles, num_sihits)

        particle_ids_np = particle_ids.cpu().numpy()
        sihit_particle_ids_np = sihit_particle_ids.cpu().numpy()

        row_indices, matched = self._lookup_row_indices(particle_ids_np, sihit_particle_ids_np)
        if row_indices.size == 0:
            return self._empty_csr_components(num_particles, num_sihits)

        col_indices = np.nonzero(matched)[0]
        return self._build_csr_components(num_particles, num_sihits, row_indices, col_indices)

    def _build_particle_calohit_csr(self, particle_ids: torch.Tensor, calohits) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        num_calohits = len(calohits["contrib_particle_ids"])
        if num_particles == 0 or num_calohits == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        contrib_counts = ak.to_numpy(ak.num(calohits["contrib_particle_ids"], axis=1))
        if contrib_counts.sum() == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        flat_contrib_particle_ids = self._map_contrib_particle_ids(ak.to_numpy(ak.flatten(calohits["contrib_particle_ids"], axis=1)))
        calohit_indices = np.repeat(np.arange(num_calohits, dtype=np.int64), contrib_counts)

        particle_ids_np = particle_ids.cpu().numpy()
        row_indices, matched = self._lookup_row_indices(particle_ids_np, flat_contrib_particle_ids)
        if row_indices.size == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        col_indices = calohit_indices[matched]
        return self._build_csr_components(num_particles, num_calohits, row_indices, col_indices)

    def _build_calohit_loss_weight(self, calibrated_energy: torch.Tensor) -> torch.Tensor:
        """Per-cell weight for the mask loss, normalised to mean 1 over the event.

        DICE and focal count cells: a 1 MeV cell and a 1 GeV cell move the loss by the same
        amount. Every number in the writeup is energy-weighted, because a calorimeter measures
        energy rather than occupancy and cell energies here span orders of magnitude. This is
        the term that makes the objective agree with the metric.

        Two details that are not cosmetic:

        *   The exponent defaults to 0.5, not 1. Raw energy as a BCE weight would hand a
            handful of cells per event almost all of the gradient -- the median ECAL cell is
            4.6e-4 GeV against shower cores three orders of magnitude above it -- which is a
            variance problem, not an emphasis. The square root keeps the ordering while
            compressing the dynamic range.
        *   The result is normalised to mean 1 and then clipped. Normalising keeps the
            effective loss scale (and therefore the tuned learning rate) unchanged when the
            weighting is switched on, so a run with it is comparable to a run without.

        Args:
            calibrated_energy: [num_calohits] per-cell energy after subsystem calibration.

        Returns:
            [num_calohits] float32 weights, mean 1, clipped to `calohit_loss_weight_clip`.
        """
        if self.calohit_loss_weight_power <= 0.0 or calibrated_energy.numel() == 0:
            return torch.ones_like(calibrated_energy, dtype=torch.float32)

        weight = calibrated_energy.to(torch.float32).clamp_min(0.0) ** self.calohit_loss_weight_power
        mean = weight.mean()
        if not torch.isfinite(mean) or mean <= 0:
            return torch.ones_like(weight)
        return (weight / mean).clamp(max=self.calohit_loss_weight_clip)

    def _exclusive_particle_calohit_valid(self, incidence: torch.Tensor) -> torch.Tensor:
        """Reduce a multi-owner mask to the exclusive (max-energy) partition.

        Derived from `particle_incidence` rather than recomputed, so it agrees by construction
        with both the scorer's exclusive truth and the incidence head's own target. Cells no
        target particle touched stay False for everyone.

        Args:
            incidence: [num_particles, num_calohits] energy fractions, columns summing to 1
                wherever any target deposited.

        Returns:
            [num_particles, num_calohits] bool, at most one True per column.
        """
        exclusive = torch.zeros(incidence.shape, dtype=torch.bool)
        if incidence.numel() == 0:
            return exclusive
        owned = incidence.max(dim=0).values > 0
        if not bool(owned.any()):
            return exclusive
        owner = incidence.argmax(dim=0)
        cols = torch.nonzero(owned, as_tuple=True)[0]
        exclusive[owner[cols], cols] = True
        return exclusive

    def _build_particle_calohit_incidence(self, particle_ids: torch.Tensor, calohits, num_calohits: int) -> torch.Tensor:
        """Dense incidence matrix [n_particles, n_calohits] of energy fractions.

        Entry [a, i] is I_ia = E_ia / E_i, the fraction of calo hit i's energy deposited by particle
        a, i.e. each hit's column sums to 1 over the particles that produced it. This is the GLOW /
        HGPflow target: it is energy-conserving by construction and, unlike the binary
        particle_calohit_valid mask, it can represent a cell shared between several particles
        (~11% of hits here), which a mutually-exclusive mask target cannot.

        Hits with no contribution from any target particle (noise, or particles cut by the target
        selection) are left as an all-zero column. That is deliberate: the KL loss is
        `-true * log(pred)`, so a zero column contributes exactly zero loss and those hits are simply
        ignored by this task rather than being forced onto a dummy particle. Rejecting them remains
        the job of the mask and object-classification tasks.
        """
        num_particles = int(particle_ids.numel())
        incidence = torch.zeros((num_particles, num_calohits), dtype=torch.float32)
        if num_particles == 0 or num_calohits == 0:
            return incidence

        contrib_counts = ak.to_numpy(ak.num(calohits["contrib_particle_ids"], axis=1))
        if contrib_counts.sum() == 0:
            return incidence

        flat_contrib_particle_ids = self._map_contrib_particle_ids(ak.to_numpy(ak.flatten(calohits["contrib_particle_ids"], axis=1)))
        flat_contrib_energies = ak.to_numpy(ak.flatten(calohits["contrib_energies"], axis=1)).astype(np.float32, copy=False)
        calohit_indices = np.repeat(np.arange(num_calohits, dtype=np.int64), contrib_counts)

        row_indices, matched = self._lookup_row_indices(particle_ids.cpu().numpy(), flat_contrib_particle_ids)
        if row_indices.size == 0:
            return incidence

        # accumulate=True because a particle can deposit in the same cell via several contributions
        incidence.index_put_(
            (torch.from_numpy(row_indices), torch.from_numpy(calohit_indices[matched])),
            torch.from_numpy(flat_contrib_energies[matched]),
            accumulate=True,
        )

        # Normalise per hit. clamp_min leaves all-zero (noise) columns at zero rather than dividing by 0.
        incidence /= incidence.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return incidence

    @staticmethod
    def _num_calohits(calohits: ak.Record) -> int:
        return int(ak.to_numpy(calohits["total_energy"]).shape[0])

    @staticmethod
    def _subset_calohit_record(calohits: ak.Record, keep: np.ndarray) -> ak.Record:
        """Keep only the selected calo hits, filtering every per-hit field of the record.

        Event-level scalars (anything whose length is not the hit count) are passed through.
        """
        if bool(keep.all()):
            return calohits

        num_hits = int(keep.shape[0])
        keep_ak = ak.Array(keep)
        new_fields = {}
        for field in calohits.fields:
            arr = calohits[field]
            try:
                is_per_hit = len(arr) == num_hits
            except TypeError:
                is_per_hit = False
            new_fields[field] = arr[keep_ak] if is_per_hit else arr
        return ak.Record(new_fields)

    @staticmethod
    def _calohit_on_valid_particle_from_csr(
        indptr: torch.Tensor,
        indices: torch.Tensor,
        particle_valid: torch.Tensor,
        num_hits: int,
    ) -> torch.Tensor:
        """Per-hit "did a valid target particle deposit here?", straight from the sparse CSR.

        Equivalent to `dense_mask[particle_valid].any(0)` but without materialising the dense
        [particles x hits] mask, so it can be used to decide which hits to keep *before* anything
        of that size is built.
        """
        on_valid = torch.zeros(num_hits, dtype=torch.bool)
        if indices.numel():
            rows = torch.repeat_interleave(torch.arange(indptr.numel() - 1), torch.diff(indptr))
            on_valid[indices[particle_valid[rows]]] = True
        return on_valid

    def _calohit_keep_mask(self, calohits: ak.Record, targets: dict[str, torch.Tensor], num_hits: int) -> torch.Tensor | None:
        """Which calo hits survive the hit-filter stage, or None when no filtering is configured.

        Also stashes the raw probabilities in `self._last_filter_probs` so they can be handed to the
        clustering model as an input feature (see hit_filter_as_feature).
        """
        self._last_filter_probs = None

        if self.hit_eval_path is not None:
            probs = self._read_hit_filter_probs(targets["sample_id"], num_hits)
            self._last_filter_probs = probs
            return probs >= self.hit_filter_threshold

        if self.calohit_truth_filter:
            indptr, indices, _ = self._build_particle_calohit_csr(targets["particle_particle_id"], calohits)
            return self._calohit_on_valid_particle_from_csr(indptr, indices, targets["particle_valid"], num_hits)

        return None

    def _build_track_sihit_csr(self, track_hit_ids, num_sihits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tracks = len(track_hit_ids)
        if num_tracks == 0 or num_sihits == 0:
            return self._empty_csr_components(num_tracks, num_sihits)

        row_counts = ak.to_numpy(ak.num(track_hit_ids, axis=1))
        if row_counts.sum() == 0:
            return self._empty_csr_components(num_tracks, num_sihits)

        row_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), row_counts)
        col_indices = ak.to_numpy(ak.flatten(track_hit_ids, axis=1)).astype(np.int64, copy=False)

        valid = (col_indices >= 0) & (col_indices < num_sihits)
        row_indices = row_indices[valid]
        col_indices = col_indices[valid]
        return self._build_csr_components(num_tracks, num_sihits, row_indices, col_indices)

    def _build_particle_targets(self, particles: ak.Record) -> dict[str, torch.Tensor]:
        targets = self._record_to_tensors(
            particles,
            "particle",
            int_fields={"particle_id"},
            skip_fields={"parent_id"},
            default_dtype=torch.float32,
        )
        targets["particle_event_id"] = targets["particle_event_id"].expand_as(targets["particle_particle_id"])

        pt = torch.sqrt(targets["particle_px"] ** 2 + targets["particle_py"] ** 2)
        p = torch.sqrt(targets["particle_px"] ** 2 + targets["particle_py"] ** 2 + targets["particle_pz"] ** 2).clamp_min(1e-12)

        targets["particle_pt"] = pt
        targets["particle_p"] = p
        targets["particle_qopt"] = targets["particle_charge"] / pt.clamp_min(1e-6)
        targets["particle_eta"] = torch.arctanh(targets["particle_pz"] / p)
        targets["particle_theta"] = torch.arccos((targets["particle_pz"] / p).clamp(-1.0, 1.0))
        targets["particle_phi"] = torch.arctan2(targets["particle_py"], targets["particle_px"])
        targets["particle_d0"] = (-targets["particle_vx"] * targets["particle_py"] + targets["particle_vy"] * targets["particle_px"]) / pt.clamp_min(
            1e-6
        )
        targets["particle_z0"] = targets["particle_vz"]
        targets["particle_charged"] = targets["particle_charge"] != 0
        targets["particle_neutral"] = ~targets["particle_charged"]
        self._add_particle_classes_inplace(targets)
        return targets

    @staticmethod
    def _add_particle_classes_inplace(targets: dict[str, torch.Tensor]) -> None:
        pdg_key = "particle_pdg_id" if "particle_pdg_id" in targets else "particle_pdgid"
        pdg_id = targets[pdg_key].to(torch.int64)
        abs_pdg_id = torch.abs(pdg_id)

        class_masks: dict[str, torch.Tensor] = {
            "is_photon": abs_pdg_id == 22,
            "is_electron": abs_pdg_id == 11,
            "is_muon": abs_pdg_id == 13,
            "is_tau": abs_pdg_id == 15,
            "is_neutrino": (abs_pdg_id == 12) | (abs_pdg_id == 14) | (abs_pdg_id == 16) | (abs_pdg_id == 18),
        }

        is_known_nonhadron = torch.zeros_like(abs_pdg_id, dtype=torch.bool)
        for mask in class_masks.values():
            is_known_nonhadron |= mask

        class_masks["is_charged_hadron"] = (~is_known_nonhadron) & targets["particle_charged"]
        class_masks["is_neutral_hadron"] = (~is_known_nonhadron) & targets["particle_neutral"]

        class_assignment = {
            "is_neutral_hadron": 0,
            "is_charged_hadron": 1,
            "is_photon": 3,
            "is_electron": 4,
            "is_muon": 5,
            "is_tau": 6,
            "is_neutrino": 7,
        }
        class_id = torch.full_like(abs_pdg_id, -1, dtype=torch.int64)
        for class_name, class_value in class_assignment.items():
            class_id[class_masks[class_name]] = class_value

        is_other = torch.ones_like(abs_pdg_id, dtype=torch.bool)
        for class_name in class_assignment:
            is_other &= ~class_masks[class_name]
        class_masks["is_other"] = is_other

        targets["particle_class"] = class_id
        for class_name, class_mask in class_masks.items():
            targets[f"particle_{class_name}"] = class_mask

    def _build_particle_kinematic_mask(self, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        particle_valid = targets["particle_pt"] >= self.particle_min_pt
        particle_valid = particle_valid & (torch.abs(targets["particle_eta"]) <= self.particle_max_abs_eta)

        if not self.particle_include_neutral:
            particle_valid = particle_valid & (~targets["particle_neutral"])
        if not self.particle_include_charged:
            particle_valid = particle_valid & (~targets["particle_charged"])
        return particle_valid

    def _build_particle_hit_cut_mask(
        self,
        targets: dict[str, torch.Tensor],
        particle_num_sihits: torch.Tensor,
        particle_num_calo_hit_fields: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        particle_valid = torch.ones_like(targets["particle_valid"], dtype=torch.bool)
        num_ecalhits = (
            particle_num_calo_hit_fields["particle_num_ecalhits"]
            if particle_num_calo_hit_fields is not None
            else torch.zeros_like(particle_num_sihits)
        )
        num_hcalhits = (
            particle_num_calo_hit_fields["particle_num_hcalhits"]
            if particle_num_calo_hit_fields is not None
            else torch.zeros_like(particle_num_sihits)
        )

        for class_mask_key, cuts in self.particle_hit_cuts.items():
            class_mask = targets[class_mask_key].to(torch.bool)
            class_valid = torch.ones_like(class_mask)
            for cut_key, counts in (
                ("min_num_sihit", particle_num_sihits),
                ("min_num_ecal", num_ecalhits),
                ("min_num_hcal", num_hcalhits),
            ):
                threshold = cuts[cut_key]
                if threshold > 0:
                    class_valid &= counts >= threshold

            particle_valid &= (~class_mask) | class_valid

        return particle_valid

    def _build_particle_sihit_fields(
        self,
        particle_ids: torch.Tensor,
        sihit_particle_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indptr, indices, shape = self._build_particle_sihit_csr(
            particle_ids,
            sihit_particle_ids,
        )
        num_sihits = torch.diff(indptr).to(torch.float32)
        return indptr, indices, shape, num_sihits

    def _build_sihit_inputs(self, sihits: ak.Record) -> dict[str, torch.Tensor]:
        inputs = self._record_to_tensors(
            sihits,
            "sihit",
            int_fields={"particle_id"},
            default_dtype=torch.float32,
        )
        self._scale_xyz_inplace(inputs, "sihit", scale=1e-3)
        self._add_cylindrical_coords_inplace(inputs, "sihit")
        inputs["sihit_valid"] = self._valid_mask_like(inputs["sihit_x"])
        return inputs

    def _read_calohits(self, particle_path: Path, event_idx: int) -> ak.Record:
        t_calo_read = perf_counter()
        calohits_path = self._get_collection_path("calo_hits", particle_path)
        calohits = self._read_event_from_file(calohits_path, event_idx)
        # Zero-suppression is applied here, at the single point where calo hits enter the pipeline,
        # so every downstream consumer sees the same hits. This matters for the
        # particle_min_num_calohits cut in load_event: if the cut counted hits before suppression,
        # particles could pass "has >=1 calo hit" and then be left with an all-zero truth mask
        # (measured at 8.6% of targets with a 1e-3 threshold) - unreconstructable by construction,
        # but still counted in the efficiency denominator.
        calohits = self._apply_calohit_energy_cut(calohits)
        # Region cut after the energy cut: both are pure subsets of the cell list, so the order
        # does not change the result, and both must precede the particle_min_num_calohits filter
        # so that "how many cells does this particle leave" is counted over cells that SURVIVE.
        calohits = self._apply_calohit_eta_cut(calohits)
        self._debug(f"read calohits in {perf_counter() - t_calo_read:.3f}s")
        return calohits

    def _apply_calohit_energy_cut(self, calohits: ak.Record) -> ak.Record:
        """Drop calo cells below calohit_min_energy (mostly noise). No-op when the threshold is 0."""
        if self.calohit_min_energy <= 0.0:
            return calohits

        energy = ak.to_numpy(calohits["total_energy"]).astype(np.float32, copy=False)
        return self._subset_calohit_record(calohits, energy >= self.calohit_min_energy)

    def _apply_calohit_eta_cut(self, calohits: ak.Record) -> ak.Record:
        """Keep only calo cells within |eta|, i.e. restrict the event to a detector region.

        No-op when the threshold is 0, which is what pu0 runs with.

        This exists because pileup-200 does not fit otherwise. A pu200 event carries ~532k cells
        against ~22k at pu0, and MaskFormer's memory goes as num_queries x num_hits, so a faithful
        pu200 event is ~200x the pu0 footprint on a card that already OOMs at 4x.

        Cutting in eta is the one way of shrinking that which does not distort the task. Raising
        calohit_min_energy removes cells but leaves the target particles untouched, so the fraction
        of cells owned by a target collapses (measured: 2.5% against pu0's ~37%) and the mask loss
        is then minimised by predicting empty everywhere -- which is exactly what happened, a run
        with flat validation loss for seven epochs. Cutting in eta removes cells and targets
        together, so the ratio is preserved: measured 38.1% owned at |eta| < 0.88, with 15.4 hits
        per target against pu0's 13.

        0.88 is not arbitrary -- it is where the barrel ends. The HCAL barrel reaches r = 3441 with
        |z| <= 3450, so a track steeper than eta = 0.883 leaves through the barrel end before the
        outer radius and deposits the rest of its shower in the endcap. Cutting there means every
        particle in the sample is fully contained, rather than the model being asked to reconstruct
        showers whose energy is partly in cells that were removed. The endcaps start well outside
        it (ece at eta ~ 1.55, hce at ~ 1.24), so nothing straddles the boundary.

        Set data.particle_max_abs_eta to the same value: this cut alone would leave target
        particles whose cells have been removed.
        """
        if self.calohit_max_abs_eta <= 0.0:
            return calohits

        x = ak.to_numpy(calohits["x"]).astype(np.float32, copy=False)
        y = ak.to_numpy(calohits["y"]).astype(np.float32, copy=False)
        z = ak.to_numpy(calohits["z"]).astype(np.float32, copy=False)
        # arctanh(z / |r|), matching how the cell eta feature itself is built above.
        norm = np.sqrt(x * x + y * y + z * z)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.arctanh(np.clip(np.divide(z, np.where(norm == 0.0, 1.0, norm)), -0.999999, 0.999999))
        return self._subset_calohit_record(calohits, np.abs(eta) <= self.calohit_max_abs_eta)

    def _build_particle_num_calo_hit_fields(self, particle_ids: torch.Tensor, calohits: ak.Record) -> dict[str, torch.Tensor]:
        particle_calohit_indptr, particle_calohit_indices, _ = self._build_particle_calohit_csr(particle_ids, calohits)
        num_calohits = torch.diff(particle_calohit_indptr).to(torch.float32)
        num_particles = int(num_calohits.numel())
        if num_particles == 0 or particle_calohit_indices.numel() == 0:
            empty_counts = {
                f"particle_num_{group_name}": torch.zeros(num_particles, dtype=torch.float32) for group_name in self.CALO_GROUP_DETECTOR_IDS
            }
            return {
                "particle_num_calohits": num_calohits,
                **empty_counts,
            }

        indptr_np = particle_calohit_indptr.cpu().numpy().astype(np.int64, copy=False)
        indices_np = particle_calohit_indices.cpu().numpy().astype(np.int64, copy=False)
        row_indices = np.repeat(np.arange(num_particles, dtype=np.int64), np.diff(indptr_np))

        linked_detector_ids = self._get_calohit_detector_ids(calohits)[indices_np]
        grouped_counts = {
            f"particle_num_{group_name}": torch.from_numpy(
                self._float32_bincount(
                    row_indices[np.isin(linked_detector_ids, detector_ids)],
                    minlength=num_particles,
                )
            )
            for group_name, detector_ids in self.CALO_GROUP_DETECTOR_IDS.items()
        }

        return {
            "particle_num_calohits": num_calohits,
            **grouped_counts,
        }

    def _build_particle_num_calo_hit_fields_if_available(
        self,
        particle_ids: torch.Tensor,
        calohits: ak.Record | None,
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
        if calohits is None:
            return None, None

        fields = self._build_particle_num_calo_hit_fields(particle_ids, calohits)
        return fields, fields["particle_num_calohits"]

    def _build_particle_calo_energy_fields(self, particle_ids: torch.Tensor, calohits: ak.Record) -> dict[str, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        particle_energy_raw_np = {key: np.zeros(num_particles, dtype=np.float32) for key in (*self.CALO_SUBSYSTEMS, "calo_sum")}

        contrib_counts = ak.to_numpy(ak.num(calohits["contrib_particle_ids"], axis=1))
        if num_particles > 0 and contrib_counts.sum() > 0:
            flat_contrib_particle_ids = self._map_contrib_particle_ids(ak.to_numpy(ak.flatten(calohits["contrib_particle_ids"], axis=1)))
            flat_contrib_energies = ak.to_numpy(ak.flatten(calohits["contrib_energies"], axis=1)).astype(np.float32, copy=False)
            flat_contrib_detector = np.repeat(self._get_calohit_detector_ids(calohits), contrib_counts)

            particle_ids_np = particle_ids.cpu().numpy()
            row_indices, matched = self._lookup_row_indices(particle_ids_np, flat_contrib_particle_ids)
            if row_indices.size > 0:
                matched_energies = flat_contrib_energies[matched]
                matched_detectors = flat_contrib_detector[matched]
                particle_energy_raw_np["calo_sum"] = self._float32_bincount(
                    row_indices,
                    minlength=num_particles,
                    weights=matched_energies,
                )
                for subsystem, detector_ids in self.CALO_SUBSYSTEM_DETECTOR_IDS.items():
                    subsystem_mask = np.isin(matched_detectors, detector_ids)
                    if not np.any(subsystem_mask):
                        continue

                    particle_energy_raw_np[subsystem] = self._float32_bincount(
                        row_indices[subsystem_mask],
                        minlength=num_particles,
                        weights=matched_energies[subsystem_mask],
                    )

        particle_energy_raw = {key: torch.from_numpy(values) for key, values in particle_energy_raw_np.items()}
        particle_energy_raw["ecal"] = particle_energy_raw["ecb"] + particle_energy_raw["ece"]
        particle_energy_raw["hcal"] = particle_energy_raw["hcb"] + particle_energy_raw["hce"]

        particle_energy_calib = {
            f"{subsystem}_calib": particle_energy_raw[subsystem] * scale for subsystem, scale in self.CALO_SUBSYSTEM_CALIBRATION.items()
        }
        particle_energy_calib["ecal_calib"] = particle_energy_calib["ecb_calib"] + particle_energy_calib["ece_calib"]
        particle_energy_calib["hcal_calib"] = particle_energy_calib["hcb_calib"] + particle_energy_calib["hce_calib"]
        particle_energy_calib["calo_calib"] = particle_energy_calib["ecal_calib"] + particle_energy_calib["hcal_calib"]

        out = {f"particle_energy_{subsystem}": particle_energy_raw[subsystem] for subsystem in (*self.CALO_SUBSYSTEMS, "ecal", "hcal")}
        out["particle_energy_calo_sum"] = particle_energy_raw["calo_sum"]
        out.update({f"particle_energy_{name}": values for name, values in particle_energy_calib.items()})
        return out

    @staticmethod
    def _csr_to_dense_bool(indptr: torch.Tensor, indices: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = int(shape[0]), int(shape[1])
        dense = torch.zeros((n_rows, n_cols), dtype=torch.bool)
        rows = torch.repeat_interleave(torch.arange(n_rows), torch.diff(indptr))
        if indices.numel():
            dense[rows, indices] = True
        return dense

    @staticmethod
    def _build_calohit_contrib_energy_sum(calohits: ak.Record) -> torch.Tensor:
        contrib_energy_sum = ak.to_numpy(ak.sum(calohits["contrib_energies"], axis=1, mask_identity=False)).astype(np.float32, copy=False)
        return torch.from_numpy(contrib_energy_sum)

    def _add_calohits(
        self,
        inputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        particle_path: Path,
        event_idx: int,
        calohits: ak.Record | None = None,
    ) -> None:
        # Note: zero-suppression is already applied inside _read_calohits, so `calohits` here is
        # always the suppressed set regardless of which path supplied it.
        if calohits is None:
            calohits = self._read_calohits(particle_path, event_idx)

        # Particle-level truth quantities are built from the FULL hit set, before any hit filtering,
        # so that "how many hits did this particle deposit" and "how much calo energy did it leave"
        # stay properties of the particle rather than of whichever filter happens to be configured.
        # They are the denominators our efficiency metrics are quoted against, so they must not move
        # when the filter changes.
        targets.update(self._build_particle_num_calo_hit_fields(targets["particle_particle_id"], calohits))
        targets.update(self._build_particle_calo_energy_fields(targets["particle_particle_id"], calohits))

        # Hit filtering happens HERE, before the per-hit tensors and the dense [particles x hits]
        # targets are built, so that everything downstream is constructed at the reduced size.
        # Filtering afterwards gives identical results but first materialises the full-size incidence
        # matrix (1000 x ~22000 float32 = 88 MB per event), which dominated both the loader time and
        # the per-worker memory footprint — it is why the truth-filter diagnostic measured SLOWER
        # than the unfiltered run despite processing 2.9x fewer hits.
        #
        # Note this differs from trackml, which filters before applying its particle cuts, so
        # particles whose hits the filter destroys silently leave the target set. Here the particle
        # cuts have already been applied in load_event, so the set of target particles — and hence
        # the efficiency denominator — is the same whether or not a filter is in use. That keeps
        # end-to-end efficiency honest: hits the filter wrongly deletes count against us.
        keep = self._calohit_keep_mask(calohits, targets, self._num_calohits(calohits))
        filter_probs = self._last_filter_probs
        if keep is not None:
            calohits = self._subset_calohit_record(calohits, keep.numpy())
            if filter_probs is not None:
                filter_probs = filter_probs[keep]

        inputs.update(
            self._record_to_tensors(
                calohits,
                "calohit",
                skip_fields={"detector", "contrib_particle_ids", "contrib_energies", "contrib_times"},
                default_dtype=torch.float32,
            )
        )
        inputs["calohit_detector"] = ak.to_torch(calohits["detector"]).to(torch.int64)
        inputs["calohit_contrib_energy_sum"] = self._build_calohit_contrib_energy_sum(calohits)
        self._scale_xyz_inplace(inputs, "calohit", scale=1e-3)
        self._add_cylindrical_coords_inplace(inputs, "calohit")
        # Calo energies span several orders of magnitude, so provide a log feature as well.
        if "calohit_total_energy" in inputs:
            inputs["calohit_log_energy"] = torch.log(inputs["calohit_total_energy"].clamp_min(1e-8))
        if self.hit_filter_as_feature:
            if filter_probs is None:
                msg = "hit_filter_as_feature requires hit_eval_path — there are no filter scores without it."
                raise ValueError(msg)
            inputs["calohit_filter_prob"] = filter_probs
        inputs["calohit_valid"] = self._valid_mask_like(inputs["calohit_x"])
        targets["calohit_valid"] = inputs["calohit_valid"]

        if not self.build_calohit_associations:
            self._debug("skipping calo association build (build_calohit_associations=False)")
            return

        self._debug(f"building calo associations: n_particles={targets['particle_particle_id'].size(0)} n_calohits={inputs['calohit_x'].size(0)}")
        t_calo_assoc = perf_counter()
        particle_calohit_indptr, particle_calohit_indices, particle_calohit_shape = self._build_particle_calohit_csr(
            targets["particle_particle_id"],
            calohits,
        )
        self._debug(
            f"built calo CSR in {perf_counter() - t_calo_assoc:.3f}s "
            f"shape=({int(particle_calohit_shape[0])}, {int(particle_calohit_shape[1])}) nnz={particle_calohit_indices.numel()}"
        )

        targets["particle_calohit_indptr"] = particle_calohit_indptr
        targets["particle_calohit_indices"] = particle_calohit_indices
        targets["particle_calohit_shape"] = particle_calohit_shape

        # Soft, energy-weighted ownership for the IncidenceRegressionTask. The key must be
        # exactly "particle_incidence": the task reads targets[f"{target_object}_incidence"].
        # Built before the mask below because the exclusive variant of the mask is derived
        # from it, which is what keeps the two targets consistent with each other and with
        # the scorer's definition of exclusive truth.
        targets["particle_incidence"] = self._build_particle_calohit_incidence(
            targets["particle_particle_id"], calohits, int(inputs["calohit_valid"].size(0))
        )

        # Dense truth mask [n_particles, n_calohits] for the ObjectHitMaskTask. Multi-owner by
        # default: a calo hit is True for every particle it contributed energy to. With
        # particle_calohit_exclusive, reduced to one owner per cell to match how the result is
        # scored -- see the constructor for why that mismatch was worth removing.
        if self.particle_calohit_exclusive:
            targets["particle_calohit_valid"] = self._exclusive_particle_calohit_valid(targets["particle_incidence"])
        else:
            targets["particle_calohit_valid"] = self._csr_to_dense_bool(
                particle_calohit_indptr, particle_calohit_indices, particle_calohit_shape
            )

        # Per-cell weight for the mask loss. Always present, so a config can switch the
        # weighting on without the loader having to change and ObjectHitMaskTask never has to
        # test for the key; it is all ones when disabled or when there is no cell energy to
        # weight by. Calibrated first, because ECAL and HCAL differ by ~20% and the weight is
        # meant to track the energy the metric actually sums.
        num_calohits = int(inputs["calohit_valid"].size(0))
        if self.calohit_loss_weight_power > 0.0 and "calohit_total_energy" in inputs:
            calibration = torch.ones(num_calohits, dtype=torch.float32)
            detector = inputs["calohit_detector"].cpu().numpy()
            for subsystem, detector_ids in self.CALO_SUBSYSTEM_DETECTOR_IDS.items():
                selected = torch.from_numpy(np.isin(detector, detector_ids))
                calibration[selected] = self.CALO_SUBSYSTEM_CALIBRATION[subsystem]
            targets["calohit_loss_weight"] = self._build_calohit_loss_weight(inputs["calohit_total_energy"] * calibration)
        else:
            targets["calohit_loss_weight"] = torch.ones(num_calohits, dtype=torch.float32)

        # Per-hit seed target for dynamic query initialisation: True for the single cell in which
        # each valid particle deposited the most energy — its shower core. This is the calorimeter
        # analogue of trackml's `is_first` (the first hit of a track): the decoder's `query_init`
        # task learns to spot these, and queries are initialised from those hits' embeddings rather
        # than from random learnable vectors, so each query starts already localised on a shower.
        # Uses the particle's OWN deposit E_ia = incidence * cell energy, not the raw cell energy,
        # so a particle whose largest cell is dominated by a neighbour still seeds on its own core.
        seed = torch.zeros(int(inputs["calohit_valid"].size(0)), dtype=torch.bool)
        if "calohit_total_energy" in inputs and bool(targets["particle_valid"].any()):
            deposits = targets["particle_incidence"] * inputs["calohit_total_energy"].unsqueeze(0)
            rows = torch.where(targets["particle_valid"] & (deposits.sum(-1) > 0))[0]
            if rows.numel():
                seed[deposits[rows].argmax(dim=-1)] = True
        targets["calohit_is_seed"] = seed

        # Per-hit target used to TRAIN the hit filter: True if the hit received energy from at least
        # one VALID target particle. ~37% of hits qualify; the other 63% are noise as far as this
        # task is concerned (in practice, deposits from particles cut by the target selection rather
        # than detector noise). Computed on whatever hits remain, so when a filter is already in use
        # this describes the surviving hits.
        particle_valid = targets["particle_valid"]
        targets["calohit_on_valid_particle"] = (
            targets["particle_calohit_valid"][particle_valid].any(0)
            if bool(particle_valid.any())
            else torch.zeros(int(inputs["calohit_valid"].size(0)), dtype=torch.bool)
        )

    def _read_hit_filter_probs(self, sample_id, num_hits: int) -> torch.Tensor:
        """Load the trained hit filter's per-hit probability for this event.

        Raises:
            KeyError: If the event is missing from the hit eval file.
            ValueError: If the file's hit count disagrees with this event's, which means the filter
                was run with a different calo-hit selection.
        """
        key = str(int(sample_id))
        with h5py.File(self.hit_eval_path, "r") as f:
            if key not in f:
                msg = f"sample_id {key} not found in hit eval file {self.hit_eval_path}"
                raise KeyError(msg)
            probs = f[f"{key}/preds/final/hit_filter/calohit_on_valid_particle_prob"][0]

        # The filter must have been run with the same calohit_min_energy, otherwise its per-hit
        # predictions do not line up with the hits we just built. Catch that here rather than
        # silently filtering with a shifted mapping.
        if probs.shape[0] != num_hits:
            msg = (
                f"hit eval for sample {key} has {probs.shape[0]} hits but this event has {num_hits}. "
                "The filter and clustering configs must use identical calo-hit selection "
                "(calohit_min_energy in particular)."
            )
            raise ValueError(msg)

        return torch.from_numpy(probs.astype(np.float32, copy=False))

    def _add_tracks(self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], particle_path: Path, event_idx: int) -> None:
        t_track_read = perf_counter()
        tracks_path = self._get_collection_path("tracks", particle_path)
        tracks = self._read_event_from_file(tracks_path, event_idx)
        self._debug(f"read tracks in {perf_counter() - t_track_read:.3f}s")

        targets.update(
            self._record_to_tensors(
                tracks,
                "track",
                skip_fields={"hit_ids"},
                default_dtype=None,
            )
        )
        targets["track_valid"] = self._valid_mask_like(targets["track_phi"])

        t_track_assoc = perf_counter()
        track_sihit_indptr, track_sihit_indices, track_sihit_shape = self._build_track_sihit_csr(
            tracks["hit_ids"],
            num_sihits=len(inputs["sihit_valid"]),
        )
        self._debug(
            f"built track_sihit CSR in {perf_counter() - t_track_assoc:.3f}s "
            f"shape=({int(track_sihit_shape[0])}, {int(track_sihit_shape[1])}) nnz={track_sihit_indices.numel()}"
        )
        targets["track_sihit_indptr"] = track_sihit_indptr
        targets["track_sihit_indices"] = track_sihit_indices
        targets["track_sihit_shape"] = track_sihit_shape

    def _pad_particle_targets_inplace(self, targets: dict[str, torch.Tensor]) -> None:
        # Dense per-particle targets, including the dense particle_calohit_valid mask
        # ([n_particles, n_calohits]). CSR association keys (*_indptr/_indices/_shape) are
        # sparse and must not be padded along the particle dimension.
        particle_keys = [
            k
            for k in targets
            if k.startswith("particle_") and not k.startswith("particle_sihit_") and not k.endswith(("_indptr", "_indices", "_shape"))
        ]
        if not particle_keys:
            return

        # MaskFormer requires the target object dimension to equal num_queries, so pad (or truncate)
        # to a fixed size when event_max_num_particles is set. Otherwise fall back to the per-event
        # max (single-batch convenience; not compatible with the matching loss).
        if self.event_max_num_particles is not None:
            target_size = self.event_max_num_particles
        else:
            target_size = max(int(targets[k].size(0)) for k in particle_keys)

        for key in particle_keys:
            values = targets[key]
            current = int(values.size(0))
            if current == target_size:
                continue
            if current > target_size:
                targets[key] = values[:target_size]
                continue
            pad_shape = (target_size - current, *values.shape[1:])
            pad_value = False if values.dtype is torch.bool else 0
            targets[key] = torch.cat([values, values.new_full(pad_shape, pad_value)], dim=0)

    def load_event(self, sample_id):
        t0 = perf_counter()
        if sample_id < 0 or sample_id >= len(self.sample_index):
            msg = f"sample_id {sample_id} is out of range [0, {len(self.sample_index) - 1}]"
            raise ValueError(msg)

        shard_idx, event_idx = self.sample_index[sample_id]
        particle_path = self.particle_shard_paths[shard_idx]
        self._debug(f"sample_id={sample_id} shard_idx={shard_idx} event_idx={event_idx} file={particle_path.name}")

        # Read in the data. Tracker hits are absent in calo-only datasets (e.g. pu0).
        t_read = perf_counter()
        particles = self._read_event_from_file(particle_path, event_idx)
        sihits = None
        if not self.calo_only:
            sihit_path = self._get_collection_path("tracker_hits", particle_path)
            sihits = self._read_event_from_file(sihit_path, event_idx)
        self._debug(f"read particles+sihits in {perf_counter() - t_read:.3f}s")

        # Collapse shower secondaries onto the particle that entered the calorimeter. This must run
        # BEFORE any target is built: it sets the contrib-id remap that every calo association
        # below goes through, so the merged target inherits its whole shower's cells, energy and
        # cell count and is then cut on those. Doing it after the cuts would judge a fragment on
        # its own deposit and only then merge it.
        entry_ancestors = None
        if self.particle_collapse_shower_secondaries:
            entry_ancestors = self._build_calo_entry_ancestors(particles)
        else:
            self._contrib_id_remap = None

        targets = self._build_particle_targets(particles)

        if entry_ancestors is not None:
            # Keep only the particles that entered the calorimeter. Everything else has had its
            # deposits re-pointed at one of these, so dropping it loses no calorimeter energy --
            # it stops being a target, not a contributor.
            is_entry_particle = torch.from_numpy(entry_ancestors == np.arange(entry_ancestors.size))
            self._apply_mask_to_prefixed_keys(targets, "particle_", is_entry_particle)
            self._debug(f"after shower collapse: n_particles={targets['particle_particle_id'].size(0)}")

        particle_valid = self._build_particle_kinematic_mask(targets)
        self._apply_mask_to_prefixed_keys(targets, "particle_", particle_valid)
        self._debug(f"after kinematic cuts: n_particles={targets['particle_particle_id'].size(0)}")

        targets["particle_valid"] = self._valid_mask_like(targets["particle_pt"])

        inputs: dict[str, torch.Tensor] = {}
        t_assoc = perf_counter()
        if not self.calo_only:
            inputs = self._build_sihit_inputs(sihits)
            targets["sihit_valid"] = inputs["sihit_valid"]
            _, _, _, particle_num_sihits = self._build_particle_sihit_fields(
                targets["particle_particle_id"],
                inputs["sihit_particle_id"],
            )
        else:
            # No silicon hits: sihit-based hit cuts are effectively disabled.
            particle_num_sihits = torch.zeros_like(targets["particle_particle_id"], dtype=torch.float32)

        calohits = self._read_calohits(particle_path, event_idx) if self.return_calohits or self._requires_calohits_for_hit_cuts else None
        particle_num_calo_hit_fields, particle_num_calohits = self._build_particle_num_calo_hit_fields_if_available(
            targets["particle_particle_id"],
            calohits,
        )

        # Apply constituent based particle cuts
        particle_valid = self._build_particle_hit_cut_mask(
            targets,
            particle_num_sihits,
            particle_num_calo_hit_fields,
        )

        # Optionally require particles to deposit a minimum number of calo hits to be a target.
        if self.particle_min_num_calohits > 0 and particle_num_calohits is not None:
            particle_valid = particle_valid & (particle_num_calohits >= self.particle_min_num_calohits)

        self._apply_mask_to_prefixed_keys(
            targets,
            "particle_",
            particle_valid,
            skip_prefixes=("particle_sihit_", "particle_calohit_"),
        )
        if not self.calo_only:
            particle_sihit_indptr, particle_sihit_indices, particle_sihit_shape, particle_num_sihits = self._build_particle_sihit_fields(
                targets["particle_particle_id"],
                inputs["sihit_particle_id"],
            )
        particle_num_calo_hit_fields, particle_num_calohits = self._build_particle_num_calo_hit_fields_if_available(
            targets["particle_particle_id"],
            calohits,
        )
        if particle_num_calo_hit_fields is not None:
            targets.update(particle_num_calo_hit_fields)

        n_sihits = 0 if self.calo_only else inputs["sihit_particle_id"].size(0)
        constituent_debug = f"after constituent cuts: n_particles={targets['particle_particle_id'].size(0)} n_sihits={n_sihits}"
        if particle_num_calohits is not None:
            constituent_debug += f" n_calohits={int(particle_num_calohits.sum().item())}"
            if particle_num_calo_hit_fields is not None:
                constituent_debug += (
                    f" n_ecalhits={int(particle_num_calo_hit_fields['particle_num_ecalhits'].sum().item())}"
                    f" n_hcalhits={int(particle_num_calo_hit_fields['particle_num_hcalhits'].sum().item())}"
                )
        self._debug(constituent_debug)

        if not self.calo_only:
            self._debug(
                f"built particle_sihit CSR in {perf_counter() - t_assoc:.3f}s "
                f"shape=({int(particle_sihit_shape[0])}, {int(particle_sihit_shape[1])}) nnz={particle_sihit_indices.numel()}"
            )
            targets["particle_sihit_indptr"] = particle_sihit_indptr
            targets["particle_sihit_indices"] = particle_sihit_indices
            targets["particle_sihit_shape"] = particle_sihit_shape
            targets["particle_num_sihits"] = particle_num_sihits

        # Set before _add_calohits: the hit-eval lookup keys the h5 by sample_id.
        targets["sample_id"] = torch.tensor(sample_id, dtype=torch.int64)

        # Return the calorimeter hit info if requested
        if self.return_calohits:
            self._add_calohits(inputs, targets, particle_path, event_idx, calohits=calohits)

        # Return ACTS track info if requested
        if self.return_tracks:
            self._add_tracks(inputs, targets, particle_path, event_idx)
        self._debug(f"load_event finished in {perf_counter() - t0:.3f}s")

        return inputs, targets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())

        if idx < 0:
            idx += len(self.sample_ids)
        if idx < 0 or idx >= len(self.sample_ids):
            msg = f"Dataset index {idx} out of range for length {len(self.sample_ids)}"
            raise IndexError(msg)

        sample_id = self.sample_ids[idx]
        inputs, targets = self.load_event(sample_id)

        self._pad_particle_targets_inplace(targets)

        # Add leading batch dimension. Samples are collated into real batches by
        # collate_calohit_batch when the datamodule is configured with batch_size > 1.
        inputs_out = {k: v.unsqueeze(0) for k, v in inputs.items()}
        targets_out = {k: v.unsqueeze(0) for k, v in targets.items()}

        return inputs_out, targets_out


def _pad_last_dim(tensor: torch.Tensor, size: int, value) -> torch.Tensor:
    """Right-pad the last dimension of `tensor` up to `size` with `value`."""
    if tensor.shape[-1] >= size:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[-1] = size - tensor.shape[-1]
    return torch.cat([tensor, tensor.new_full(pad_shape, value)], dim=-1)


# CSR association fields have a per-event length and are only ever consumed inside this module
# (they are the intermediate used to build the dense particle_calohit_valid mask), so they are
# dropped when batching rather than padded to a meaningless common length.
_CSR_KEYS = ("particle_calohit_indptr", "particle_calohit_indices", "particle_calohit_shape")


def collate_calohit_batch(
    batch: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    sort_field: str = "phi",
    sort_pad_value: float = 100.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collate single events into a batch, padding the variable-length calo-hit dimension.

    Events differ a lot in hit count (~6k to ~45k), so every `calohit_*` field is right-padded to
    the largest event in the batch, as is the trailing hit axis of the dense
    `particle_calohit_valid` mask. `calohit_valid` marks the padding so the model masks it out.

    The sort field is padded with a large sentinel rather than 0. The encoder sorts constituents by
    this field before windowed attention, so it decides which hits share a window; padding it with 0
    would drop padded tokens into the middle of the physical range (phi and eta are both centred on
    0) and fragment the attention windows of the smaller events in the batch. A value beyond the
    physical range keeps all padding sorted to the end, where it is harmlessly masked.

    `sort_field` MUST match the model's `input_sort_field`, which is why the datamodule takes it as
    a parameter rather than relying on this default.
    """
    inputs_list = [sample[0] for sample in batch]
    targets_list = [sample[1] for sample in batch]

    if len(batch) == 1:
        return inputs_list[0], targets_list[0]

    max_hits = max(inp["calohit_valid"].shape[-1] for inp in inputs_list)

    def collate_dict(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key in dicts[0]:
            if key in _CSR_KEYS:
                continue

            tensors = []
            for sample, sample_inputs in zip(dicts, inputs_list, strict=True):
                tensor = sample[key]
                num_hits = sample_inputs["calohit_valid"].shape[-1]

                if key in ("particle_calohit_valid", "particle_incidence"):
                    # (1, num_particles, num_hits): pad the trailing hit axis. Zero/False padding is
                    # correct for both — a padded hit belongs to no particle and carries no energy.
                    tensor = _pad_last_dim(tensor, max_hits, False if tensor.dtype is torch.bool else 0)
                elif key.startswith("calohit_") and tensor.dim() == 2 and tensor.shape[-1] == num_hits:
                    # Per-hit field. Guarded on the calohit_ prefix so fixed-size particle fields
                    # are never mistaken for hit fields if their lengths happen to coincide.
                    pad_value = sort_pad_value if key == f"calohit_{sort_field}" else 0
                    tensor = _pad_last_dim(tensor, max_hits, pad_value)

                tensors.append(tensor)

            out[key] = torch.cat(tensors, dim=0)
        return out

    return collate_dict(inputs_list), collate_dict(targets_list)


class ColliderMLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        train_start_event: int = 0,
        val_start_event: int = 0,
        test_start_event: int = 0,
        batch_size: int = 1,
        input_sort_field: str = "phi",
        hit_filter_as_feature: bool = False,
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        # Must match the model's input_sort_field: the collate function pads this field with an
        # out-of-range sentinel so padded tokens sort to the end of the encoder's attention windows.
        self.input_sort_field = input_sort_field
        # Passed through to every split's dataset; see ColliderMLDataset for why this exists.
        self.hit_filter_as_feature = hit_filter_as_feature
        # Per-split hit-filter predictions from a trained filter (stage 1). Each split needs its own
        # file because the filter is run separately over each event window.
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        # Starting event offset for each split. When train/val/test share a directory these must be
        # set so the windows do not overlap, otherwise validation/test metrics leak training events.
        # e.g. train_start_event=0, val_start_event=num_train, test_start_event=num_train+num_val.
        self.train_start_event = train_start_event
        self.val_start_event = val_start_event
        self.test_start_event = test_start_event
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dset = ColliderMLDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                start_event=self.train_start_event,
                hit_eval_path=self.hit_eval_train,
                hit_filter_as_feature=self.hit_filter_as_feature,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dset = ColliderMLDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                start_event=self.val_start_event,
                hit_eval_path=self.hit_eval_val,
                hit_filter_as_feature=self.hit_filter_as_feature,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        # `self.trainer` is None when the datamodule is used outside a Trainer, which analysis
        # scripts do (eval/dump.py builds it just to get a dataloader).
        if stage == "fit" and (self.trainer is None or self.trainer.is_global_zero):
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ColliderMLDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                start_event=self.test_start_event,
                hit_eval_path=self.hit_eval_test,
                hit_filter_as_feature=self.hit_filter_as_feature,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: ColliderMLDataset, shuffle: bool):
        # batch_size=1 keeps the original path (dataset items already carry a leading batch dim of
        # 1, so no collation is needed). For larger batches we hand PyTorch real batches and pad the
        # variable-length hit dimension in collate_calohit_batch.
        if self.batch_size == 1:
            return DataLoader(
                dataset=dataset,
                batch_size=None,
                collate_fn=None,
                sampler=None,
                num_workers=self.num_workers,
                shuffle=shuffle,
                pin_memory=self.pin_memory,
            )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=partial(collate_calohit_batch, sort_field=self.input_sort_field),
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
