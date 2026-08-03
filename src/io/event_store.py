"""Reading the event store: the cells, truth and model predictions both methods are scored on.

This is a hand-written mirror of the writer in
``hepattn/experiments/colliderml/eval/format.py``, deliberately not an import of it. The
comparison is only worth anything if the scoring code can be read, checked and re-run by
someone who has neither hepattn nor a GPU nor the multi-terabyte dataset, so the dependency
stops here: numpy and the standard library, nothing else.

The store is also where "both algorithms saw the same cells" stops being a promise and
becomes a fact. The cells in these files are the ones the network was given, after its own
zero-suppression; CLUE clusters those same arrays rather than re-deriving a hit set from the
raw files, so the two cannot drift apart.

Nothing physical is hardcoded here. The sampling calibrations, the detector-id groups, the
layer geometry and the selection cuts all travel inside the store as metadata, and
:class:`EventStore` checks them against what the experiment config expects rather than
restating them.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_FORMAT_VERSIONS = frozenset({1})

# Mirrors format.py. Only used to decode; the authoritative values travel in the metadata
# and are cross-checked in _check_encoding.
LOGIT_MIN = -8.0
LOGIT_MAX = 8.0
LOGIT_LEVELS = 256

# Metadata the thesis config has an opinion about. Any disagreement is an error, because it
# means the store was produced under different definitions from the ones being reported.
CONTRACT_KEYS: tuple[tuple[str, ...], ...] = (
    ("units", "length"),
    ("units", "energy"),
    ("hit_selection", "calohit_min_energy"),
    ("particle_selection", "particle_min_pt"),
    ("particle_selection", "particle_max_abs_eta"),
    ("particle_selection", "particle_min_num_calohits"),
    ("detector", "subsystem_order"),
    ("detector", "subsystem_calibration"),
)


class EventStoreError(Exception):
    """Base class for problems with an event store."""


class EventStoreVersionError(EventStoreError):
    """The store was written in a format this reader does not understand."""


class EventStoreMismatchError(EventStoreError):
    """The store disagrees with itself, or with what the caller expected."""


def logit_code_for_threshold(probability: float) -> int:
    """Smallest stored uint8 code whose probability is at least ``probability``.

    Working points are applied as an integer comparison against the stored codes, so no
    float round-trip is involved and the test is exact up to the half-step quantisation of
    +/-0.031 in logit.
    """
    if not 0.0 < probability < 1.0:
        msg = f"threshold must be in (0, 1), got {probability}"
        raise ValueError(msg)
    logit = float(np.log(probability / (1.0 - probability)))
    return int(np.ceil((logit - LOGIT_MIN) / (LOGIT_MAX - LOGIT_MIN) * (LOGIT_LEVELS - 1)))


def probability_for_logit_code(code: np.ndarray) -> np.ndarray:
    """Decode stored uint8 codes back to mask probabilities.

    The inverse of :func:`logit_code_for_threshold`, and exact only up to the +/-0.031 logit
    half-step of the quantisation. That is fine for a weight -- it is a soft share of a cell's
    energy, not a threshold test -- but it is why working points are still applied as integer
    comparisons against the codes rather than against decoded floats.
    """
    logit = LOGIT_MIN + np.asarray(code, dtype=np.float64) / (LOGIT_LEVELS - 1) * (LOGIT_MAX - LOGIT_MIN)
    return 1.0 / (1.0 + np.exp(-logit))


def _dig(meta: Mapping, path: Sequence[str]):
    node = meta
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


@dataclass(frozen=True)
class EventRecord:
    """One event: its cells, the truth partition over them, and the model's masks."""

    sample_id: int

    # Cells, in the store's geometry order.
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    energy: np.ndarray
    detector: np.ndarray
    subsystem: np.ndarray
    layer: np.ndarray

    # Exclusive truth: index into the particle table, -1 where no target particle deposited.
    truth_label: np.ndarray

    # Multi-owner truth, particle-major CSR over cells.
    truth_indptr: np.ndarray
    truth_indices: np.ndarray
    truth_incidence: np.ndarray

    # Truth particles.
    particle_id: np.ndarray
    particle_px: np.ndarray
    particle_py: np.ndarray
    particle_pz: np.ndarray
    particle_energy: np.ndarray
    particle_pt: np.ndarray
    particle_eta: np.ndarray
    particle_phi: np.ndarray
    particle_pdg_id: np.ndarray
    particle_class: np.ndarray
    particle_num_calohits: np.ndarray
    particle_energy_calo_sum: np.ndarray

    # MaskFormer masks, query-major CSR over cells, stored above a loose threshold.
    mf_query_index: np.ndarray
    mf_valid_prob: np.ndarray
    mf_indptr: np.ndarray
    mf_indices: np.ndarray
    mf_logit_u8: np.ndarray

    n_hits: int
    n_particles: int
    n_particles_untruncated: int
    truncated: bool
    event_energy_raw: float
    event_energy_calib: float
    event_energy_on_target_calib: float

    calibration: np.ndarray

    @property
    def energy_calib(self) -> np.ndarray:
        """Cell energy with its subsystem's sampling calibration applied.

        Not optional and not a constant factor: ECAL and HCAL are calibrated differently
        (37.5/38.7 against 45.0/46.9), so a cluster's effective calibration depends on how
        its energy is split between them and does not cancel in a ratio.
        """
        return self.energy * self.calibration[self.subsystem]

    @property
    def truth_deposit(self) -> np.ndarray:
        """Per cell, the energy its owning particle put there; 0 where no target owns it.

        This is ``E_ia``, the quantity the energy-weighted efficiency is built from. It is
        recovered by walking the multi-owner CSR and keeping the entries whose particle is
        also the cell's exclusive owner.
        """
        deposit = np.zeros(self.n_hits, dtype=np.float64)
        if self.n_particles == 0 or self.truth_indices.size == 0:
            return deposit
        rows = np.repeat(np.arange(self.n_particles), np.diff(self.truth_indptr))
        cols = self.truth_indices
        winner = self.truth_label[cols] == rows
        deposit[cols[winner]] = self.truth_incidence[winner] * self.energy[cols[winner]]
        return deposit

    def eta(self) -> np.ndarray:
        s = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        return np.arctanh(np.clip(self.z / np.maximum(s, 1e-12), -1 + 1e-12, 1 - 1e-12))

    def phi(self) -> np.ndarray:
        return np.arctan2(self.y, self.x)

    def r(self) -> np.ndarray:
        return np.hypot(self.x, self.y)

    def maskformer_labels(
        self,
        mask_threshold: float,
        object_threshold: float,
        min_cluster_hits: int = 1,
    ) -> tuple[np.ndarray, int]:
        """Exclusive cell -> cluster labels at an arbitrary working point.

        The store keeps the masks sparsely, above a loose threshold, so any working point at
        or above that floor can be re-derived here without touching a GPU. This is what lets
        the comparison be shown as a curve over working points rather than a single point
        that depends on one tuning choice.

        Where several accepted queries claim the same cell, the highest mask logit wins:
        MaskFormer's masks may overlap but CLUE's cannot, so the head-to-head is run on an
        exclusive partition for both. Overlap is reported separately, as a capability.

        Args:
            mask_threshold: minimum mask probability for a cell to join a cluster.
            object_threshold: minimum object-head probability for a query to count at all.
            min_cluster_hits: clusters smaller than this are dropped.

        Returns:
            ``(label, n_clusters)`` with ``label`` an int32 per cell, -1 where unclaimed,
            and cluster ids compacted to ``0..n_clusters-1``.
        """
        code = logit_code_for_threshold(mask_threshold)
        n_q = int(self.mf_valid_prob.size)
        label = np.full(self.n_hits, -1, dtype=np.int32)
        if n_q == 0 or self.mf_indices.size == 0:
            return label, 0

        rows = np.repeat(np.arange(n_q), np.diff(self.mf_indptr))
        keep = (self.mf_logit_u8 >= code) & (self.mf_valid_prob[rows] >= object_threshold)
        if not keep.any():
            return label, 0

        rows = rows[keep]
        cols = self.mf_indices[keep]
        codes = self.mf_logit_u8[keep]

        # Resolve overlapping claims: sort by (cell, code) and let the last entry per cell
        # win, which is the highest-logit claim.
        order = np.lexsort((codes, cols))
        rows, cols = rows[order], cols[order]
        last = np.empty(cols.size, dtype=bool)
        last[-1] = True
        last[:-1] = cols[1:] != cols[:-1]
        rows, cols = rows[last], cols[last]

        if min_cluster_hits > 1:
            counts = np.bincount(rows, minlength=n_q)
            big = counts[rows] >= min_cluster_hits
            rows, cols = rows[big], cols[big]
            if rows.size == 0:
                return label, 0

        used, compact = np.unique(rows, return_inverse=True)
        label[cols] = compact.astype(np.int32)
        return label, int(used.size)

    def maskformer_soft_masks(
        self,
        mask_threshold: float,
        object_threshold: float,
        min_cluster_hits: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Overlapping masks as fractional claims on each cell's energy.

        :meth:`maskformer_labels` throws this away. It resolves every contested cell to a
        single highest-logit winner, because CLUE cannot represent a shared cell and the
        head-to-head has to be run on something both methods can express. That is right for
        the head-to-head and wrong as the last word: dividing a shared cell is the capability
        the architecture has and the baseline does not, and collapsing it means the comparison
        never measures the thing the model was built to do.

        Here a cell contested by several accepted queries is *divided* between them in
        proportion to their mask probabilities, rather than awarded to one. The weights are
        normalised per cell, so a method whose masks never overlap -- CLUE -- comes through
        this function with every weight equal to 1 and is neither helped nor penalised by the
        change. That is the property that makes the resulting metric a fair one to compare on
        rather than a MaskFormer-only diagnostic.

        Args:
            mask_threshold: minimum mask probability for a cell to enter a cluster's claim.
            object_threshold: minimum object-head probability for a query to count at all.
            min_cluster_hits: clusters holding fewer cells than this are dropped.

        Returns:
            ``(cluster, cell, weight, n_clusters)`` as parallel arrays, with `weight` summing
            to 1 over the clusters claiming any given cell.
        """
        code = logit_code_for_threshold(mask_threshold)
        n_q = int(self.mf_valid_prob.size)
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), 0)
        if n_q == 0 or self.mf_indices.size == 0:
            return empty

        rows = np.repeat(np.arange(n_q), np.diff(self.mf_indptr))
        keep = (self.mf_logit_u8 >= code) & (self.mf_valid_prob[rows] >= object_threshold)
        if not keep.any():
            return empty

        rows = rows[keep]
        cols = self.mf_indices[keep].astype(np.int64)
        weight = probability_for_logit_code(self.mf_logit_u8[keep])

        if min_cluster_hits > 1:
            counts = np.bincount(rows, minlength=n_q)
            big = counts[rows] >= min_cluster_hits
            rows, cols, weight = rows[big], cols[big], weight[big]
            if rows.size == 0:
                return empty

        # Normalise each cell's claims to sum to one, so the division is a partition of that
        # cell's energy and total predicted energy stays equal to total clustered energy.
        per_cell = np.bincount(cols, weights=weight, minlength=self.n_hits)
        weight = weight / np.maximum(per_cell[cols], 1e-30)

        used, compact = np.unique(rows, return_inverse=True)
        return compact.astype(np.int64), cols, weight, int(used.size)


class EventStore:
    """A directory of chunk files, presented as a sequence of events."""

    def __init__(self, root: Path | str, expect: Mapping | None = None, strict: bool = True):
        """Open a store and validate it.

        Args:
            root: directory holding ``chunk_*.npz``.
            expect: optional mapping of the experiment config's expectations, checked
                against the store's metadata. Keys are the leaf names of ``CONTRACT_KEYS``.
            strict: when False, a contract disagreement is a warning rather than an error.
                Only reasonable when deliberately re-reading an old store to reproduce an
                old figure.
        """
        self.root = Path(root)
        self.chunks = sorted(self.root.glob("chunk_*.npz"))
        if not self.chunks:
            msg = f"no chunk_*.npz found in {self.root}"
            raise EventStoreError(msg)

        self._handles: dict[Path, np.lib.npyio.NpzFile] = {}
        self.meta = self._load_and_check_metadata()
        self._check_encoding()

        order = self.meta["detector"]["subsystem_order"]
        table = self.meta["detector"]["subsystem_calibration"]
        self.calibration = np.array([table[name] for name in order], dtype=np.float32)
        self.subsystem_code = {name: i for i, name in enumerate(order)}

        self._index: list[tuple[Path, int]] = []
        for path in self.chunks:
            for sample_id in json.loads(str(self._handle(path)["meta_json"]))["chunk"]["sample_ids"]:
                self._index.append((path, int(sample_id)))

        if expect is not None:
            self._check_contract(expect, strict=strict)

    def _handle(self, path: Path) -> np.lib.npyio.NpzFile:
        if path not in self._handles:
            self._handles[path] = np.load(path)
        return self._handles[path]

    def _load_and_check_metadata(self) -> dict:
        """Every chunk must agree outside its own chunk block."""
        reference: dict | None = None
        for path in self.chunks:
            meta = json.loads(str(self._handle(path)["meta_json"]))
            version = meta.get("format_version")
            if version not in SUPPORTED_FORMAT_VERSIONS:
                msg = (
                    f"{path.name} is format version {version}; this reader supports "
                    f"{sorted(SUPPORTED_FORMAT_VERSIONS)}. Regenerate the store, or update "
                    f"src/io/event_store.py to match hepattn's eval/format.py."
                )
                raise EventStoreVersionError(msg)

            body = {k: v for k, v in meta.items() if k not in {"chunk", "created_utc"}}
            if reference is None:
                reference = body
            elif body != reference:
                differing = sorted(k for k in set(body) | set(reference) if body.get(k) != reference.get(k))
                msg = (
                    f"{path.name} was written under different settings from {self.chunks[0].name} "
                    f"(differing blocks: {differing}). The store was regenerated in pieces; "
                    f"rebuild it in one go."
                )
                raise EventStoreMismatchError(msg)
        assert reference is not None
        return reference

    def _check_encoding(self) -> None:
        encoding = _dig(self.meta, ("encoding", "prob_encoding")) or {}
        expected = {"kind": "logit_uint8", "logit_min": LOGIT_MIN, "logit_max": LOGIT_MAX, "levels": LOGIT_LEVELS}
        differing = {k: (encoding.get(k), v) for k, v in expected.items() if encoding.get(k) != v}
        if differing:
            lines = "\n".join(f"  {k}: store={got!r} reader={want!r}" for k, (got, want) in differing.items())
            msg = f"mask probability encoding differs from this reader's:\n{lines}"
            raise EventStoreMismatchError(msg)

    def _check_contract(self, expect: Mapping, strict: bool) -> None:
        """Compare the store's definitions against the config's, reporting all differences at once."""
        problems = []
        for path in CONTRACT_KEYS:
            leaf = path[-1]
            if leaf not in expect:
                continue
            found = _dig(self.meta, path)
            wanted = expect[leaf]
            if isinstance(found, float) or isinstance(wanted, float):
                same = found is not None and np.isclose(float(found), float(wanted))
            else:
                same = found == wanted
            if not same:
                problems.append(f"  {'.'.join(path)}: store={found!r} config={wanted!r}")

        window = _dig(self.meta, ("event_window",)) or {}
        trained = _dig(self.meta, ("maskformer", "trained_event_window"))
        if trained and window:
            start, num = int(window.get("start_event", 0)), int(window.get("num_events", 0))
            if start < int(trained[1]) and start + num > int(trained[0]):
                problems.append(
                    f"  event_window [{start}, {start + num}) overlaps the checkpoint's training window "
                    f"[{trained[0]}, {trained[1]}) -- the model has seen these events"
                )

        if problems:
            body = "\n".join(problems)
            msg = f"event store at {self.root} disagrees with the experiment config:\n{body}"
            if strict:
                raise EventStoreMismatchError(msg)
            import warnings

            warnings.warn(msg, stacklevel=2)

    @property
    def sample_ids(self) -> list[int]:
        return [sample_id for _, sample_id in self._index]

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[EventRecord]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> EventRecord:
        path, sample_id = self._index[index]
        data = self._handle(path)

        def get(name: str) -> np.ndarray:
            return data[f"e{sample_id:06d}__{name}"]

        return EventRecord(
            sample_id=sample_id,
            x=get("cell_x"),
            y=get("cell_y"),
            z=get("cell_z"),
            energy=get("cell_energy"),
            detector=get("cell_detector"),
            subsystem=get("cell_subsystem"),
            layer=get("cell_layer"),
            truth_label=get("cell_truth_label"),
            truth_indptr=get("truth_indptr"),
            truth_indices=get("truth_indices"),
            truth_incidence=get("truth_incidence"),
            particle_id=get("particle_id"),
            particle_px=get("particle_px"),
            particle_py=get("particle_py"),
            particle_pz=get("particle_pz"),
            particle_energy=get("particle_energy"),
            particle_pt=get("particle_pt"),
            particle_eta=get("particle_eta"),
            particle_phi=get("particle_phi"),
            particle_pdg_id=get("particle_pdg_id"),
            particle_class=get("particle_class"),
            particle_num_calohits=get("particle_num_calohits"),
            particle_energy_calo_sum=get("particle_energy_calo_sum"),
            mf_query_index=get("mf_query_index"),
            mf_valid_prob=get("mf_valid_prob"),
            mf_indptr=get("mf_indptr"),
            mf_indices=get("mf_indices"),
            mf_logit_u8=get("mf_logit_u8"),
            n_hits=int(get("n_hits")),
            n_particles=int(get("n_particles")),
            n_particles_untruncated=int(get("n_particles_untruncated")),
            truncated=bool(get("truncated")),
            event_energy_raw=float(get("event_energy_raw")),
            event_energy_calib=float(get("event_energy_calib")),
            event_energy_on_target_calib=float(get("event_energy_on_target_calib")),
            calibration=self.calibration,
        )
