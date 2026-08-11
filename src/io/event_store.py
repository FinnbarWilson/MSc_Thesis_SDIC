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

A note on the MaskFormer output, since :class:`EventRecord` offers four ways to turn it into
clusters and the choice is a physics decision rather than a detail. The model has two heads.
The **mask** head scores each (query, cell) pair independently through a sigmoid: a detection
score, with nothing in its loss relating one cell's claims to each other. The **incidence**
head softmaxes over queries within a cell and is trained against ``I_ia = E_ia / E_i``: a
share of that cell's energy. Both an exclusive labelling and a fractional one can be built
from either head, which is the 2x2 the methods below cover. The incidence-based ones need a
format-2 store.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Version 2 adds the incidence head (`mf_incidence_*`). Version 1 stores are still readable
# -- everything that was in them still means the same thing -- but they carry masks only, so
# the incidence-based labellings below raise on one rather than silently falling back to the
# mask head and reporting the difference between two methods as zero.
SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2})

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
    # The truth DEFINITION, not a cut on it: whether Geant secondaries made inside the
    # calorimeter were merged onto the particle that entered it. The three cuts above are
    # identical under both definitions while the target set differs threefold, so without this
    # key a store dumped under one would validate against a config describing the other.
    ("particle_selection", "particle_collapse_shower_secondaries"),
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

    # MaskFormer incidence head, cell-major `[n_hits, k]`, descending in share. Width 0 for a
    # format-1 store or a checkpoint without the head. The query axis indexes the same kept
    # queries as `mf_valid_prob`; -1 is padding.
    mf_incidence_query: np.ndarray
    mf_incidence_share: np.ndarray

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

    @property
    def has_incidence(self) -> bool:
        """Whether this store carries the incidence head at all."""
        return bool(self.mf_incidence_query.size) and self.mf_incidence_query.shape[1] > 0

    def _require_incidence(self) -> None:
        if not self.has_incidence:
            msg = (
                "this event store has no incidence head. It is either format version 1 or was "
                "dumped from a checkpoint without an IncidenceRegressionTask. Re-dump with a "
                "current hepattn eval/dump.py to score the incidence-based labellings."
            )
            raise EventStoreMismatchError(msg)

    def _claimed_cells(self, mask_threshold: float, object_threshold: float) -> np.ndarray:
        """Cells claimed by at least one accepted query, by the mask head.

        This is the DETECTION half of the two heads, factored out because the incidence
        labellings reuse it unchanged. Keeping detection identical is what makes them a
        controlled comparison against :meth:`maskformer_labels`: total claimed energy is the
        same, so any difference in the metrics is the assignment rule and nothing else.
        """
        code = logit_code_for_threshold(mask_threshold)
        claimed = np.zeros(self.n_hits, dtype=bool)
        n_q = int(self.mf_valid_prob.size)
        if n_q == 0 or self.mf_indices.size == 0:
            return claimed
        rows = np.repeat(np.arange(n_q), np.diff(self.mf_indptr))
        keep = (self.mf_logit_u8 >= code) & (self.mf_valid_prob[rows] >= object_threshold)
        claimed[self.mf_indices[keep]] = True
        return claimed

    def maskformer_incidence_labels(
        self,
        mask_threshold: float,
        object_threshold: float,
        min_cluster_hits: int = 1,
        incidence_floor: float = 0.0,
        restrict_to_mask: bool = False,
    ) -> tuple[np.ndarray, int]:
        """Exclusive labels using the incidence head to decide ownership.

        :meth:`maskformer_labels` resolves a contested cell by the highest mask logit. That
        rule reads the wrong quantity. The mask head emits an independent sigmoid per
        (query, cell) and nothing in its loss makes one cell's claims sum to anything, so a
        mask probability is a detection score and not a share of a cell. The incidence head
        softmaxes over queries and is trained by KL divergence against ``I_ia = E_ia / E_i``,
        which is exactly "how much of this cell is that particle's".

        The split of labour here follows that difference, and it is why this is not a step
        towards hard borders. Detection stays with the mask head, unchanged, so the same
        cells are claimed and the same 63% of sub-threshold deposits are declined -- the
        incidence softmax sums to one over queries for *every* cell and on its own could
        never reject anything. Only the question "whose is it" moves, to the head that was
        taught to answer it in fractions.

        Args:
            mask_threshold: minimum mask probability for a cell to be claimed at all.
            object_threshold: minimum object-head probability for a query to count.
            min_cluster_hits: clusters smaller than this are dropped.
            incidence_floor: a cell whose best share is below this is left unclaimed. 0
                assigns every detected cell, which is the setting comparable to CLUE.
            restrict_to_mask: when True the winner must also be a query whose *mask* claims
                the cell. A control rather than a working point: it separates "the incidence
                head assigns detected cells better" from "the incidence head reaches cells the
                mask head gave to nobody".

        Returns:
            ``(label, n_clusters)`` as in :meth:`maskformer_labels`.
        """
        self._require_incidence()
        label = np.full(self.n_hits, -1, dtype=np.int32)
        claimed = self._claimed_cells(mask_threshold, object_threshold)
        if not claimed.any():
            return label, 0

        query = self.mf_incidence_query.astype(np.int64)
        share = self.mf_incidence_share.astype(np.float64)

        # Disqualify padding and queries the object head rejects, then optionally anything the
        # mask head did not put on this cell. A disqualified entry gets share -1 so it can
        # never win an argmax, which keeps this branch-free over ~24k cells.
        eligible = query >= 0
        eligible &= self.mf_valid_prob[np.clip(query, 0, None)] >= object_threshold
        if restrict_to_mask:
            eligible &= self._mask_claims(mask_threshold, object_threshold)[
                np.clip(query, 0, None), np.arange(self.n_hits)[:, None]
            ]

        share = np.where(eligible, share, -1.0)
        best = share.argmax(axis=1)
        rows = np.arange(self.n_hits)
        winner = query[rows, best]
        winning_share = share[rows, best]

        take = claimed & (winning_share >= max(incidence_floor, 0.0)) & (winning_share >= 0.0) & (winner >= 0)
        if not take.any():
            return label, 0

        cols = np.flatnonzero(take)
        rows = winner[cols]

        if min_cluster_hits > 1:
            counts = np.bincount(rows, minlength=int(self.mf_valid_prob.size))
            big = counts[rows] >= min_cluster_hits
            rows, cols = rows[big], cols[big]
            if rows.size == 0:
                return label, 0

        used, compact = np.unique(rows, return_inverse=True)
        label[cols] = compact.astype(np.int32)
        return label, int(used.size)

    def _mask_claims(self, mask_threshold: float, object_threshold: float) -> np.ndarray:
        """Dense `[n_queries, n_hits]` boolean of which accepted query claims which cell.

        Only built for ``restrict_to_mask``, which is a diagnostic path; the normal ones stay
        sparse. At ~600 kept queries by ~24k cells this is a 14 MB bool array per event.
        """
        n_q = int(self.mf_valid_prob.size)
        dense = np.zeros((max(n_q, 1), self.n_hits), dtype=bool)
        if n_q == 0 or self.mf_indices.size == 0:
            return dense
        code = logit_code_for_threshold(mask_threshold)
        rows = np.repeat(np.arange(n_q), np.diff(self.mf_indptr))
        keep = (self.mf_logit_u8 >= code) & (self.mf_valid_prob[rows] >= object_threshold)
        dense[rows[keep], self.mf_indices[keep]] = True
        return dense

    def maskformer_incidence_soft_masks(
        self,
        mask_threshold: float,
        object_threshold: float,
        min_cluster_hits: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Fractional claims on each cell, weighted by the incidence head.

        The counterpart of :meth:`maskformer_soft_masks`, and the one the capability study
        should be reading. That method divides a contested cell in proportion to *mask
        probabilities*, which measures the model's overlap handling with a quantity never
        trained to divide a cell -- the measured symptom being that it splits each cell 2.04
        ways against truth's 1.22, an over-division that survives every mask threshold up to
        0.95. Incidence shares are trained against the true energy fractions, so this asks
        the capability question with the calibrated quantity.

        Detection is again the mask head's, so the set of claimed cells matches the rest of
        the MaskFormer numbers and only the division changes.

        Returns:
            ``(cluster, cell, weight, n_clusters)``, weights summing to 1 over the clusters
            claiming any given cell.
        """
        self._require_incidence()
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), 0)
        claimed = self._claimed_cells(mask_threshold, object_threshold)
        if not claimed.any():
            return empty

        query = self.mf_incidence_query.astype(np.int64)
        share = self.mf_incidence_share.astype(np.float64)
        eligible = (query >= 0) & (self.mf_valid_prob[np.clip(query, 0, None)] >= object_threshold)
        eligible &= claimed[:, None]

        cells, slots = np.nonzero(eligible)
        if cells.size == 0:
            return empty
        rows = query[cells, slots]
        weight = share[cells, slots]

        if min_cluster_hits > 1:
            counts = np.bincount(rows, minlength=int(self.mf_valid_prob.size))
            big = counts[rows] >= min_cluster_hits
            rows, cells, weight = rows[big], cells[big], weight[big]
            if rows.size == 0:
                return empty

        # Renormalise per cell. The store deliberately keeps the raw softmax rather than a
        # normalised top-k, so the division by the row sum happens here, where k is known.
        per_cell = np.bincount(cells, weights=weight, minlength=self.n_hits)
        weight = weight / np.maximum(per_cell[cells], 1e-30)

        used, compact = np.unique(rows, return_inverse=True)
        return compact.astype(np.int64), cells.astype(np.int64), weight, int(used.size)

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

        def get_optional(name: str, dtype: np.dtype, width: int = 0) -> np.ndarray:
            """A format-2 array, or an empty stand-in when reading a format-1 store."""
            key = f"e{sample_id:06d}__{name}"
            if key in data:
                return data[key]
            return np.empty((int(get("n_hits")), width), dtype=dtype)

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
            mf_incidence_query=get_optional("mf_incidence_query", np.dtype(np.int16)),
            mf_incidence_share=get_optional("mf_incidence_share", np.dtype(np.float16)),
            n_hits=int(get("n_hits")),
            n_particles=int(get("n_particles")),
            n_particles_untruncated=int(get("n_particles_untruncated")),
            truncated=bool(get("truncated")),
            event_energy_raw=float(get("event_energy_raw")),
            event_energy_calib=float(get("event_energy_calib")),
            event_energy_on_target_calib=float(get("event_energy_on_target_calib")),
            calibration=self.calibration,
        )
