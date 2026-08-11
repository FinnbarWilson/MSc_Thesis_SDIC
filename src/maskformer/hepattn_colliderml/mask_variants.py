"""Three targeted changes to the mask head, each switchable on its own.

All three subclass `ObjectHitMaskTask` and live in THIS repository rather than in a patch against
hepattn. `ConstituentAffinityTask` established that a task named by `class_path` is reached through
the same sync as every other file here, and overriding `forward()` and `loss()` bypasses hepattn's
loss-name registry entirely -- so a new objective needs no new entry in `loss_fns` and no fourth
entry in `hepattn-changes.patch`.

**Every switch defaults to off, and with all of them off this class is exactly
`ObjectHitMaskTask`.** That is what makes each overlay a single-variable change.

WHAT EACH ONE ATTACKS, and the measurement behind it
----------------------------------------------------

1. `coverage_weight` -- FRAGMENTATION AND THE THRESHOLD MISMATCH.
   DICE is normalised by object size, so covering a 6-cell fragment perfectly scores exactly as
   well as covering a 38-cell shower perfectly. Nothing in the objective prefers one correct large
   cluster to five correct small ones, and the model emits 753 clusters per event for 538 true
   particles, with a high-energy particle's two largest fragments holding 36% and 16.5% of its
   energy. Separately, the reported metric is "at least half of a particle's energy in one cluster",
   and at E > 20 GeV MaskFormer already recovers a HIGHER mean energy fraction than CLUE (0.269 vs
   0.259) while scoring far worse on the threshold (0.150 vs 0.224) -- it spreads partial credit
   where the metric rewards commitment. A coverage term is the same quantity as the metric, used as
   a loss: for each matched target, penalise the energy fraction its query failed to claim.

2. `bce_pos_weight` -- THE 1:1700 CLASS IMBALANCE.
   Each query sees ~22,000 cells of which ~13 are positive. Plain BCE at equal weight is minimised
   by claiming almost nothing, and the model duly gives under 2% probability to a third of a
   high-energy shower's energy -- confident exclusion, not hedging, which is why no working point
   recovers it. hepattn's `mask_bce_loss` has no `pos_weight`, so this reimplements the BCE term
   with one. Recall is bought at the cost of precision; that is the intended trade and it must be
   read on the efficiency-purity curve rather than on efficiency alone.

3. `propagation_lambda` -- THE ONE THE WHOLE INVESTIGATION POINTS AT.
   The mask logit is a dot product between a query and a single cell, so a cell can only ask "do I
   look like this query" and never "am I attached to a cell that does". That is why the matched
   cluster holds ~6 cells whether the particle deposits 13 or 38, spans ~0.06 in angle and ~1 cm of
   a 42 cm shower, and why six previous interventions moved none of it. This adds one message-
   passing step over the cell neighbour graph before thresholding, which converts the question into
   "do I, or does my neighbourhood". It is the smallest change that gives the head a relational
   term at all.

   Implemented as a sparse matrix product rather than a gather. Gathering k neighbours of every
   (query, cell) pair would materialise [queries x cells x k] = 1000 x 22000 x 8 floats, about
   700 MB in fp32 on top of a run that already peaks at 49 GB. `logits @ A` with a sparse normalised
   adjacency costs the same memory as the logits themselves.
"""

import torch
from torch import Tensor

from hepattn.models.loss import loss_fns
from hepattn.models.task import ObjectHitMaskTask


def _knn_graph(coords: Tensor, k: int, chunk: int = 2048) -> tuple[Tensor, Tensor]:
    """k nearest neighbours of every cell, computed in chunks.

    Chunked because the full [N, N] distance matrix at N ~ 22,000 is 1.9 GB in fp32, and this runs
    inside the training step. Chunks of 2048 rows keep it at ~180 MB.
    """
    n = coords.shape[0]
    idx_out = torch.empty((n, k), dtype=torch.long, device=coords.device)
    dist_out = torch.empty((n, k), dtype=coords.dtype, device=coords.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(coords[start:stop], coords)
        # Exclude self by pushing the diagonal out of reach.
        rows = torch.arange(start, stop, device=coords.device)
        d[rows - start, rows] = float("inf")
        dd, ii = torch.topk(d, min(k, n - 1), dim=1, largest=False)
        idx_out[start:stop, : ii.shape[1]] = ii
        dist_out[start:stop, : dd.shape[1]] = dd
    return dist_out, idx_out


class ExtendedObjectHitMaskTask(ObjectHitMaskTask):
    """`ObjectHitMaskTask` with coverage, recall weighting and neighbour propagation, each optional."""

    def __init__(
        self,
        *args,
        coverage_weight: float = 0.0,
        coverage_energy_field: str | None = None,
        bce_pos_weight: float = 0.0,
        propagation_lambda: float = 0.0,
        propagation_neighbours: int = 8,
        propagation_radius: float = 0.06,
        **kwargs,
    ):
        """
        Args:
            coverage_weight: weight on the per-target energy-coverage penalty. 0 disables it.
            coverage_energy_field: per-cell target field to weight coverage by, e.g.
                ``calohit_loss_weight``. None weights every cell equally, which measures cell
                coverage rather than energy coverage -- energy is what the metric uses.
            bce_pos_weight: `pos_weight` for a re-implemented BCE term. 0 disables it and leaves
                whatever is in `losses` untouched.
            propagation_lambda: strength of the neighbour message-passing on mask logits. 0
                disables it and the forward pass is the parent's exactly.
            propagation_neighbours: k for the neighbour graph.
            propagation_radius: metres; neighbours beyond this are dropped, so the graph reflects
                adjacency rather than merely rank.
        """
        super().__init__(*args, **kwargs)
        # (source coordinate tensor, adjacency) for the most recent event. The decoder calls this
        # task's forward ONCE PER DECODER LAYER, and the neighbour graph depends only on cell
        # coordinates -- not on the query embeddings that change between layers -- so without this
        # the kNN is rebuilt 4x per step for an identical result. Measured at 182 ms a build on a
        # 54k-cell barrel event, i.e. 727 ms/step of pure waste against a ~2300 ms step.
        #
        # Keyed by tensor IDENTITY (`is`), deliberately, not by data_ptr: the caching allocator
        # reuses addresses, so a freed event's pointer can reappear on a later event and a
        # ptr-keyed cache would silently propagate over the wrong graph. Holding the reference also
        # keeps that tensor alive, which is what makes the identity check sound. It is one event of
        # coordinates (~650 KB) and is replaced every step.
        self._graph_cache: tuple[Tensor, Tensor] | None = None
        self.coverage_weight = coverage_weight
        self.coverage_energy_field = coverage_energy_field
        self.bce_pos_weight = bce_pos_weight
        self.propagation_lambda = propagation_lambda
        self.propagation_neighbours = propagation_neighbours
        self.propagation_radius = propagation_radius

    # ------------------------------------------------------------------ forward

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        out = super().forward(x, outputs=outputs)
        if self.propagation_lambda <= 0.0:
            return out

        key = self.output_object_hit + "_logit"
        logit = out[key]
        inputs = x.get("inputs", {})
        coord_keys = [f"{self.input_constituent}_{c}" for c in ("x", "y", "z")]
        if not all(k in inputs for k in coord_keys):
            raise KeyError(f"propagation needs {coord_keys} in inputs to build the neighbour graph")

        # The cache key is the raw input tensor, not the stacked copy below, because `torch.stack`
        # makes a new object every call and would never match. This one IS the same object across
        # the decoder's layers within a step. See _graph_cache in __init__.
        cache_key = inputs[coord_keys[0]]
        n = int(cache_key.shape[-1])
        if n != logit.shape[-1]:
            return out

        if self._graph_cache is not None and self._graph_cache[0] is cache_key:
            adj = self._graph_cache[1]
        else:
            coords = torch.stack([inputs[k][0] for k in coord_keys], dim=-1).to(torch.float32)
            dist, idx = _knn_graph(coords, self.propagation_neighbours)
            keep = dist <= self.propagation_radius
            rows = torch.arange(n, device=coords.device).unsqueeze(1).expand_as(idx)[keep]
            cols = idx[keep]
            if rows.numel() == 0:
                return out

            # Row-normalised adjacency, so propagation is an average over a cell's neighbours and
            # does not inflate logits in dense regions simply because they have more neighbours.
            # fp32, NOT logit.dtype. torch.sparse.mm has no bf16 CUDA kernel -- under the bf16-mixed
            # precision this project trains at, the sparse product raises
            #   NotImplementedError: "addmm_sparse_cuda" not implemented for 'BFloat16'
            # so the adjacency and the operand are both built in fp32 and the result is cast back
            # below. Caught by the smoke run this file's docstring asks for; the path had never
            # executed.
            counts = torch.bincount(rows, minlength=n).clamp(min=1).to(torch.float32)
            vals = (1.0 / counts[rows]).to(torch.float32)
            adj = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()
            self._graph_cache = (cache_key, adj)

        # [B, Q, N] -> [N, B*Q] so the sparse product is over the cell axis, then back.
        b, q, _ = logit.shape
        flat = logit.reshape(b * q, n).transpose(0, 1)
        # Finite-min padding entries would poison the average, so neutralise them first.
        finite = torch.isfinite(flat) & (flat > torch.finfo(flat.dtype).min / 2)
        # .float() for the same bf16 reason as the adjacency above, then straight back to the
        # logits' dtype so nothing downstream sees a precision change.
        src = torch.where(finite, flat, torch.zeros_like(flat)).to(torch.float32)
        # CASTING THE OPERANDS TO fp32 IS NOT ENOUGH -- autocast has to be switched off too.
        # torch.sparse.mm is on autocast's cast list, so inside an autocast region it demotes its
        # fp32 arguments straight back to bf16 and raises anyway. loss.py uses this same
        # `enabled=False` context to force its costs to fp32, for the same reason.
        with torch.autocast(device_type=flat.device.type, enabled=False):
            neighbour_mean = torch.sparse.mm(adj, src).to(flat.dtype)
        propagated = flat + self.propagation_lambda * neighbour_mean
        out[key] = torch.where(finite, propagated, flat).transpose(0, 1).reshape(b, q, n)
        return out

    # ------------------------------------------------------------------ loss

    def _coverage_loss(self, output: Tensor, target: Tensor, targets: dict[str, Tensor]) -> Tensor:
        """1 - (energy the matched query claims) / (energy the target actually has), per target.

        Outputs are already permuted into target order by the matcher when this runs, so row i of
        the prediction is the query matched to target i. That is what makes a per-target coverage
        term well defined here and meaningless before matching.
        """
        prob = output.sigmoid()
        weight = torch.ones_like(target)
        if self.coverage_energy_field is not None and self.coverage_energy_field in targets:
            weight = targets[self.coverage_energy_field].type_as(target).unsqueeze(-2).expand_as(target)

        claimed = (prob * target * weight).sum(-1)
        total = (target * weight).sum(-1)
        object_pad = targets[self.target_object + "_valid"]
        valid = object_pad & (total > 0)
        if not valid.any():
            return output.sum() * 0.0
        return (1.0 - claimed[valid] / total[valid].clamp(min=1e-12)).mean()

    def _weighted_bce(self, output: Tensor, target: Tensor, targets: dict[str, Tensor]) -> Tensor:
        """BCE with `pos_weight`, which hepattn's `mask_bce_loss` does not expose."""
        hit_pad = targets[self.input_constituent + "_valid"]
        object_pad = targets[self.target_object + "_valid"]
        mask = object_pad.unsqueeze(-1) & hit_pad.unsqueeze(-2)
        if not mask.any():
            return output.sum() * 0.0
        pos_weight = torch.tensor(self.bce_pos_weight, device=output.device, dtype=output.dtype)
        per_cell = torch.nn.functional.binary_cross_entropy_with_logits(
            output, target, pos_weight=pos_weight, reduction="none"
        )
        return per_cell[mask].mean()

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        losses = super().loss(outputs, targets, layer_outputs=layer_outputs)

        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_" + self.target_field].type_as(output)

        if self.coverage_weight > 0.0:
            losses["mask_coverage"] = self.coverage_weight * self._coverage_loss(output, target, targets)
        if self.bce_pos_weight > 0.0:
            losses["mask_bce_pos"] = self._weighted_bce(output, target, targets)
        return losses


__all__ = ["ExtendedObjectHitMaskTask", "loss_fns"]
