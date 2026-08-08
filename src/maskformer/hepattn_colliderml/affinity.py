"""An auxiliary head that asks the encoder which cells share a particle.

WHY THIS EXISTS
---------------
Everything measured on the epoch-6 checkpoint says the model has no representation of a RELATION
between cells:

* the mask head is an independent sigmoid per (query, cell), so a cell can only ask "do I resemble
  this query", never "am I connected to a cell that does";
* consequently the matched cluster holds ~6 cells whatever the shower's true size, spans ~0.06 in
  angle and ~1 cm of a 42 cm depth, and no threshold, target definition, positional bandwidth or
  masked-attention setting moved it (five arms, all null);
* a learned per-cell attribution model rediscovered proximity -- its dominant feature was distance
  by 3x, and stacking it on geometric chaining gave bit-identical results;
* and the encoder's own post-encoder embeddings separate same-particle cell pairs at AUC 0.683
  against plain 3D distance's 0.661, i.e. +0.022. The relation is not in there either.

The last point is the one this head answers. The encoder has four layers of attention over the
cells and could represent co-membership, but nothing in the objective ever asks it to, so it does
not. This adds that request as an auxiliary loss and changes nothing else.

WHAT IT DOES
------------
Projects each post-encoder cell embedding into a small affinity space, and trains the cosine
similarity of pairs of NEARBY cells to predict whether they belong to the same truth particle.
Nearby, because that is the decision that matters: distant cells are trivially different particles,
and training on them would let the head score well while learning nothing about which of the two
showers reaching a contested cell owns it. The measured geometry is the reason -- in a jet core the
median distance to the next particle is 0.008 while a high-energy shower spans 0.233.

It runs in `encoder_tasks`, which hepattn provides for exactly this: tasks over post-encoder
features, whose losses need no Hungarian matching. The mask head, the classification head, the
matcher and the decoder are all untouched, so this is an addition to the architecture rather than a
replacement of it -- which matters, because the point of the study is whether a MaskFormer-style
model can do calorimeter clustering, not whether some other model can.

WHAT SUCCESS LOOKS LIKE, AND IT IS NOT THE HEADLINE METRIC
----------------------------------------------------------
One epoch will not move efficiency much. The signal to read is whether the projected embeddings beat
plain distance at separating same-particle pairs -- `dias/probe_encoder_affinity.py` measures
exactly that and gives the 0.683 / 0.661 baseline to beat. If the gap opens, the relation is
learnable and affinity-driven chaining becomes available, which is the thing every post-hoc method
so far has lacked. If it does not open, the relation is not learnable from calorimeter cells alone
at this granularity, and that is a real result for the write-up.
"""

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.task import Task


class ConstituentAffinityTask(Task):
    """Predict whether two nearby constituents belong to the same target object."""

    def __init__(
        self,
        name: str,
        input_constituent: str,
        target_object: str,
        dim: int,
        affinity_dim: int = 32,
        num_anchors: int = 2048,
        num_neighbours: int = 12,
        radius: float = 0.06,
        loss_weight: float = 1.0,
        has_intermediate_loss: bool = False,
    ):
        """
        Args:
            name: Task name, used as the key in the outputs and loss dictionaries.
            input_constituent: Constituent type, e.g. ``calohit``.
            target_object: Target object type, e.g. ``particle``. Its
                ``{target_object}_{input_constituent}_valid`` mask defines co-membership.
            dim: Embedding dimension of the encoder output.
            affinity_dim: Width of the projected affinity space. Small on purpose -- the head should
                shape the encoder rather than solve the problem in its own parameters.
            num_anchors: Cells sampled per event to build pairs from. The full pair set is O(N^2)
                at N ~ 22,000 and is neither affordable nor necessary.
            num_neighbours: Nearest neighbours per anchor, which become the pairs.
            radius: Metres. Pairs beyond this are dropped -- see the docstring on why only local
                pairs carry the decision that matters.
            loss_weight: Scale on the BCE term.
            has_intermediate_loss: Unused here; the head runs once, after the encoder.
        """
        super().__init__(has_intermediate_loss=has_intermediate_loss, permute_loss=False)

        self.name = name
        self.input_constituent = input_constituent
        self.target_object = target_object
        self.affinity_dim = affinity_dim
        self.num_anchors = num_anchors
        self.num_neighbours = num_neighbours
        self.radius = radius
        self.loss_weight = loss_weight

        self.input_objects = [f"{input_constituent}_embed"]
        self.project = Dense(dim, affinity_dim)
        # A learnable temperature and offset, so the head can calibrate how sharply cosine
        # similarity maps to a probability without the encoder having to inflate its norms.
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _sample_pairs(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Local cell pairs: `num_neighbours` nearest cells for each of `num_anchors` anchors.

        Done with a chunked cdist rather than a spatial index because it has to run on-device inside
        the training step, and N is small enough that the brute-force distance to a sampled anchor
        set is cheaper than moving data to the host to build a tree.
        """
        n = coords.shape[0]
        device = coords.device
        if n < 2:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        k = min(self.num_anchors, n)
        anchors = torch.randperm(n, device=device)[:k]
        d = torch.cdist(coords[anchors], coords)
        # Exclude self-pairs by pushing the diagonal out of range.
        d[torch.arange(k, device=device), anchors] = float("inf")

        kk = min(self.num_neighbours, n - 1)
        near_d, near_i = torch.topk(d, kk, dim=1, largest=False)
        keep = near_d <= self.radius
        a = anchors.unsqueeze(1).expand_as(near_i)[keep]
        b = near_i[keep]
        return a, b

    def forward(self, x: dict[str, Tensor], outputs: dict[str, dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        embed = x[f"{self.input_constituent}_embed"]
        z = torch.nn.functional.normalize(self.project(embed), dim=-1)

        inputs = x.get("inputs", {})
        coord_keys = [f"{self.input_constituent}_{c}" for c in ("x", "y", "z")]
        if not all(key in inputs for key in coord_keys):
            # Without coordinates there is no notion of "nearby", and a head trained on random pairs
            # would learn the trivial task. Fail loudly rather than train something meaningless.
            raise KeyError(f"ConstituentAffinityTask needs {coord_keys} in inputs to sample local pairs")
        coords = torch.stack([inputs[key][0] for key in coord_keys], dim=-1).to(z.dtype)

        a, b = self._sample_pairs(coords)
        logit = self.log_scale.exp() * (z[0][a] * z[0][b]).sum(-1) + self.bias
        return {"pair_a": a, "pair_b": b, "pair_logit": logit, f"{self.input_constituent}_affinity_embed": z}

    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {"pair_prob": outputs["pair_logit"].sigmoid()}

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        logit = outputs["pair_logit"]
        if logit.numel() == 0:
            return {"pair_bce": logit.sum() * 0.0}

        a, b = outputs["pair_a"], outputs["pair_b"]
        valid = targets[f"{self.target_object}_{self.input_constituent}_valid"][0]  # [n_obj, n_cells]
        # Two cells share a particle if some particle covers both. Computed on the multi-owner mask
        # rather than an exclusive partition, because co-membership is a genuinely symmetric
        # relation and does not need the arbitrary tie-break that exclusivity imposes.
        same = (valid[:, a] & valid[:, b]).any(dim=0).type_as(logit)

        # Positives are the minority -- 18.4% of local pairs, measured. Weighting them keeps the
        # head from collapsing onto "never the same", which is the same class-imbalance trap the
        # mask head's BCE sits in.
        pos = same.mean().clamp(min=1e-6)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, same, pos_weight=(1 - pos) / pos)
        return {"pair_bce": self.loss_weight * loss}

    def metrics(self, preds: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}
