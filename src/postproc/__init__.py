"""Post-processing that runs on a clustering after the fact, with no GPU and no model.

Everything here reads an event store and a set of predicted labels and returns another set of
labels, so it composes with `scripts.score` exactly as an algorithm does. Nothing in this package
imports `hepattn` or torch -- it belongs to the numpy-only half of the repository described in
`src/maskformer/README.md`, and the dependency boundary is what lets it be re-run by anyone with
the committed event store.
"""

from src.postproc.chain import chain_labels
from src.postproc.flow import flow_labels
from src.postproc.axis import axis_labels
from src.postproc.attribute import attribute_labels
from src.postproc.split import split_labels
from src.postproc.affinity import EmbeddingCache, make_affinity_fn

__all__ = ["chain_labels", "flow_labels", "axis_labels", "attribute_labels", "split_labels",
           "EmbeddingCache", "make_affinity_fn"]
