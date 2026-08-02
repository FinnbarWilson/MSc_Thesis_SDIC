"""Energy-weighted purity and efficiency, pooled across events.

Cells are weighted by their energy rather than counted, because a calorimeter
measures energy and not occupancy: an algorithm can discard most of a particle's
energy while still recovering most of its cells. Efficiency is taken per truth
particle rather than per cluster, so a particle merged into a neighbour counts
as a miss instead of vanishing from the denominator.

Metrics are pooled over events rather than averaged per event, so every particle
carries the same weight regardless of how busy its event was. A per-event mean
would let a quiet event outweigh a crowded one, which is the opposite of what a
clustering benchmark should do.

The same definitions serve tuning and reporting. Tuning toward one quantity and
reporting another leaves the reported number optimised for nothing.
"""

from dataclasses import dataclass, field

import numpy as np

from src.evaluation.matching import cluster_matches, particle_matches


@dataclass
class MetricAccumulator:
    """Running totals for pooled purity and efficiency.

    Purity and efficiency are ratios of summed energies, so they are accumulated
    as four totals and divided once at the end rather than averaged per event.
    """

    matched_energy: float = 0.0
    cluster_energy: float = 0.0
    recovered_energy: float = 0.0
    target_energy: float = 0.0
    n_targets: int = 0
    n_clusters: int = 0
    efficiencies: list = field(default_factory=list)

    def add_event(self, event, cluster_ids, targets, hit_mask=None):
        """Fold one event's clustering into the running totals.

        Args:
            event: one event row.
            cluster_ids: label per cell, -1 for unclustered.
            targets: reconstructable particles for this event, from
                :func:`src.data.truth.reconstructable_particles`.
            hit_mask: optional boolean mask over cells.
        """
        matches = cluster_matches(event, cluster_ids, hit_mask)
        self.matched_energy += float(matches["matched_energy"].sum()) if not matches.empty else 0.0
        self.cluster_energy += float(matches["cluster_energy"].sum()) if not matches.empty else 0.0
        self.n_clusters += len(matches)

        if targets.empty:
            return
        per_particle = particle_matches(matches, targets)
        self.recovered_energy += float(per_particle["recovered_energy"].sum())
        self.target_energy += float(per_particle["deposited_energy"].sum())
        self.n_targets += len(per_particle)
        self.efficiencies.extend(per_particle["efficiency"].fillna(0.0).tolist())

    def purity(self):
        """Pooled energy-weighted purity, or 0 if nothing was clustered."""
        if self.cluster_energy <= 0:
            return 0.0
        return self.matched_energy / self.cluster_energy

    def efficiency(self):
        """Pooled energy-weighted efficiency, or 0 if there were no targets."""
        if self.target_energy <= 0:
            return 0.0
        return self.recovered_energy / self.target_energy

    def f1(self):
        """Harmonic mean of pooled purity and efficiency."""
        return harmonic_mean(self.purity(), self.efficiency())

    def working_point_fractions(self, working_points):
        """Fraction of target particles reconstructed above each efficiency threshold.

        Args:
            working_points: efficiency thresholds, for example ``[0.5, 0.75, 1.0]``.

        Returns:
            dict: threshold to the fraction of targets meeting it.
        """
        if not self.efficiencies:
            return {point: 0.0 for point in working_points}
        values = np.asarray(self.efficiencies, dtype=float)
        return {
            point: float(np.mean(values >= point - 1e-9)) for point in working_points
        }

    def summary(self, working_points=None):
        """Every headline number as a plain dictionary, ready to serialise."""
        out = {
            "purity": self.purity(),
            "efficiency": self.efficiency(),
            "f1": self.f1(),
            "n_targets": self.n_targets,
            "n_clusters": self.n_clusters,
        }
        if working_points:
            out["working_points"] = {
                str(point): value
                for point, value in self.working_point_fractions(working_points).items()
            }
        return out


def harmonic_mean(first, second):
    """Harmonic mean of two non-negative numbers, 0 if either is 0."""
    if first <= 0 or second <= 0:
        return 0.0
    return 2.0 * first * second / (first + second)


def collapse_overlapping_masks(scores):
    """Reduce per-object scores over cells to one label per cell.

    The CLUE baseline returns a partition, whereas a set-prediction model may
    claim a cell with more than one object. Collapsing by the strongest claim
    puts the two on identical footing, at the cost of discarding the overlap the
    learned model is able to express. Use it as a cross-check rather than as the
    primary comparison.

    Args:
        scores: array of shape ``(n_objects, n_cells)``.

    Returns:
        np.ndarray: object index per cell, -1 where no object claims it.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.zeros(0, dtype=int)
    best = scores.argmax(axis=0)
    claimed = scores.max(axis=0) > 0
    return np.where(claimed, best, -1)
