"""The differential figures.

Everything here reads the pooled long tables and nothing else -- never the event store,
never a checkpoint, never the dataset. That is deliberate and it is the acceptance criterion
for the whole design: the tables are a few tens of megabytes and can sit beside the thesis,
so an assessor regenerates every figure with numpy, scipy, pandas and matplotlib, without a
GPU, without cluster access, and without the 991 GB of ColliderML.

One workhorse, :func:`plot_binned`, draws every differential panel. Efficiency-style
quantities get a proper binomial interval; pooled energy ratios get a bootstrap band. The
distinction is enforced by which function is passed, not left to the caller's memory.
"""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import differential as diff
from src.plotting import style


def plot_binned(
    tables: Mapping[str, pd.DataFrame],
    x: str,
    y: str,
    bins: np.ndarray,
    kind: str = "fraction",
    threshold: float | None = 0.5,
    denominator: str | None = None,
    ax: plt.Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    logx: bool = True,
    ylim: tuple[float, float] | None = (0.0, 1.05),
    by_event: bool = True,
) -> plt.Axes:
    """Draw one differential panel, one line per algorithm.

    Args:
        tables: algorithm name -> long table.
        x: column to bin against.
        y: column being measured.
        bins: bin edges.
        kind: ``"fraction"`` (proportion, with an interval), ``"ratio"`` (pooled ratio of
            sums, bootstrap band) or ``"median"`` (median with an interquartile band).
        threshold: for ``kind="fraction"``, the value `y` must reach to count as a success.
            Pass ``None`` when `y` is already boolean.
        denominator: for ``kind="ratio"``, the column to divide by.
        ax: axes to draw on; a new figure is made if omitted.
        by_event: for ``kind="fraction"``, take the interval by resampling events rather
            than assuming independent particles. On by default -- particles within an event
            are correlated, so the binomial interval is too narrow.

    The three kinds are drawn differently on purpose. Fractions and ratios get error bars,
    which mean uncertainty on the value. A median gets a shaded band, because the
    interquartile range is the *spread of the distribution* and not an uncertainty on the
    median at all -- drawing it with caps, as this function used to, invited exactly the
    misreading the module docstring warns about.
    """
    ax = ax or plt.subplots()[1]

    for algo, table in tables.items():
        values = table[x].to_numpy()
        if kind == "fraction":
            passed = table[y].to_numpy() if threshold is None else (table[y].to_numpy() >= threshold)
            groups = table["sample_id"].to_numpy() if (by_event and "sample_id" in table) else None
            result = diff.binned_fraction(values, passed, bins, cluster=groups)
        elif kind == "ratio":
            result = diff.binned_ratio(values, table[y].to_numpy(), table[denominator].to_numpy(), bins)
        elif kind == "median":
            result = diff.binned_median(values, table[y].to_numpy(), bins)
        else:
            msg = f"unknown kind {kind!r}"
            raise ValueError(msg)

        populated = result.count > 0
        colour = style.colour(algo)
        if kind == "median":
            ax.fill_between(
                result.centres[populated], result.low[populated], result.high[populated],
                color=colour, alpha=0.16, linewidth=0,
            )
            ax.plot(
                result.centres[populated], result.value[populated],
                color=colour, marker=style.marker(algo), label=style.label(algo),
                linestyle=style.linestyle(algo),
            )
        else:
            ax.errorbar(
                result.centres[populated],
                result.value[populated],
                yerr=result.yerr[:, populated],
                color=colour,
                marker=style.marker(algo),
                label=style.label(algo),
                linestyle=style.linestyle(algo),
                alpha=0.75 if style.is_reference(algo) else 1.0,
                zorder=2 if style.is_reference(algo) else 3,
            )

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if ylim and kind != "median":
        ax.set_ylim(*ylim)
    ax.legend()
    return ax


def efficiency_and_purity_vs_energy(particles, clusters, working_point=0.5, out=None):
    """Deliverable 1: the headline plot, both algorithms on shared axes.

    Four panels rather than two, for two reasons.

    The working-point fraction on the top row is a step function on a continuous variable: it
    cannot distinguish a method whose recovery distribution genuinely shifted from one whose
    median merely drifted across 0.5, and the two methods here cross at almost exactly that
    threshold. The median on the bottom row says which is happening, and its band is the
    interquartile spread of the distribution itself.

    The two columns also do NOT share an x-axis -- efficiency is binned in *particle* energy
    and purity in *cluster* energy, which are different quantities over different objects.
    Sitting side by side on identical-looking log axes they used to invite a comparison that
    is not valid, so the columns are titled and labelled to make the distinction unmissable.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    plot_binned(
        _by_algo(particles), "p_energy", "eff_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[0, 0],
        xlabel="particle energy [GeV]",
        ylabel=f"fraction with eff $\\geq$ {working_point}",
    )
    plot_binned(
        _by_algo(clusters), "e_calib", "pur_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[0, 1],
        xlabel="cluster energy [GeV, calibrated]",
        ylabel=f"fraction with purity $\\geq$ {working_point}",
    )
    plot_binned(
        _by_algo(particles), "p_energy", "eff_e", diff.ENERGY_BINS,
        kind="median", ax=axes[1, 0],
        xlabel="particle energy [GeV]", ylabel="energy efficiency (median, IQR)",
    )
    plot_binned(
        _by_algo(clusters), "e_calib", "pur_e", diff.ENERGY_BINS,
        kind="median", ax=axes[1, 1],
        xlabel="cluster energy [GeV, calibrated]", ylabel="energy purity (median, IQR)",
    )
    axes[0, 0].set_title("Efficiency (per truth particle)")
    axes[0, 1].set_title("Purity (per predicted cluster)")
    for ax in axes[1]:
        ax.set_ylim(0.0, 1.05)
    return _finish(fig, out)


def performance_vs_density(particles, working_point=0.5, out=None):
    """Deliverable 2: the isolated-versus-jet-core story."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    plot_binned(
        _by_algo(particles), "dr_min", "eff_e", diff.DR_BINS,
        threshold=working_point, ax=axes[0],
        xlabel="$\\Delta R$ to nearest other target particle",
        ylabel=f"fraction with energy efficiency $\\geq$ {working_point}",
    )
    plot_binned(
        _by_algo(particles), "n_within_02", "eff_e", diff.DENSITY_BINS,
        threshold=working_point, ax=axes[1], logx=False,
        xlabel="target particles within $\\Delta R < 0.2$",
        ylabel=f"fraction with energy efficiency $\\geq$ {working_point}",
    )
    axes[0].set_title("Isolation")
    axes[1].set_title("Crowding")
    return _finish(fig, out)


def split_and_merge(particles, clusters, out=None):
    """Deliverable 3: where the two algorithms are expected to differ most.

    All three panels are **energy-weighted**, which is a change from the hit-counted versions
    and, for the left panel, not a cosmetic one: above ~8 GeV the two definitions disagree on
    the sign of MaskFormer's trend, hit-counted splitting falling with particle energy while
    energy-weighted splitting rises. Reporting the hit-counted version would have supported
    the opposite claim about whether the model fragments energetic showers.
    :func:`weighting_comparison` shows both definitions side by side; on merging they very
    nearly coincide.

    The middle panel exists because both `n_frag` definitions share a blind spot: a particle
    spread over more than ten clusters gives none of them 10%, so it registers as unsplit.
    57% of particles here have more than ten cells, so that is not a corner case. `frag_frac`,
    the share of the particle outside its largest piece, has no such blind spot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))

    plot_binned(
        _by_algo(particles), "p_energy", "is_split_e", diff.ENERGY_BINS,
        threshold=None, ax=axes[0],
        xlabel="particle energy [GeV]", ylabel="split rate", ylim=(0.0, 1.05),
    )
    plot_binned(
        _by_algo(particles), "p_energy", "frag_frac", diff.ENERGY_BINS,
        kind="median", ax=axes[1],
        xlabel="particle energy [GeV]", ylabel="energy outside largest piece",
    )
    plot_binned(
        _by_algo(clusters), "e_calib", "is_merge_e", diff.ENERGY_BINS,
        threshold=None, ax=axes[2],
        xlabel="cluster energy [GeV, calibrated]", ylabel="merge rate", ylim=(0.0, 1.05),
    )
    axes[0].set_title("Split rate (energy-weighted)")
    axes[1].set_title("Fragmentation (no blind spot)")
    axes[2].set_title("Merge rate (energy-weighted)")
    axes[1].set_ylim(0.0, 1.05)
    return _finish(fig, out)


def weighting_comparison(particles, clusters, out=None):
    """The same split and merge definitions under hit counting and under energy weighting.

    This is the figure that justifies changing the definition rather than merely asserting
    the change, and it is worth reading because the answer is asymmetric. On the right the
    two definitions of merging very nearly coincide, so nothing was being rescued there. On
    the left they cross: MaskFormer's hit-counted split rate turns over above ~8 GeV and
    falls, while its energy-weighted split rate keeps rising. The hit-counted turnover is the
    `n_frag` blind spot, not a physical improvement -- a particle scattered over more than
    ten clusters gives none of them the required 10% of its cells, so the most badly
    fragmented particles are counted as unsplit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    for ax, table, x, columns, xlabel, ylabel in (
        (axes[0], particles, "p_energy", ("is_split", "is_split_e"), "particle energy [GeV]", "split rate"),
        (axes[1], clusters, "e_calib", ("is_merge", "is_merge_e"), "cluster energy [GeV, calibrated]", "merge rate"),
    ):
        for algo, group in _by_algo(table).items():
            if style.is_reference(algo):
                continue
            for column, dash, name in ((columns[0], "--", "hits"), (columns[1], "-", "energy")):
                result = diff.binned_fraction(
                    group[x].to_numpy(), group[column].to_numpy(), diff.ENERGY_BINS,
                    cluster=group["sample_id"].to_numpy(),
                )
                populated = result.count > 0
                ax.plot(
                    result.centres[populated], result.value[populated],
                    color=style.colour(algo), marker=style.marker(algo), markersize=3.5,
                    linestyle=dash, label=f"{style.label(algo)} ({name})",
                )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.05)
        ax.legend()

    axes[0].set_title("Split rate by definition")
    axes[1].set_title("Merge rate by definition")
    return _finish(fig, out)


def reference_ceiling(particles, clusters, working_point=0.5, out=None):
    """What the task allows, next to what the two methods achieve.

    The single most important context for every other figure. An efficiency of 0.31 is not
    interpretable against 1.0, because 1.0 is not available: the left panel shows what an
    idealised method reaches when handed the true particle count and the true shower axes,
    and the right shows the purity no exclusive-partition algorithm can exceed once particles
    that share each other's cells are merged and sub-threshold deposits are accounted for.

    Read the two references differently. The seeded one is not a bound -- a better assignment
    rule beats it, and on purity it is beaten by both real methods, which is itself the
    finding that nearest-axis assignment is a weak strategy. The resolution one is a genuine
    ceiling on purity, and it is low mostly because 46% of the calorimeter energy comes from
    particles below the pT cut and has to land somewhere.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    plot_binned(
        _by_algo(particles), "p_energy", "eff_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[0],
        xlabel="particle energy [GeV]",
        ylabel=f"fraction with eff $\\geq$ {working_point}",
    )
    plot_binned(
        _by_algo(clusters), "e_calib", "pur_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[1],
        xlabel="cluster energy [GeV, calibrated]",
        ylabel=f"fraction with purity $\\geq$ {working_point}",
    )
    axes[0].set_title("Efficiency against reference")
    axes[1].set_title("Purity against ceiling")
    return _finish(fig, out)


def energy_decomposition(particles, out=None):
    """Why each method loses energy: unclustered cells versus cells given to another cluster.

    A density threshold and genuine mis-clustering are different failures with different
    fixes, and the two algorithms are expected to fail in different proportions -- CLUE
    discards cells as noise, while the model misassigns them.
    """
    fig, ax = plt.subplots()
    for algo, table in _by_algo(particles, references=False).items():
        total = table["e_dep_calib"].to_numpy()
        for column, linestyle, name in (
            ("e_lost_noise", "--", "unclustered"),
            ("e_lost_other", ":", "wrong cluster"),
        ):
            result = diff.binned_ratio(table["p_energy"].to_numpy(), table[column].to_numpy(), total, diff.ENERGY_BINS)
            populated = result.count > 0
            ax.plot(
                result.centres[populated], result.value[populated],
                color=style.colour(algo), linestyle=linestyle,
                marker=style.marker(algo), markersize=3,
                label=f"{style.label(algo)}: {name}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("particle energy [GeV]")
    ax.set_ylabel("fraction of deposited energy lost")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    return _finish(fig, out)


def working_point_curve(scan: pd.DataFrame, reference: Mapping[str, tuple[float, float]] | None = None, out=None):
    """Purity against efficiency over each method's working points.

    The comparison should not hinge on one tuning choice, so both methods are shown as
    curves with their nominal points marked.

    Args:
        scan: output of ``scripts.scan_working_points``.
        reference: optional ``algo -> (efficiency, purity)`` for the reference clusterings.
            Worth passing: on a bare 0-1 square both methods sit in a small blob near
            (0.3, 0.3) and read as uniformly poor, when the reachable region is a fraction of
            that square. The axes stay 0-1 so the plot is not quietly rescaled to flatter
            limits, but the references mark where the corner actually is.
    """
    fig, ax = plt.subplots()

    for algo, (efficiency, purity) in (reference or {}).items():
        ax.scatter(
            efficiency, purity, marker=style.marker(algo), s=70,
            color=style.colour(algo), label=style.label(algo), zorder=4, alpha=0.9,
        )
        ax.axhline(purity, color=style.colour(algo), linestyle=":", linewidth=0.9, alpha=0.5, zorder=1)
        ax.axvline(efficiency, color=style.colour(algo), linestyle=":", linewidth=0.9, alpha=0.5, zorder=1)

    for algo, group in scan.groupby("algo", observed=True):
        ordered = group.sort_values("efficiency")
        ax.plot(
            ordered["efficiency"], ordered["purity"],
            color=style.colour(algo), marker=style.marker(algo), label=style.label(algo),
        )
        nominal = ordered[ordered.get("nominal", False)]
        if len(nominal):
            ax.scatter(
                nominal["efficiency"], nominal["purity"],
                s=110, facecolors="none", edgecolors=style.colour(algo), linewidths=1.6, zorder=5,
            )
    ax.set_xlabel("efficiency")
    ax.set_ylabel("purity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return _finish(fig, out)


def efficiency_decomposition(particles, working_point=0.5, out=None):
    """Why the efficiency is what it is: the match rate is a hard ceiling on it.

    A truth particle can only reach the working point if the one-to-one assignment gave it
    a cluster at all, so `eff@wp = P(matched) x P(eff >= wp | matched)`. Plotting the two
    factors separately says whether an algorithm is failing to find particles or finding
    them and recovering too little of them -- which the single number cannot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))

    plot_binned(
        _by_algo(particles), "p_energy", "matched", diff.ENERGY_BINS,
        threshold=None, ax=axes[0],
        xlabel="particle energy [GeV]", ylabel="fraction matched to any cluster",
    )
    matched_only = {k: v[v["matched"]] for k, v in _by_algo(particles).items()}
    plot_binned(
        matched_only, "p_energy", "eff_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[1],
        xlabel="particle energy [GeV]",
        ylabel=f"fraction with eff $\\geq$ {working_point}, given matched",
    )
    plot_binned(
        _by_algo(particles), "n_hits", "eff_e", diff.NHITS_BINS,
        threshold=working_point, ax=axes[2],
        xlabel="cells on the particle (exclusive truth)",
        ylabel=f"fraction with energy efficiency $\\geq$ {working_point}",
    )
    axes[0].set_title("Ceiling: was it found at all?")
    axes[1].set_title("Given found: how much recovered?")
    axes[2].set_title("Efficiency vs particle size")
    return _finish(fig, out)


def fake_and_match_rates(particles, clusters, out=None):
    """Fakes are the other half of purity, and a particle-level cut cannot select them.

    Any purity restricted to clusters matched to a chosen set of particles silently omits
    every unmatched cluster, so it must be read next to the fake rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    plot_binned(
        _by_algo(clusters), "e_calib", "matched", diff.ENERGY_BINS,
        threshold=None, ax=axes[0],
        xlabel="cluster energy [GeV, calibrated]", ylabel="fraction of clusters matched",
    )
    plot_binned(
        _by_algo(particles), "dr_min", "matched", diff.DR_BINS,
        threshold=None, ax=axes[1],
        xlabel="$\\Delta R$ to nearest other target particle", ylabel="fraction matched to any cluster",
    )
    axes[0].set_title("Cluster match rate (1 - fake rate)")
    axes[1].set_title("Particle match rate vs isolation")
    return _finish(fig, out)


def multiowner_capability(soft_particles, working_point=0.5, out=None):
    """The capability study: what the exclusive head-to-head is unable to show.

    The left panel is the whole argument. Particles owning no cell exclusively give a
    partitioning algorithm no cell of its own to award them, so no cluster is ever about them
    and its bar sits near zero as a property of the class rather than of its tuning. (Near,
    not at: a cluster built around another particle can hold cells this one contributed to
    sub-dominantly, which the soft metric credits.) Whatever a mask-based method reaches there
    is capability the baseline does not have.

    The middle panel is the honest counterweight: soft efficiency against the particle's real
    deposited energy, where an exclusive method is capped at its `exclusive_share` and the
    model is capped by how well its mask probabilities divide a contested cell. The right
    panel says which of those is binding, by comparing how often each side calls a cell shared
    at all -- a method whose masks overlap far more than the truth does is dividing cells it
    should not have divided, and pays for it in the middle panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.9))
    tables = _by_algo(soft_particles)

    names = list(tables)
    positions = np.arange(len(names))

    impossible = [t[t["no_exclusive_cell"]] for t in tables.values()]
    recovered = [float((t["eff_e"] >= working_point).mean()) if len(t) else 0.0 for t in impossible]
    axes[0].bar(positions, recovered, color=[style.colour(a) for a in names], width=0.6)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([style.label(a) for a in names])
    axes[0].set_ylabel(f"fraction recovered at eff $\\geq$ {working_point}")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(f"Particles no partition can reach (n={len(impossible[0])})")

    plot_binned(
        tables, "p_energy", "eff_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[1],
        xlabel="particle energy [GeV]",
        ylabel=f"fraction with soft eff $\\geq$ {working_point}",
    )
    axes[1].set_title("Efficiency on multi-owner truth")

    # exclusive_share is a property of the event, so one algorithm's copy is representative.
    reference = next(iter(tables.values()))
    plot_binned(
        {"exclusive ceiling": reference}, "p_energy", "exclusive_share", diff.ENERGY_BINS,
        kind="median", ax=axes[2],
        xlabel="particle energy [GeV]", ylabel="share of energy in dominated cells",
    )
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Ceiling for any partitioning method")
    return _finish(fig, out)


def definitions_paragraph(meta: Mapping, metrics: Mapping | None = None) -> str:
    """Deliverable 7, generated from the store's own metadata so it cannot go stale.

    Args:
        meta: the store's ``meta.json``.
        metrics: the config's ``metrics`` block. The match floor and the split weighting are
            this repository's choices rather than the store's, so they are not in the store's
            metadata and have to be passed in to be stated.
    """
    hits = meta["hit_selection"]
    parts = meta["particle_selection"]
    window = meta["event_window"]
    mf = meta["maskformer"]
    layers = meta["detector"].get("layer_centres_m", {})
    calib = meta["detector"]["subsystem_calibration"]

    scoring = ""
    if metrics:
        scoring = (
            f" A matched pair must share at least {metrics['min_overlap_frac']:g} of the smaller of the "
            f"two energy totals, so a cluster merely grazing a particle is not counted as reconstructing "
            f"it. Splitting and merging use a {metrics['split_fraction']:g} threshold applied to "
            f"calibrated energy rather than to cell counts. Uncertainties on fractions come from "
            f"resampling events rather than particles, since particles within an event are not "
            f"independent. Reported figures carry two reference clusterings: a geometric ceiling, in "
            f"which an idealised method given the true particle count and shower axes assigns each "
            f"cell to the nearest axis in angle and depth at the best of a scanned range of depth "
            f"weightings; and a resolution ceiling, in which target particles sharing more than half "
            f"of the smaller one's energy are merged and then clustered perfectly."
        )

    return (
        f"Both algorithms were run on identical input: events "
        f"[{window['start_event']}, {window['start_event'] + window['num_events']}) of ColliderML ttbar pu0, "
        f"with cells zero-suppressed at {hits['calohit_min_energy']:g} GeV. Target particles satisfy "
        f"pT >= {parts['particle_min_pt']} GeV, |eta| <= {parts['particle_max_abs_eta']}, and at least "
        f"{parts['particle_min_num_calohits']} calorimeter cells counted after zero-suppression. Truth is the "
        f"exclusive partition: each cell belongs to the particle depositing the most energy in it. Truth "
        f"particles and predicted clusters are matched one-to-one by maximising shared calibrated energy with "
        f"scipy.optimize.linear_sum_assignment; unmatched truth particles count as inefficiencies and unmatched "
        f"clusters as fakes. Cell energies are calibrated per subsystem "
        f"({', '.join(f'{k} {v}' for k, v in calib.items())}). Layer indices come from a frozen geometry of "
        f"{', '.join(f'{k} {len(v)}' for k, v in layers.items())} layers. MaskFormer results use checkpoint "
        f"{str(mf['checkpoint']).rsplit('/', 1)[-1]} at mask threshold {mf['nominal_mask_threshold']} and object "
        f"threshold {mf['nominal_object_threshold']}; it was trained on events "
        f"{mf['trained_event_window']}, disjoint from those reported here." + scoring
    )


def _by_algo(table: pd.DataFrame, references: bool = True) -> dict[str, pd.DataFrame]:
    """Split a pooled table by algorithm, optionally dropping the reference clusterings.

    Panels that already carry several lines per algorithm drop them: a reference is context,
    and context that doubles the line count stops being context.
    """
    return {
        str(algo): group
        for algo, group in table.groupby("algo", observed=True)
        if references or not style.is_reference(str(algo))
    }


def _finish(fig, out):
    fig.tight_layout()
    if out is not None:
        fig.savefig(out)
        plt.close(fig)
    return fig
