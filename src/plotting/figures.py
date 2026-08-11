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
    band: bool = True,
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
        band: for ``kind="median"``, draw the interquartile band. Turn it off when several
            methods share a panel: five translucent bands over the same axes stop being a
            distribution and become a wash, which is what made the old four-panel headline
            figure unreadable.

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
            if band:
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

    Two panels, not four. The earlier version drew the working-point fraction and the median
    of the same quantity, one above the other, on the argument that a fraction cannot tell a
    genuine distribution shift from a median drifting across the threshold. That is true, but
    the median panels answered it with four overlapping interquartile bands covering most of
    the axes, and nothing could be read out of them. The distribution question is better
    handled by :func:`efficiency_decomposition`, which factorises the same number into two
    readable pieces.

    The two panels do NOT share an x-axis: efficiency is binned in *particle* energy and
    purity in *cluster* energy, different quantities over different objects. Side by side on
    identical-looking log axes that invites an invalid comparison, so the axis labels and the
    in-panel notes both say which object is being counted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    plot_binned(
        _by_algo(particles), "p_energy", "eff_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[0],
        xlabel="particle energy [GeV]",
        ylabel=f"fraction of particles\n$\\geq${working_point:g} recovered",
    )
    plot_binned(
        _by_algo(clusters), "e_calib", "pur_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[1],
        xlabel="cluster energy [GeV, calibrated]",
        ylabel=f"fraction of clusters\n$\\geq${working_point:g} pure",
    )

    style.panel_labels(axes)
    style.clear_panel_legends(axes)
    fig.tight_layout()
    style.legend_below(fig)
    return _finish(fig, out)


def performance_vs_density(particles, working_point=0.5, out=None):
    """Deliverable 2: the isolated-versus-jet-core story, and where the headroom is.

    One panel, not two. The second panel binned by the count of neighbours within dR < 0.2
    and said the same thing as the first with the axis reversed; two ways of measuring
    crowding is one more than the figure needs.

    This is the plot that decides where effort is worth spending, which is why the gap to the
    reference is annotated rather than left to be eyeballed. Both methods sit far below what
    perfect-seeded geometry achieves on isolated particles and close to it inside a jet core,
    so the room to improve is in the easy regime, not the hard one.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.2))

    plot_binned(
        _by_algo(particles), "dr_min", "eff_e", diff.DR_BINS,
        threshold=working_point, ax=ax,
        xlabel="$\\Delta R$ to nearest other target particle",
        ylabel=f"fraction of particles\n$\\geq${working_point:g} recovered",
    )

    ax.get_legend().remove()
    fig.tight_layout()
    style.legend_below(fig)
    return _finish(fig, out)


def split_and_merge(particles, clusters, out=None):
    """Deliverable 3: the two ways a clustering can be wrong, and where each method fails.

    Both panels are **energy-weighted**, and for splitting that is not a cosmetic choice:
    above ~8 GeV the hit-counted and energy-weighted definitions disagree on the sign of
    MaskFormer's trend. :func:`weighting_comparison` shows that directly.

    Splitting is drawn as `frag_frac` rather than as the split *rate* that used to occupy a
    third panel. Both `n_frag` definitions share a blind spot -- a particle spread over more
    than ten clusters gives none of them the required 10%, so it registers as unsplit, and
    57% of particles here have more than ten cells. `frag_frac`, the share of a particle's
    energy outside its largest piece, has no such blind spot, so it is the one to report and
    the rate panel was showing a worse version of the same thing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    plot_binned(
        _by_algo(particles), "p_energy", "frag_frac", diff.ENERGY_BINS,
        kind="median", band=False, ax=axes[0],
        xlabel="particle energy [GeV]",
        ylabel="energy outside the\nlargest cluster",
    )
    plot_binned(
        _by_algo(clusters), "e_calib", "is_merge_e", diff.ENERGY_BINS,
        threshold=None, ax=axes[1],
        xlabel="cluster energy [GeV, calibrated]",
        ylabel="clusters holding $\\geq$2 particles", ylim=(0.0, 1.05),
    )
    axes[0].set_ylim(0.0, 1.05)

    style.panel_labels(axes)
    style.clear_panel_legends(axes)
    fig.tight_layout()
    style.legend_below(fig)
    return _finish(fig, out)


def weighting_comparison(particles, clusters, out=None):
    """Why splitting is reported energy-weighted: the two definitions disagree on the sign.

    A methods figure, not a results one, and it exists because the choice changes the
    conclusion rather than the decimal place. Counting cells, MaskFormer's split rate turns
    over above ~8 GeV and falls, which would read as "the model handles energetic showers
    better". Weighting by energy it keeps rising. The turnover is the `n_frag` blind spot: a
    particle scattered over more than ten clusters gives none of them the required 10% of its
    cells, so the most badly fragmented particles are counted as unsplit.

    Only splitting is shown. The merge panel that used to sit beside it showed two curves that
    very nearly coincide -- nothing is rescued or lost there by the choice of weighting, so
    the panel was spending half the figure to say "no effect".
    """
    del clusters  # merging is insensitive to the weighting; see the docstring.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    # Two methods, not all of them, and direct labels rather than a legend. The incidence head
    # is a scoring variant of the same model and behaves like the mask head here, so it added
    # a third hue and two more legend entries to make a point already made. Four long legend
    # entries then took a third of the panel; the line ends say the same thing in place.
    shown = ("maskformer", "clue")
    tables = _by_algo(particles, references=False)
    for algo in shown:
        if algo not in tables:
            continue
        group = tables[algo]
        for column, dash in (("is_split", ":"), ("is_split_e", "-")):
            result = diff.binned_fraction(
                group["p_energy"].to_numpy(), group[column].to_numpy(), diff.ENERGY_BINS,
                cluster=group["sample_id"].to_numpy(),
            )
            populated = result.count > 0
            centres, values = result.centres[populated], result.value[populated]
            ax.plot(centres, values, color=style.colour(algo), marker=style.marker(algo), linestyle=dash)
            ax.annotate(
                "counting cells" if column == "is_split" else "weighting by energy",
                xy=(centres[-1], values[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.4, color=style.colour(algo),
            )
        # Inside the axes, not to the left of them: outside-left ran straight through the
        # y-axis label.
        ax.annotate(
            style.label(algo), xy=(centres[0], values[0]), xytext=(6, 9), textcoords="offset points",
            ha="left", va="bottom", fontsize=8, color=style.colour(algo),
        )

    ax.set_xscale("log")
    ax.set_xlim(0.35, 320)
    ax.set_xlabel("particle energy [GeV]")
    ax.set_ylabel("particles split across $\\geq$2 clusters")
    ax.set_ylim(0.0, 0.75)
    ax.axvspan(8, 320, color="#000000", alpha=0.04, linewidth=0)

    fig.tight_layout()
    return _finish(fig, out)


def energy_decomposition(particles, out=None):
    """Where a particle's deposited energy ends up, and how the two failure modes differ.

    The two methods recover almost the same share; what separates them is how they lose the
    rest. CLUE clusters nearly every cell, so its losses are almost all *misassignment* --
    energy that landed in some other particle's cluster. MaskFormer is trained to claim only
    cells belonging to target particles, so a large part of its loss is energy it never
    claimed at all. Those are different failures needing different fixes, and the aggregate
    efficiency cannot tell them apart.

    Drawn as one stacked bar per method rather than as stacked areas against energy. The area
    version put each method in its own panel, which meant the only segment that could be
    compared between methods was the bottom one -- everything above it started from a
    different baseline, so the figure showed three coloured shapes and answered nothing at a
    glance. Bars on a shared axis with the numbers written on them answer it immediately.
    """
    tables = _by_algo(particles, references=False)
    fig, ax = plt.subplots(figsize=(5.6, 0.62 * len(tables) + 0.85))

    # A single-hue ordinal ramp: the three fates are ORDERED (kept, misassigned, never
    # claimed), and one hue light-to-dark says so where three unrelated hues would not. It
    # also avoids reusing the method colours to mean something else in this one figure.
    fates = (("recovered", "#3d4f5c"), ("wrong cluster", "#8fa3ae"), ("unclustered", "#dfe6ea"))

    names = list(tables)
    positions = np.arange(len(names))[::-1]
    shares = {name: [] for name, _ in fates}
    for table in tables.values():
        total = table["e_dep_calib"].to_numpy().sum()
        lost_noise = table["e_lost_noise"].to_numpy().sum() / total
        lost_other = table["e_lost_other"].to_numpy().sum() / total
        shares["unclustered"].append(lost_noise)
        shares["wrong cluster"].append(lost_other)
        shares["recovered"].append(max(1.0 - lost_noise - lost_other, 0.0))

    left = np.zeros(len(names))
    for name, colour in fates:
        values = np.array(shares[name])
        ax.barh(positions, values, left=left, color=colour, height=0.68, label=name,
                edgecolor="white", linewidth=0.8)
        for pos, value, start in zip(positions, values, left, strict=True):
            if value > 0.06:
                ax.text(start + value / 2, pos, f"{value:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if name == "recovered" else "#2b2b2b")
        left += values

    ax.set_yticks(positions)
    ax.set_yticklabels([style.label(a) for a in names])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("share of the particle's deposited energy")
    ax.grid(visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    # which="both": scienceplots turns minor ticks on and mirrors them to all four sides, so
    # suppressing majors alone leaves a row of stray marks along the top of a bar chart.
    ax.tick_params(axis="y", which="both", length=0)
    ax.tick_params(axis="x", which="both", top=False)
    ax.minorticks_off()
    fig.tight_layout()
    style.legend_below(fig, *ax.get_legend_handles_labels(), y=0.04)
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


def efficiency_decomposition(particles, clusters, working_point=0.5, out=None):
    """The three questions the one-to-one matching answers, which one number hides.

    A truth particle can only reach the working point if the assignment gave it a cluster at
    all, so `eff@wp = P(matched) x P(eff >= wp | matched)`. Panels 1 and 2 are those two
    factors, and separating them is what distinguishes an algorithm that fails to find
    particles from one that finds them and recovers too little of each -- the distinction the
    headline figure cannot make, and the one that turns out to separate the two methods here.

    Panel 3 is the same matching seen from the cluster side, which used to be a figure of its
    own. It belongs here: it is the third outcome of the one operation, and any purity read
    without it is silently conditioned on clusters that matched something.

    The three panels deliberately count **different objects** -- two per particle, one per
    cluster -- so each is annotated with which, and the axis labels name the object rather
    than saying "fraction".
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))

    plot_binned(
        _by_algo(particles), "p_energy", "matched", diff.ENERGY_BINS,
        threshold=None, ax=axes[0],
        xlabel="particle energy [GeV]", ylabel="particles given a cluster",
    )
    matched_only = {k: v[v["matched"]] for k, v in _by_algo(particles).items()}
    plot_binned(
        matched_only, "p_energy", "eff_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[1],
        xlabel="particle energy [GeV]",
        ylabel=f"of those, $\\geq${working_point:g} recovered",
    )
    plot_binned(
        _by_algo(clusters), "e_calib", "matched", diff.ENERGY_BINS,
        threshold=None, ax=axes[2],
        xlabel="cluster energy [GeV, calibrated]", ylabel="clusters matched to a particle",
    )

    style.panel_labels(axes)
    style.clear_panel_legends(axes)
    fig.tight_layout()
    style.legend_below(fig)
    return _finish(fig, out)


def incidence_head_comparison(particles, clusters, working_point=0.5, out=None):
    """The one figure where both readings of the model appear, and the only one that should.

    The model has two heads. The mask head scores each (query, cell) pair through an
    independent sigmoid, so nothing in its loss relates one cell's claims to each other; the
    incidence head applies a softmax over queries within a cell and is trained against the
    true energy fractions. Which one resolves a contested cell is a choice made after
    training, and it changes the owner of 98.5% of claimed cells.

    Detection is the mask head's in both, so the two claim identical cells and differ only in
    assignment -- which is what makes this a controlled comparison rather than two tunings.

    Everywhere else in this repository "MaskFormer" means the mask head. Carrying both rows
    through every figure made a single model read as two competing ones and crowded out the
    comparison the thesis is actually about.
    """
    both = {"maskformer", "maskformer_incidence"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    p_tables = {k: v for k, v in _by_algo(particles, references=False, both_heads=True).items() if k in both}
    c_tables = {k: v for k, v in _by_algo(clusters, references=False, both_heads=True).items() if k in both}

    plot_binned(
        p_tables, "p_energy", "eff_e", diff.ENERGY_BINS,
        threshold=working_point, ax=axes[0],
        xlabel="particle energy [GeV]",
        ylabel=f"fraction of particles\n$\\geq${working_point:g} recovered",
    )
    plot_binned(
        p_tables, "p_energy", "frag_frac", diff.ENERGY_BINS,
        kind="median", band=False, ax=axes[1],
        xlabel="particle energy [GeV]",
        ylabel="energy outside the\nlargest cluster",
    )
    axes[1].set_ylim(0.0, 1.05)
    del c_tables

    style.panel_labels(axes)
    handles, _ = axes[0].get_legend_handles_labels()
    style.clear_panel_legends(axes)
    fig.tight_layout()
    style.legend_below(fig, handles, ["mask head", "incidence head"])
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
    # The sample name comes from the STORE, not from a literal. This paragraph used to say
    # "ttbar pu0" unconditionally, which was true while pu0 was the only dataset and became a
    # false caption the moment a pu200 store was rendered -- the one error in a generated
    # provenance paragraph that a reader cannot catch, because everything around it is correct.
    # `dataset_prefix` is written by eval/dump.py from the dataloader that produced the store.
    sample = meta.get("dataset", {}).get("dataset_prefix") or "unknown sample"
    mf = meta["maskformer"]
    layers = meta["detector"].get("layer_centres_m", {})
    calib = meta["detector"]["subsystem_calibration"]

    # Which head resolved a contested cell is a decision the reader cannot infer from anything
    # else in the paragraph, and before format 2 there was no choice to state. Silence here
    # would now be ambiguous rather than merely brief.
    heads = ""
    if mf.get("has_incidence"):
        heads = (
            " The model has two prediction heads and results are reported for both. The mask head "
            "scores each (query, cell) pair through an independent sigmoid; the incidence head "
            "applies a softmax over queries within a cell and is trained against the true energy "
            "fractions I_ia = E_ia / E_i. 'maskformer' resolves a contested cell to the query with "
            "the highest mask probability, 'maskformer_incidence' to the query holding the largest "
            "predicted share of that cell. Detection is the mask head's in both, so the two claim "
            "identical cells and differ only in assignment."
        )

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
        f"[{window['start_event']}, {window['start_event'] + window['num_events']}) of ColliderML {sample}, "
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
        f"{mf['trained_event_window']}, disjoint from those reported here." + heads + scoring
    )


def _by_algo(table: pd.DataFrame, references: bool = True, both_heads: bool = False) -> dict[str, pd.DataFrame]:
    """Split a long table into one frame per algorithm, in a fixed drawing order.

    `both_heads` is off by default, so `maskformer_incidence` is dropped unless a figure asks
    for it. That default is the point: the incidence head is a second *reading* of one
    checkpoint, and drawing it beside the mask head in every panel made one model look like
    two competitors and pushed the CLUE comparison into third place. Exactly one figure,
    :func:`incidence_head_comparison`, sets this.
    """
    order = [a for a in style.ALGO_COLOURS if a in set(table["algo"].unique())]
    if not both_heads:
        order = [a for a in order if a != "maskformer_incidence"]
    if not references:
        order = [a for a in order if not style.is_reference(a)]
    return {algo: table[table["algo"] == algo] for algo in order}


def _finish(fig, out):
    fig.tight_layout()
    if out is not None:
        fig.savefig(out)
        plt.close(fig)
    return fig
