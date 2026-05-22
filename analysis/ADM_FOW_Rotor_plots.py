import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_phase_relplot(
    df,
    x_col="Phase_Rounded",
    y_col="an_Turb",
    hue_col="Amplitude",
    style_col="Frequency",
    col_col="Local Thrust Coefficient",
    palette=None,
    linewidth=2.5,
    height=4,
    aspect=1.2,
    save_path=None,
    dpi=300,
):
    """
    Create a phase-averaged LES relplot.

    Parameters
    ----------
    df : pandas.DataFrame
        LES data
    x_col : str
        Column for x axis (phase)
    y_col : str
        Column for y axis
    hue_col : str
        Column controlling color (e.g. amplitude)
    style_col : str
        Column controlling line style grouping (e.g. frequency)
    col_col : str
        Column used for faceting
    palette : list or seaborn palette
        Colors for hue variable
    linewidth : float
        Line width
    height : float
        Facet height
    aspect : float
        Facet aspect ratio
    save_path : str or None
        Path to save figure
    dpi : int
        Output resolution
    """

    # ===================== CREATE RELPLOT =====================
    g = sns.relplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        style=style_col,
        palette=palette,
        linewidth=linewidth,
        kind="line",
        col=col_col,
        height=height,
        aspect=aspect,
        facet_kws={"sharey": True, "sharex": True},
    )

    g.fig.set_dpi(dpi)

    # Remove seaborn auto legend
    if g._legend is not None:
        g._legend.remove()

    # ===================== APPLY ALPHA =====================

    freqs = sorted(df[style_col].unique())
    alpha_vals = np.linspace(1.0, 0.25, len(freqs))

    amps_sorted = sorted(df[hue_col].unique())

    for ax in g.axes.flatten():

        lines = ax.get_lines()

        # Repeat alpha values for each amplitude
        alphas = list(alpha_vals) * len(amps_sorted)

        for line, alpha in zip(lines, alphas):
            line.set_alpha(alpha)
            line.set_linestyle("-")

    # ===================== PHASE REFERENCE LINES =====================

    for ax in g.axes.flatten():

        ax.axvline(0.25, color="darkgrey", linestyle="--", linewidth=1)
        ax.axvline(0.75, color="darkgrey", linestyle="--", linewidth=1)

        ax.axvspan(
            0.25,
            0.75,
            color="lightgrey",
            alpha=0.3,
            label=r"$U_{\infty,t} > U_{\infty,g}$",
        )

    # ======================= LEGEND =======================

    les_header = Line2D([], [], linestyle="none", label=r"\textbf{LES}", color="none")

    amp_handles = [
        Line2D([], [], color=palette[i] if palette else "C{}".format(i),
               lw=linewidth, label=f"$A = {amps_sorted[i]}$")
        for i in range(len(amps_sorted))
    ]

    freq_handles = [
        Line2D([], [], color="grey", lw=linewidth,
               alpha=alpha_vals[i], label=f"$f = {freqs[i]}$")
        for i in range(len(freqs))
    ]

    legend_handles = [les_header] + amp_handles + freq_handles

    g.fig.subplots_adjust(right=0.82)

    g.fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1.12, 0.5),
        frameon=False,
    )

    # ======================= LABELS =======================

    for ax in g.axes.flatten():

        ax.set_xlabel("$t/T$")
        ax.set_ylabel(y_col)

    g.set_titles("{col_name}")

    # ======================= SAVE =======================

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return g

def umm_les_contour_plot(
    df,
    x_col="Amplitude",
    y_col="Frequency",
    z_col="avg_max",
    facet_col="Ct_prime",
    z_label=r"$\overline{\max{C_{T}}}$",
    levels=20,
    cmap="viridis_r",
    show_scatter=True,
    show_contour_labels=False,
    vmin=None,
    vmax=None,
    save_path=None,
    dpi=300,
    figsize=(8, 6),
):
    """
    Create triangulated contour plots of LES/UMM data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    x_col : str
        Column used for x-axis.

    y_col : str
        Column used for y-axis.

    z_col : str
        Column used for contour values.

    facet_col : str
        Column used to create subplots (e.g. Ct').

    z_label : str
        Label for colorbar.

    levels : int or array
        Number of contour levels or explicit level list.

    cmap : str
        Colormap.

    show_scatter : bool
        Overlay sample locations.

    show_contour_labels : bool
        Draw contour labels.

    vmin, vmax : float
        Color limits.

    save_path : str
        Optional file path for saving.

    dpi : int
        Output resolution.

    figsize : tuple
        Figure size.

    Returns
    -------
    fig, axes
    """

    TITLE_FONTSIZE = 14
    LABEL_FONTSIZE = 12
    TICK_FONTSIZE = 11
    CONTOUR_LABEL_FONTSIZE = 10
    LINE_WIDTH = 1.2

    facet_vals = np.sort(df[facet_col].unique())
    n_facets = len(facet_vals)

    # ---- Determine subplot grid automatically ----
    ncols = int(np.ceil(np.sqrt(n_facets)))
    nrows = int(np.ceil(n_facets / ncols))

    # ---- Global contour range ----
    if vmin is None:
        vmin = df[z_col].min()

    if vmax is None:
        vmax = df[z_col].max()

    if isinstance(levels, int):
        levels = np.linspace(vmin, vmax, levels)

    # ---- Create figure ----
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    axes = np.atleast_1d(axes).flatten()

    cf_last = None

    for ax, val in zip(axes, facet_vals):

        sub = df[df[facet_col] == val]

        # Remove NaNs
        mask = ~np.isnan(sub[z_col])

        x = sub.loc[mask, x_col].values
        y = sub.loc[mask, y_col].values
        z = sub.loc[mask, z_col].values

        if len(z) == 0:
            continue

        # ---- Filled contour ----
        cf = ax.tricontourf(
            x,
            y,
            z,
            levels=levels,
            cmap=cmap,
            extend="both",
        )

        # ---- Contour lines ----
        cs = ax.tricontour(
            x,
            y,
            z,
            levels=levels,
            colors="k",
            linewidths=LINE_WIDTH,
        )

        if show_contour_labels:
            ax.clabel(
                cs,
                inline=True,
                fontsize=CONTOUR_LABEL_FONTSIZE,
                fmt="%.2f",
            )

        # ---- Data point markers ----
        if show_scatter:
            ax.scatter(x, y, marker="+", color="k", zorder=3)

        # ---- Titles & formatting ----
        ax.set_title(f"$C_T' = {val}$", fontsize=TITLE_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)

        cf_last = cf

    # Remove unused axes
    for ax in axes[n_facets:]:
        ax.remove()

    # ---- Axis labels (outer only) ----
    for ax in axes[:n_facets]:
        ax.set_xlabel(x_col, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(y_col, fontsize=LABEL_FONTSIZE)

    # ---- Colorbar ----
    cbar = fig.colorbar(
        cf_last,
        ax=axes[:n_facets],
        shrink=0.9,
        pad=0.02,
    )

    cbar.set_label(z_label, fontsize=LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.tight_layout()

    # ---- Save figure ----
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes