import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_events_data(
    fig, events: list[dict], df: pd.DataFrame, column_names: list[str], suptitle: str
) -> None:
    from matplotlib.patches import ConnectionPatch

    # Dive the figure in two gridspecs
    (gs_left, gs_right) = fig.add_gridspec(1, 2, width_ratios=[2, 10])

    # Setup the left plot
    ax_left = fig.add_subplot(gs_left)
    # Add all lineplots
    for column_name in column_names:
        ax_left.plot(df[column_name], df["timestamp"], label=column_name)
    ax_left.invert_yaxis()

    # ax_left.yaxis.set_major_locator(mdates.HourLocator(interval=2))
    # plt.xticks(np.linspace(*ax_left.get_xlim(), 3))  # type: ignore

    ax_right = gs_right.subgridspec(len(events), 1)
    for i, event in enumerate(events):
        df_sel = df[
            (df["timestamp"] >= event["start"]) & (df["timestamp"] <= event["end"])
        ]

        ax = fig.add_subplot(ax_right[i])
        ax.set_title(event["name"], loc="center")

        # Add all lineplots
        for column_name in column_names:
            ax.plot(df_sel["timestamp"], df_sel[column_name], label=column_name)

        # Setup axis
        # ax.set_ylabel('$V(t) [V]$')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.yticks(np.linspace(*ax.get_ylim(), 3))  # type: ignore

        ax.set_xlim((df_sel["timestamp"].iloc[0], df_sel["timestamp"].iloc[-1]))

        # Create the shaded area
        ax_left.axhspan(event["start"], event["end"], facecolor="gray", alpha=0.25)

        # Create the connection patches from the shaded area to this plot
        ylim = ax.get_xlim()
        xlim = ax_left.get_xlim()
        fig.add_artist(
            ConnectionPatch(
                xyA=(xlim[-1], ylim[0]),
                coordsA=ax_left.transData,
                xyB=(0, 1),
                coordsB=ax.transAxes,
                color="gray",
                alpha=0.75,
            )
        )
        fig.add_artist(
            ConnectionPatch(
                xyA=(xlim[-1], ylim[-1]),
                coordsA=ax_left.transData,
                xyB=(0, 0),
                coordsB=ax.transAxes,
                color="gray",
                alpha=0.75,
            )
        )

    fig.suptitle(suptitle)


def pallete():
    """Colors from Okabe & Ito color-blind pallete. Ref: https://dovydas.com/blog/colorblind-friendly-diagrams"""
    return {
        "orange": "#E69F00",
        "sky-blue": "#56B4E9",
        "reddish-purple": "#CC79A7",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermilion": "#D55E00",
        "bluish-green": "#009E73",
    }


def figsize(width="abntex2", fraction=1, subplots=(1, 1)):
    """Computes the figure dimensions to avoid scaling in LaTeX.

    Reference: https://jwalton.info/Embed-Publication-Matplotlib-Latex

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "abntex2":
        width_pt = 455.24411
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def config_matplotlib(
    set_linestyles=True, set_markers=False, show_dpi=200, savefig_dpi=300
):
    # Great reference: https://www.fschuch.com/blog/2020/10/14/graficos-com-qualidade-de-publicacao-em-python-com-matplotlib/

    import locale

    import matplotlib
    import scienceplots  # noqa: F401

    # Reset parameters
    matplotlib.rcdefaults()

    colors = matplotlib.cycler(color=pallete().values())
    cycle = colors

    if set_linestyles:
        linestyles = matplotlib.cycler(
            ls=["-", "--", "-.", ":", "--", "-.", ":"]
        )  # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        cycle += linestyles

    if set_markers:
        markers = matplotlib.cycler(marker=["o", "s", "^", "v", "<", ">", "d"])
        cycle += markers

    locale.setlocale(locale.LC_ALL, "pt_BR.utf8")

    matplotlib.style.use(["science", "grid", "ieee"])
    matplotlib.rcParams.update(
        {
            "axes.prop_cycle": cycle,
            "axes.formatter.use_locale": True,
            "savefig.format": "pdf",
            "figure.dpi": show_dpi,
            "figure.figsize": figsize("abntex2"),
            "savefig.bbox": "tight",
            "savefig.dpi": savefig_dpi,
            "savefig.pad_inches": 0,
            "legend.frameon": True,
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            # Tex
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{icomma}",
        }
    )


def fig_save_and_show(
    filename, save_title, show_title, ncol=4, fig=None, ax=None, **fig_legend_kws
):
    """
    Save the current figure and show it with a title.

    Args:
        filename (str): The filename of the saved figure file, with file extension.
        save_title (str): The title to save in a separate file.
        show_title (str): The title to display when showing the figure.
        ncol (int, optional): The number of columns in the legend. Defaults to 4.
        fig (matplotlib.figure.Figure, optional): The figure to save. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axes to use for the legend. Defaults to None.
        **fig_legend_kws: Additional keyword arguments for the figure legend.

    Returns:
        None

    Examples:
        fig_save_and_show("path/to/save.pdf", "Save Title", "Show Title")
    """

    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    fig_legend_params = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=ncol,
        frameon=False,  # Removes the legend frame
    )

    # Add a common legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        **fig_legend_params,
        **fig_legend_kws,
    )

    # Dynamically adjust bbox_to_anchor based on the legend height
    legend_height = legend.get_window_extent().height / plt.rcParams["figure.dpi"]
    fig_legend_params["bbox_to_anchor"] = (0.5, -0.25 * legend_height)

    # Redraw the legend with the adjusted bbox_to_anchor
    legend.remove()
    fig.legend(
        handles,
        labels,
        **fig_legend_params,
        **fig_legend_kws,
    )

    # Save the image
    plt.tight_layout()
    plt.savefig(filename)

    # Save the title
    with open(f"{filename}.title", "w") as file:
        file.write(save_title)

    # Show the image with the title
    plt.suptitle(show_title)
    plt.tight_layout()
    plt.show()
