import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
from matplotlib.cm import ScalarMappable


def plot_events_data(
    events: list[dict],
    df: pd.DataFrame,
    column_names: list[str],
    normalize: bool = False,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import matplotlib.dates as mdates

    # Copy to avoid mutating caller data
    df = df.copy(deep=True)

    if normalize:
        df[column_names] = df[column_names] / df[column_names].max()

    # -----------------------------
    # Spacing that scales with N
    # -----------------------------
    n = max(len(events), 1)

    # Event axis height (inches): large for small N, shrinks smoothly for large N.
    base_height_per_event = 2.2
    min_height_per_event = 0.9
    height_per_event = max(
        min_height_per_event,
        base_height_per_event * (6 / max(n, 6)) ** 0.4,
    )

    # Figure height (inches)
    min_left_height = 5.0
    right_total_height = len(events) * height_per_event
    total_height = max(min_left_height, right_total_height)

    # Right-panel top/bottom padding as fraction of one event row (dimensionless)
    blank_padding = max(0.15, 0.6 / n)

    # Spacing between event axes (dimensionless GridSpec hspace)
    right_hspace = max(0.2, 1.4 / n)

    # -----------------------------
    # Figure + outer layout
    # -----------------------------
    width, _ = figsize()
    fig = plt.figure(figsize=(width, total_height), constrained_layout=True)

    (gs_left, gs_right) = fig.add_gridspec(1, 2, width_ratios=[2, 10])

    # -----------------------------
    # Left plot (full timeline)
    # -----------------------------
    ax_left = fig.add_subplot(gs_left)
    ax_left.invert_yaxis()

    interval = 4
    ax_left.yaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax_left.yaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    ax_left.set_ylabel("Time")
    ax_left.set_xlabel("Values")

    for column_name in column_names:
        ax_left.plot(df[column_name], df.index, label=column_name, alpha=0.8)

    if normalize:
        ax_left.set_xlim([0, 1])

    # -----------------------------
    # Right plot (events stack)
    # Keep the original padding-rows structure, but scale padding/hspace with N.
    # -----------------------------
    if len(events) == 0:
        return fig  # nothing to plot on the right

    height_ratios = [blank_padding] + [1] * len(events) + [blank_padding]
    ax_right = gs_right.subgridspec(
        len(height_ratios),
        1,
        height_ratios=height_ratios,
        hspace=right_hspace,
    )

    # Cache left x-limits once (used by connection patches)
    xlim_left = ax_left.get_xlim()

    for i, event in enumerate(events):
        df_sel = df[(df.index >= event["start"]) & (df.index <= event["end"])]

        ax = fig.add_subplot(ax_right[i + 1])  # +1 due to top padding row
        ax.set_title(event["name"], loc="center")

        for column_name in column_names:
            ax.plot(df_sel.index, df_sel[column_name], label=column_name, alpha=0.8)

        # Axis styling
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        if not df_sel.empty:
            ax.set_xlim((df_sel.index[0], df_sel.index[-1]))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if i == len(events) - 1:
            ax.set_xlabel("Time")

        if normalize:
            ax.set_ylim([0, 1])

        # Shaded selection on left plot
        ax_left.axhspan(event["start"], event["end"], facecolor="gray", alpha=0.5)

        # Connection patches:
        # Map the *right plot's* x-range to y-coordinates on the left plot's time axis.
        # (This is the same behavior as your original code.)
        xspan_right = ax.get_xlim()

        fig.add_artist(
            ConnectionPatch(
                xyA=(xlim_left[-1], xspan_right[0]),
                coordsA=ax_left.transData,
                xyB=(0, 1),
                coordsB=ax.transAxes,
                color="gray",
                alpha=0.75,
            )
        )
        fig.add_artist(
            ConnectionPatch(
                xyA=(xlim_left[-1], xspan_right[1]),
                coordsA=ax_left.transData,
                xyB=(0, 0),
                coordsB=ax.transAxes,
                color="gray",
                alpha=0.75,
            )
        )

    return fig


def pallete():
    """Colors from Okabe & Ito color-blind pallete. Ref: https://dovydas.com/blog/colorblind-friendly-diagrams"""
    return {
        "orange": "#E69F00",
        "sky-blue": "#56B4E9",
        "reddish-purple": "#CC79A7",
        "blue": "#0072B2",
        "vermilion": "#D55E00",
        "bluish-green": "#009E73",
        "yellow": "#F0E442",
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
            "figure.constrained_layout.use": True,
            "axes.formatter.useoffset": False,
            "axes.formatter.limits": (-3, 3),
        }
    )
    matplotlib.pyplot.ioff()


def fig_save_and_show(
    filename,
    save_title,
    show_title,
    ncol=4,
    fig=None,
    ax=None,
    **fig_legend_kws,
):
    """
    Save the current figure and show it with a title.

    NOTE: for a good result, the fig should not have been proccessed w/ tight_layout,
    and the subplot called with "constrained" layout (e.g.: `plt.subplots(layout="constrained")`)
    For instance, one can set the rc parameter as in `plt.rcParams['figure.constrained_layout.use'] = True`,
    but still care must be taken not to call tight_layout(). More on: https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html#sphx-glr-users-explain-axes-constrainedlayout-guide-py

    Args:
        filename (str, optional): The filename of the saved figure file, with file extension.
        save_title (str): The title to save in a separate file.
        show_title (str): The title to display when showing the figure.
        ncol (int, optional): The number of columns in the legend. Defaults to 4.
        fig (matplotlib.figure.Figure, optional): The figure to save. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axes to use for the legend. Defaults to None.
        **fig_legend_kws: Additional keyword arguments for the figure legend.

    Returns:
        None

    Example:
        ```
        import matplotlib.pyplot as plt

        subplots = (2,2)
        fig = plt.subplots(*subplots, figsize=figsize(subplots=subplots), layout="constrained")
        fig_save_and_show("path/to/save.pdf", "Save Title", "Show Title")
        ```
    """

    def update_handle(handle, orig):
        handle.update_from(orig)
        handle.set_alpha(1)

    if fig is None:
        fig = plt.gcf()
        if fig is None:
            return

    # Collect subplot-based axes (ignore helper axes)
    plot_axes = []
    for a in fig.get_axes():
        try:
            a.get_subplotspec()
        except Exception:
            continue
        plot_axes.append(a)
    if not plot_axes:
        return

    # Collect legend entries (deduplicated)
    handles, labels = [], []
    for a in plot_axes:
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    for leg in list(fig.legends):
        leg.remove()

    fig.canvas.draw()
    if fig.get_constrained_layout():
        fig.set_constrained_layout(False)

    fig_width_in, fig_height_in = fig.get_size_inches()
    renderer = fig.canvas.get_renderer()

    # Content bounds (including labels/ticks) in figure inches.
    x0s = []
    x1s = []
    y0s = []
    y1s = []
    for a in plot_axes:
        bbox = a.get_tightbbox(renderer)
        if bbox is not None and np.isfinite([bbox.x0, bbox.x1, bbox.y0, bbox.y1]).all():
            x0s.append(bbox.x0 / fig.dpi)
            x1s.append(bbox.x1 / fig.dpi)
            y0s.append(bbox.y0 / fig.dpi)
            y1s.append(bbox.y1 / fig.dpi)
        else:
            pos = a.get_position()
            x0s.append(pos.x0 * fig_width_in)
            x1s.append(pos.x1 * fig_width_in)
            y0s.append(pos.y0 * fig_height_in)
            y1s.append(pos.y1 * fig_height_in)

    content_x0_in = min(x0s)
    content_x1_in = max(x1s)
    content_y0_in = min(y0s)
    content_y1_in = max(y1s)

    x0_center = max(0.0, min(x0s))
    x1_center = min(fig_width_in, max(x1s))
    x_center = (x0_center + x1_center) / 2 / fig_width_in

    if handles and labels:
        leg = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(x_center, 0.0),
            bbox_transform=fig.transFigure,
            ncol=ncol,
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.8,
            labelspacing=0.6,
            borderaxespad=0.0,
            borderpad=0.2,
            handler_map={
                PathCollection: HandlerPathCollection(update_func=update_handle),
                plt.Line2D: HandlerLine2D(update_func=update_handle),
            },
            **fig_legend_kws,
        )
        leg.set_in_layout(True)
    else:
        leg = None

    title_obj = fig._suptitle
    if title_obj is None:
        title_obj = fig.suptitle(show_title)
    else:
        title_obj.set_text(show_title)
    title_obj.set_in_layout(True)
    title_obj.set_ha("center")
    title_obj.set_va("bottom")
    title_obj.set_x(x_center)
    title_obj.set_y(1.0)
    title_obj.set_multialignment("center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    title_height_in = 0.0
    if title_obj is not None:
        title_bbox = title_obj.get_window_extent(renderer)
        title_height_in = title_bbox.height / fig.dpi

    legend_height_in = 0.0
    if leg is not None:
        leg_bbox = leg.get_window_extent(renderer)
        legend_height_in = leg_bbox.height / fig.dpi

    font_size = float(plt.rcParams.get("font.size", 12))
    legend_pad_in = max(2.0, 0.30 * font_size) / 72.27
    title_pad_in = max(2.0, 0.45 * font_size) / 72.27

    top_margin_in = fig_height_in - content_y1_in
    bottom_margin_in = content_y0_in
    extra_top = max(0.0, title_height_in + title_pad_in - top_margin_in)
    extra_bottom = max(0.0, legend_height_in + legend_pad_in - bottom_margin_in)

    if extra_top > 0.0 or extra_bottom > 0.0:
        new_height_in = fig_height_in + extra_top + extra_bottom
        fig.set_size_inches(fig_width_in, new_height_in, forward=True)
        for a in plot_axes:
            pos = a.get_position()
            y0_in = pos.y0 * fig_height_in + extra_bottom
            y1_in = pos.y1 * fig_height_in + extra_bottom
            a.set_position(
                [
                    pos.x0,
                    y0_in / new_height_in,
                    pos.width,
                    (y1_in - y0_in) / new_height_in,
                ]
            )
        content_y0_in += extra_bottom
        content_y1_in += extra_bottom
        fig_height_in = new_height_in

    x_center = (content_x0_in + content_x1_in) / 2 / fig_width_in
    title_y = (content_y1_in + title_pad_in) / fig_height_in
    title_obj.set_x(x_center)
    title_obj.set_y(title_y)

    if leg is not None:
        legend_y = (content_y0_in - legend_pad_in) / fig_height_in
        leg.set_bbox_to_anchor((x_center, legend_y), transform=fig.transFigure)

    # Save (no title in image)
    if filename:
        title_obj.set_text("")
        title_obj.set_in_layout(False)
        fig.canvas.draw()
        fig.savefig(filename)
        with open(f"{filename}.title", "w") as f:
            f.write(save_title.replace("_", r"\_") + "\n")
        title_obj.set_in_layout(False)

    # Show with title
    title_obj.set_text(show_title)
    title_obj.set_in_layout(True)
    plt.show()


def _levels_from_percentiles(values, *, percentiles=None, pmin=None, pmax=None, n=None):
    """
    Build monotonically increasing contour levels from percentiles.
    - Either pass `percentiles=[...]` explicitly, OR pass pmin/pmax/n.
    - Returns a 1D strictly increasing float array (len>=2).
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("Not enough finite values to compute percentile levels.")

    if percentiles is None:
        if pmin is None:
            pmin = 1.0
        if pmax is None:
            pmax = 99.0
        if n is None:
            n = 20
        percentiles = np.linspace(float(pmin), float(pmax), int(n))
    else:
        percentiles = np.asarray(percentiles, float)

    # Clamp to [0,100]
    percentiles = np.clip(percentiles, 0.0, 100.0)

    levels = np.nanpercentile(v, percentiles)

    # Make strictly increasing: remove duplicates and enforce monotonicity
    levels = np.unique(levels)

    # Need at least 2 levels for contour/contourf
    if levels.size < 2:
        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))
        if np.isclose(lo, hi):
            raise ValueError(
                "Values are (nearly) constant; cannot build contour levels."
            )
        levels = np.array([lo, hi], float)

    return levels


def plot_efficiency_map_scattered(
    df,
    *,
    x: str,
    y: str,
    z: str,
    nx: int = 200,
    ny: int = 200,
    method: str = "linear",  # "linear", "nearest", "cubic"
    cmap: str = "viridis",
    fill_value=np.nan,
    # Masking
    mask_col: str | None = None,  # e.g. "motor_p_out"
    mask_min: float | None = None,  # e.g. 0
    # Filled contour levels (percentile-configurable)
    fill: bool = True,
    fill_alpha: float = 1.0,
    levels: int | list[float] | None = None,  # if provided, used directly
    level_source: str = "points",  # "points" or "grid"
    level_percentiles: (
        list[float] | None
    ) = None,  # explicit percentiles, e.g. [2,5,10,...,98]
    level_pmin: float = 0.0,
    level_pmax: float = 100.0,
    level_n: int = 100,
    # Line contours (percentile-configurable)
    contour_lines: bool = False,
    line_levels: int | list[float] | None = None,  # if provided, used directly
    line_source: str = "points",  # "points" or "grid"
    line_percentiles: list[float] | None = None,
    line_pmin: float = 0,
    line_pmax: float = 100.0,
    line_n: int = 10,
    line_alpha: float = 0.8,
    line_colors: str | None = None,
    label_lines: int = 0,
    line_label_fmt: str = "%.2f",
    label_fontsize: int = 11,
    # Cosmetics
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
    show_points: bool = False,
    point_size: float = 6.0,
    point_alpha: float = 0.35,
):
    """
    Interpolate scattered (x,y)->z data onto a regular grid and plot an efficiency map.

    Percentile behavior:
      - If `levels` is None: uses percentiles (level_* args) to build filled contour levels.
      - If `line_levels` is None and contour_lines=True: uses percentiles (line_* args) for line levels.

    Returns: (fig, ax, out) where out contains X,Y,Z + selected levels.
    """
    # Extract arrays
    xv = np.asarray(df[x], float)
    yv = np.asarray(df[y], float)
    zv = np.asarray(df[z], float)

    mask = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(zv)

    if mask_col is not None and mask_min is not None:
        mv = np.asarray(df[mask_col], float)
        mask &= np.isfinite(mv) & (mv > float(mask_min))

    xv, yv, zv = xv[mask], yv[mask], zv[mask]
    if xv.size < 3:
        raise ValueError("Not enough valid points to interpolate (need >= 3).")

    # Regular grid
    x_grid = np.linspace(float(np.min(xv)), float(np.max(xv)), int(nx))
    y_grid = np.linspace(float(np.min(yv)), float(np.max(yv)), int(ny))
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")

    # Interpolate
    Z = griddata(
        points=(xv, yv),
        values=zv,
        xi=(X, Y),
        method=method,
        fill_value=fill_value,
        rescale=True,
    )

    # Decide where to compute percentile levels from
    src_points = zv
    src_grid = Z

    def pick_source(which: str):
        if which == "points":
            return src_points
        if which == "grid":
            return src_grid
        raise ValueError("level_source/line_source must be 'points' or 'grid'")

    # Filled contour levels
    if levels is None:
        base = pick_source(level_source)
        levels_arr = _levels_from_percentiles(
            base,
            percentiles=level_percentiles,
            pmin=level_pmin,
            pmax=level_pmax,
            n=level_n,
        )
    else:
        levels_arr = np.asarray(levels, float)

    # Line contour levels
    if contour_lines:
        if line_levels is None:
            base = pick_source(line_source)
            line_levels_arr = _levels_from_percentiles(
                base,
                percentiles=line_percentiles,
                pmin=line_pmin,
                pmax=line_pmax,
                n=line_n,
            )
        else:
            line_levels_arr = np.asarray(line_levels, float)
    else:
        line_levels_arr = None

    # Color normalization: use full finite range of chosen source for stability
    norm_src = pick_source(level_source)
    norm_src = np.asarray(norm_src, float)
    norm_src = norm_src[np.isfinite(norm_src)]
    vmin = float(np.nanmin(norm_src))
    vmax = float(np.nanmax(norm_src))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot
    fig, ax = plt.subplots()

    if fill:
        cf = ax.contourf(
            X, Y, Z, levels=levels_arr, cmap=cmap, norm=norm, alpha=fill_alpha
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=(cbar_label or z))

    if contour_lines:
        cs = ax.contour(
            X,
            Y,
            Z,
            levels=line_levels_arr,
            colors=line_colors,
            cmap=None if line_colors else cmap,
            norm=norm,
            alpha=float(line_alpha),
        )

        for _ in range(label_lines):
            ax.clabel(
                cs, fmt=line_label_fmt, fontsize=int(label_fontsize), colors="black"
            )

    if show_points:
        ax.scatter(xv, yv, s=float(point_size), alpha=float(point_alpha))

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    out = {
        "X": X,
        "Y": Y,
        "Z": Z,
        "x_points": xv,
        "y_points": yv,
        "z_points": zv,
        "mask_used": mask,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "levels": levels_arr,
        "line_levels": line_levels_arr,
    }
    return fig, ax, out
