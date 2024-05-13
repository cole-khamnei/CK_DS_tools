import six
import itertools


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import KDEpy

from typing import Any, Iterable, List, Optional, Tuple, Union

# ------------------------------------------------------------------- #
# --------------------           Typing          -------------------- #
# ------------------------------------------------------------------- #

Palette = Union[str, Iterable]

# ------------------------------------------------------------------- #
# --------------------          Constants        -------------------- #
# ------------------------------------------------------------------- #

CUSTOM_PALETTES = {"custom_dark": np.array(sns.color_palette("Dark2"))[[0, 2, 5, 3, 7, 1, 4, 6]]}

# ------------------------------------------------------------------- #
# --------------------   Jupyter Notebook Tools  -------------------- #
# ------------------------------------------------------------------- #


def display_df(df: pd.DataFrame, max_rows: Optional[int] = None, max_columns: Optional[int] = None) -> None:
    """ Simple function to easily display dataframes"""

    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):
        display(df)


# ------------------------------------------------------------------- #
# --------------------       PLotting Tools      -------------------- #
# ------------------------------------------------------------------- #


def titleize(label: str) -> str:
    """ Makes label into title format"""
    return label.replace("_", " ").title()


def to_title(text: str) -> str:
    """ Makes label into title format"""
    title = ""
    prev_char = " "
    for char in text.replace("_", " "):
        if char.isupper() or not prev_char.isalpha():
            title += char.upper()
        else:
            title += char
        prev_char = char
    return title


def add_plt_labels(ax, x: Optional[str] = None, y: Optional[str] = None, title: Optional[str] = None, **kwargs) -> None:
    """ Adds plot labels"""
    if x:
        ax.set_xlabel(to_title(x))

    if y:
        ax.set_ylabel(to_title(y))

    if title:
        ax.set_title(to_title(title))


def create_palette(palette: str) -> Iterable:
    """ Creates a color palette, will add custom color palette checks"""
    if isinstance(palette, str):
        palette = CUSTOM_PALETTES[palette.lower()] if palette.lower() in CUSTOM_PALETTES else sns.color_palette(palette)
    
    return itertools.cycle(palette)


def create_subplot(n_plots: int, ncols: int = 2, width: float = 16, height_per: float = 3,
                   hspace: float = 0.3, wspace: float = .2):
    """ this function is such a homie"""

    assert n_plots > 1, f"n_plots must be greater than 1, given: {n_plots}"

    n_rows = int(np.ceil(n_plots / ncols))
    offset = 2 * max(n_rows // 3, 1)

    fig = plt.figure(figsize=(width, offset + height_per * n_rows))
    gs = fig.add_gridspec(n_rows, ncols)

    axes = []
    i = 1
    for i in range(n_plots // ncols * ncols):
        axes.append(fig.add_subplot(gs[i // ncols, i % ncols]))

    if i != n_plots - 1:
        axes.append(fig.add_subplot(gs[n_rows - 1, :]))

    fig.tight_layout(rect=[0.02, 0.02, .98, .98])
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    return fig, axes


def kde_smooth(values: np.ndarray, bw: Union[str, float] = "silverman", kernel: str = "gaussian",
               n_grid: int = 400, clip: Optional[Tuple[float, float]] = None, mean: bool = True):
    """"""

    if isinstance(bw, float):
        bw = bw * np.nanstd(values)
        bw = bw if bw > 0 else 0.05

    values = np.array(values)
    values = values[~pd.isnull(values)]

    assert len(values) >= 2, f"Values array must contained at least 2 non-null values."

    values_min, range_ = np.min(values), np.ptp(values)
    value_lower_lim, value_upper_lim = values_min - 0.05 * range_, values_min + 1.05 * range_

    kde = KDEpy.FFTKDE(bw=bw, kernel=kernel)
    x, y = kde.fit(values)(n_grid)

    if clip:
        clip_min, clip_max = clip
        start_index = 0
        multiplier = 1
        append_values = [values]
        if clip_min is not None:
            append_values.append(2 * clip_min - values)
            multiplier += 1
            start_index = n_grid
        else:
            clip_min = - np.inf

        if clip_max is not None:
            append_values.append(2 * clip_max - values)
            multiplier += 1
        else:
            clip_max = np.inf

        values = np.concatenate(append_values)

        x_fft, y_fft = KDEpy.FFTKDE(bw=bw, kernel=kernel).fit(values)(n_grid * multiplier)
        y_fft = y_fft * multiplier

        if mean:
            x_fft, y_fft = x, np.sqrt(y * y_fft[start_index: start_index + n_grid])
        else:
            x_fft, y_fft = x, y_fft[start_index: start_index + n_grid]

        clipped_indices = (x_fft <= clip_min) | (x_fft >= clip_max)
        return x_fft[~clipped_indices], y_fft[~clipped_indices]

    else:
        return x, y


def single_kde_plot(x: Union[np.ndarray, str], data: Optional[dict] = None,
                    shade: bool = True, bw: str = "silverman", kernel: str = "gaussian", mean: bool = True,
                    clip: Optional[Tuple[float, float]] = None, ax=None, add_mean: bool = False, **params):
    """"""
    x = if_str_map(x, data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    if (~pd.isnull(x)).sum() < 2:
        ax.plot([], [])
        return ax

    label = params.pop("label", None)
    fill_label, line_label = (label, None) if shade else (None, label)

    x_, y_ = kde_smooth(x, bw=bw, mean=mean, clip=clip, kernel=kernel)

    baseline, = ax.plot(x_, y_, **params, linewidth=1, label=line_label)

    if shade:
        shade_alpha = params.get("alpha", .2)
        zorder = params.get("zorder", None)
        ax.fill_between(x_, y1=y_, alpha=shade_alpha, facecolor=baseline.get_color(), label=fill_label, zorder=zorder)

    if add_mean:
        ax.axvline(np.mean(x), linestyle="--", alpha=.4, color=baseline.get_color())

    return ax


def kde_plot(x, data: Optional[dict] = None, hue: Optional[str] = None, ax=None, label: Optional[str] = None,
             labels: Optional[list] = None, shade: bool = True, alpha: float = .6, bw: float = .35, 
             add_N_label: bool =False, palette: Palette = None, **params):
    """ plots a kdeplot"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if hue is not None:

        group_variable_values = if_str_map(hue, data)
        group_variable = hue if isinstance(hue, str) else "hue"

        x_values = if_str_map(x, data)
        x = x if isinstance(x, str) else "x"

        group_values, counts = np.unique(group_variable_values, return_counts=True)
        sort_index = np.argsort(group_values)
        group_values, counts = group_values[sort_index], counts[sort_index]

        if not labels:
            labels = [f"{str(group_value).title()} ( N = {count:,})"
                      for group_value, count in zip(group_values, counts)]

        palette = create_palette(palette) if palette else None
        temp_data = pd.DataFrame({group_variable: group_variable_values, x: x_values})
        for i, (group_value, label) in enumerate(zip(group_values, labels)):
            group_data = temp_data.loc[temp_data[group_variable] == group_value]

            color = None if palette is None else next(palette)

            single_kde_plot(data=group_data, x=x, ax=ax, shade=shade, alpha=alpha, bw=bw, label=label,
                            zorder=-i, color=color, **params)

    else:
        if label and add_N_label:
            N = len(data) if data is not None else len(x)
            label = label + f" (N = {N:,})"

        single_kde_plot(data=data, x=x, ax=ax, shade=shade, alpha=alpha, bw=bw, label=label, **params)
    if label:
        ax.legend()

    return ax


def sns_wrapper(function, data, x, hue: str = None, ax=None, label: str = None, labels: list = None,
                alpha: float = .6, **params):
    """ plots a kdeplot"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if hue:
        group_variable = hue
        group_values, counts = np.unique(data[group_variable], return_counts=True)
        sort_index = np.argsort(group_values)
        group_values, counts = group_values[sort_index], counts[sort_index]

        if not labels:
            labels = [f"{str(group_value).title()} ( N = {count:,})"
                      for group_value, count in zip(group_values, counts)]

        i = 0
        for group_value, label in zip(group_values, labels):
            group_data = data.loc[data[group_variable] == group_value]
            function(data=group_data, x=x, ax=ax, alpha=alpha, label=label, zorder=i, **params)
            i -= 1

    else:
        function(data=data, x=x, ax=ax, alpha=alpha, label=label, **params)

    if label:
        ax.legend(title=hue.title())

    return ax


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, ax=None, fig=None, **kwargs):
    """ stolen from stack overflow """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    return fig, ax

# ------------------------------------------------------------------- #
# --------------------             End           -------------------- #
# ------------------------------------------------------------------- #
