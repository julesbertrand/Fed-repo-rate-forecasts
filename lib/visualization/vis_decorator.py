from typing import Union, Callable
from math import ceil
from functools import partial

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

MAIN_TITLE_FONTSIZE = 18
DEFAULT_TEXT_FONTSIZE = 14
LABEL_SIZE_CORR = -2
SUBPLOT_TITLE_CORR = 4


def visualization_grid(pass_ax_or_grid="ax"):
    """Decorator to create a grid for graphs and plot on it.
    Graphs will be printed by the subplot function to which the decorator is applied.

    Parameters
    ----------
    pass_ax_or_grid: str
        Either 'ax' or 'grid'.
        If 'ax', the subplot function will be passed as first argument the matplotlib subplot axes.
        If 'grid', an tuple (fig, grid, top, bottom, horiz) with top, bottom being \
the vertical delimitations of the figure and horiz it's horizontal position.

    Returns
    -------
    Decorator function that returns a matplotlib.pyplot figure
    """
    if pass_ax_or_grid not in ["ax", "grid"]:
        raise ValueError("Argument pass_ax_or_grid must be 'ax' or 'grid'.")

    def wrapper(subplot_function):
        partial_layout = partial(
            create_layout_and_plot,
            subplot_function=subplot_function,
            pass_ax_or_grid=pass_ax_or_grid,
        )
        return partial_layout

    return wrapper


def create_layout_and_plot(
    data: pd.DataFrame,
    subplot_function: Callable,
    pass_ax_or_grid: str = "ax",
    x_col: str = None,
    items: list = None,
    excl_items: list = None,
    subplot_kwargs: dict = None,
    subplot_titles: Union[list, dict] = None,
    fig_title: str = None,
    ncols: int = 3,
    height_per_ax: int = 3,
):
    """Creates the grid and subplots on it using the subplot function

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame from which columns must be plot.
    subplot_function: Callable
        Function to be used to plot a series on a subplot using either ax or grid mode
    pass_ax_or_grid: str
        If "ax", the subplot function will receive an axe object
        If "grid", the subplot function will receive a grid frame (top, bottom, horiz_position)
    x_col: str
        The column to use for a common x-axis to all subplots.
        Default is the DataFrame index.
    items: list
        items to be passed to the subplot_function.
        Default is all columns in data except x_col.
    excl_items: list
        if items is set to None (default), it will take all columns from data excpet x_col and excl_items.
    subplot_kwargs: dict
        Kwargs to be passed to the subplot function.
    subplot_titles: Union[list, dict]
        Titles for each subplot. If dict, keys must be in items.
    fig_title: str
        Main title of the figure
    ncols: int
        number of columns in the graph. Will be reduced to n_items if ncols > len(items).
    height_per_ax: int
        In grid points, height of each subplot.
        This parameter is tricky and impacts the whole figure size.

    Returns
    -------
    plt.figure
    """
    if items is None:
        items = data.columns
    if excl_items is None:
        excl_items = []
    if x_col not in data.columns:
        raise KeyError(f"Specified x-axis column '{x_col}' not found in data")

    plt.style.use("seaborn-darkgrid")

    # remove excluded items
    items = [item for item in items if (item not in excl_items and item != x_col)]

    # Create the grid
    n_items = len(items)
    ncols = min(ncols, n_items)
    nrows = ceil(n_items / ncols)
    text_fontsize = DEFAULT_TEXT_FONTSIZE - ncols
    fig, grid = create_grid(ncols=ncols, nrows=nrows, height_per_ax=height_per_ax)

    # Suptitle setup in the top box
    if fig_title not in [None, ""]:
        title_height = 2
        add_plot_main_title(fig, grid, fig_title=fig_title, title_height=title_height)
    else:
        title_height = 0

    # x axis min and max
    if x_col is None:  # without dates
        min_x = data.index[0]
        max_x = data.index[-1]
    else:  # with dates
        min_x = data[x_col].iloc[0]
        max_x = data[x_col].iloc[-1]

    subplot_titles = get_subplots_titles(items, subplot_titles)

    for i, item in enumerate(items):
        idx_row, idx_col = i // ncols, i % ncols
        top = idx_row * (height_per_ax + 1) + title_height
        bottom = idx_row * (height_per_ax + 1) + height_per_ax + title_height - 1
        horiz = idx_col
        grid_pos = (fig, grid, top, bottom, horiz)
        if pass_ax_or_grid == "grid":
            subplot_with_grid(
                item,
                subplot_function,
                subplot_kwargs,
                subplot_titles[item],
                grid_pos,
                min_x=min_x,
                max_x=max_x,
                text_fontsize=text_fontsize,
            )
        elif pass_ax_or_grid == "ax":
            subplot_with_ax(
                item,
                subplot_function,
                subplot_kwargs,
                subplot_titles[item],
                grid_pos,
                min_x=min_x,
                max_x=max_x,
                text_fontsize=text_fontsize,
            )
        else:
            raise ValueError(
                f"Argument 'pass_ax_or_grid' invalid value: {pass_ax_or_grid}. \
    Must be one of 'ax' and 'grid'"
            )

    fig.tight_layout()
    return fig


def create_grid(ncols: int, nrows: int, height_per_ax: int = 3):
    """Create matplotlib.pyplot grid"""
    grid_w = ncols
    grid_h = nrows * (height_per_ax + 1) + 1
    fig = plt.figure(figsize=(16, grid_h))
    grid = plt.GridSpec(grid_h, grid_w, hspace=0.5, wspace=0.2)
    return fig, grid


def subplot_with_grid(
    item,
    subplot_function: Callable,
    subplot_kwargs: dict,
    subplot_title: str,
    grid_pos: tuple,
    min_x,
    max_x,
    text_fontsize: int,
):
    """plot on subplot using the grid and create custom axes.
    Can create multiple subplots in subplot function on the specified grid area
    """
    fig = grid_pos[0]
    ax_pos = len(fig.axes)
    subplot_function(grid_pos, item, text_fontsize=text_fontsize, **subplot_kwargs)
    for axe in fig.axes[ax_pos:]:
        # set x lim and ticks for all subplots created by the subplot function
        axe.set_xlim(left=min_x, right=max_x)
        axe.tick_params(labelsize=text_fontsize + LABEL_SIZE_CORR)
    if not fig.axes[ax_pos].get_title():
        # Set title for first axe created in given grid_pos if none has been given
        fig.axes[ax_pos].set_title(
            subplot_title,
            fontsize=text_fontsize + SUBPLOT_TITLE_CORR,
        )


def subplot_with_ax(
    item,
    subplot_function: Callable,
    subplot_kwargs: dict,
    subplot_title: str,
    grid_pos: tuple,
    min_x,
    max_x,
    text_fontsize: int,
):
    """plot using pre-defined subplot axes"""
    fig, grid, top, bottom, horiz = grid_pos
    axe = fig.add_subplot(grid[top : bottom + 1, horiz])
    subplot_function(axe, item, text_fontsize=text_fontsize, **subplot_kwargs)
    axe.set_xlim(left=min_x, right=max_x)
    axe.tick_params(labelsize=text_fontsize - LABEL_SIZE_CORR)
    # Set title for axe if none exists
    if not axe.get_title():
        axe.set_title(subplot_title, fontsize=text_fontsize + SUBPLOT_TITLE_CORR)


def add_plot_main_title(fig, grid, fig_title: str = None, title_height: int = 2):
    """Add title to the top of the grid on height title_height"""
    ax_title = fig.add_subplot(grid[:title_height, :], fc="w")
    ax_title.annotate(
        fig_title,
        fontsize=MAIN_TITLE_FONTSIZE,
        xy=(0.5, 0.7),
        xycoords="axes fraction",
        va="center",
        ha="center",
    )
    # remove ticks and axes for title box
    ax_title.tick_params(
        axis="both",
        which="both",
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False,
    )


def get_subplots_titles(items, subplot_titles: Union[list, dict] = None) -> dict:
    """Standardize subplots titles in a dict"""
    if isinstance(subplot_titles, dict):
        pass
    elif subplot_titles is None:
        try:
            subplot_titles = {item: str(item.__name__) for item in items}
        except AttributeError:
            logger.warning("No subplot titles found: used items or columns names instead")
            subplot_titles = {item: str(item) for item in items}
    elif isinstance(subplot_titles, list):
        titles = [f"{items[i]}, {subplot_titles[i]}" for i in range(len(items))]
        subplot_titles = dict(zip(items, titles))
    else:
        raise TypeError(
            f"subplot_titles argument has wrong type {type(subplot_titles)}. \
It must be one of 'None', dict, list."
        )
    subplot_titles = {key: str(val).replace("_", " ") for key, val in subplot_titles.items()}
    return subplot_titles
