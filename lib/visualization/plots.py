from typing import Union

import pandas as pd

from lib.visualization.vis_decorator import visualization_grid


def visualize_features(
    data: pd.DataFrame,
    x_col: str = None,
    columns: list = None,
    excl_columns: list = None,
    subplot_titles: Union[list, dict] = None,
    fig_title: str = None,
    ncols: int = 3,
    height_per_ax: int = 2,
    save_fig_filepath: str = None,
):
    """
    Given data and columns to plot, will show a pyplot graph with all columns plot

    data: pd.DataFrame
        DataFrame from which columns must be plot.
    x_col: str
        The column to use for a common x-axis to all subplots.
        Default is the DataFrame index.
    columns: list
        columns to be passed to the subplot_function.
        Default is all columns in data except x_col.
    excl_columns: list
        if columns is set to None (default), \
it will take all columns from data except x_col and excl_items.
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
        This parameter is tricky and impacts the whole figure size
    save_fig_filepath: str
        If None, do not save (default).
        If str, path to where to save the figure.

    Raises
    ------
    KeyError:
        If x_col not in data.columns
    TypeError:
        If save_to is not str or bool
    """
    if columns is None:
        columns = data.columns
    if excl_columns is None:
        excl_columns = []
    if fig_title is None:
        fig_title = "Feature visualization"
    if x_col not in data.columns:
        raise KeyError(f"Specified x-axis column '{x_col}' not found in data")
    if not isinstance(save_fig_filepath, (str, type(None))):
        raise TypeError(f"save_to must be None str or None. Current value: {save_fig_filepath}")

    @visualization_grid(pass_ax_or_grid="ax")
    # pylint: disable=unused-argument, invalid-name
    def visualize_features_subplot(
            ax, col_name: str, text_fontsize: int, data: pd.DataFrame, date_col: str = None
    ):
        """This is the subplot function"""
        if date_col is None:
            ax.plot(data[col_name])
        else:
            ax.plot(data[date_col], data[col_name])

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    fig = visualize_features_subplot(
        data=data,
        x_col=x_col,
        items=columns,
        excl_items=excl_columns,
        subplot_kwargs={"data": data, "date_col": x_col},
        subplot_titles=subplot_titles,
        fig_title=fig_title,
        ncols=ncols,
        height_per_ax=height_per_ax,
    )

    if save_fig_filepath is not None:
        fig.save_fig(save_fig_filepath)


def visualize_seasonality(*args, **kwargs):
    raise NotImplementedError


def visualize_stationarity(*args, **kwargs):
    raise NotImplementedError
