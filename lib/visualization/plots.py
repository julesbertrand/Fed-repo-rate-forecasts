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
    save_to: Union[str, bool] = False,
):
    """
    Given data and columns to plot, will show a pyplot graph with all columns plot

    data: pd.DataFrame
        DataFrame from which columns must be plot.
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
        This parameter is tricky and impacts the whole figure size.ne subplot in the pyplot grid object
            subplot_title_suffix: str, list or dict of subplot titles to add to the columns name (e.g. units)
    """
    if columns is None:
        columns = data.columns
    if excl_columns is None:
        excl_columns = []
    if fig_title is None:
        fig_title = "Feature visualization"
    if x_col not in data.columns:
        raise KeyError(f"Specified x-axis column '{x_col}' not found in data")

    @visualization_grid(pass_ax_or_grid="ax")
    def visualize_features_subplot(ax, col_name, text_fontsize, data, date_col: str = None):
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
