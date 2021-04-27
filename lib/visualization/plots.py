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
    Given data and columns to plot, will show a graph with all columns plot

    Parameters
    ----------
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
    return fig


@visualization_grid(pass_ax_or_grid="ax")
# pylint: disable=unused-argument, invalid-name
def visualize_features_subplot(
    ax, col_name: str, text_fontsize: int, data: pd.DataFrame, date_col: str = None
):
    """This is the subplot function for feature visualization"""
    if date_col is None:
        ax.plot(data[col_name])
    else:
        ax.plot(data[date_col], data[col_name])


def visualize_seasonality(
    original_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    date_col: str = None,
    columns: list = None,
    excl_columns: list = None,
    fig_title: str = None,
    ncols: int = 2,
    height_per_ax: int = 3,
    save_fig_filepath: str = None,
):
    """
    Given data and columns to plot, will show a graph with all columns plot

    Parameters
    ----------
    original_df: pd.DataFrame
        DataFrame with original values for each feature to be ploted.
    seasonality_df: pd.DataFrame
        DataFrame with trend, seasonality and resid columns e.g. f'{col_name}_trend'.
    date_col: str
        The column to use for a common date sequence as x-axis.
    columns: list
        columns to be passed to the subplot_function.
        Default is all columns in data except x_col.
    excl_columns: list
        if columns is set to None (default), \
it will take all columns from data except x_col and excl_items.
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
        If date_col not in data.columns
    TypeError:
        If save_to is not str or bool
    """
    if columns is None:
        columns = original_df.columns
    if excl_columns is None:
        excl_columns = []
    if fig_title is None:
        fig_title = "Seasonality visualization"
    if date_col not in original_df.columns:
        raise KeyError(f"Specified x-axis column '{date_col}' not found in data")
    if not isinstance(save_fig_filepath, (str, type(None))):
        raise TypeError(f"save_to must be None str or None. Current value: {save_fig_filepath}")

    subplot_kwargs = {
        "original_df": original_df,
        "seasonality_df": seasonality_df,
        "date_col": date_col,
    }

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    fig = seasonality_subplot(
        data=original_df,
        x_col=date_col,
        items=columns,
        excl_items=excl_columns,
        subplot_kwargs=subplot_kwargs,
        fig_title=fig_title,
        ncols=ncols,
        height_per_ax=height_per_ax,
    )

    if save_fig_filepath is not None:
        fig.save_fig(save_fig_filepath)
    return fig


@visualization_grid(pass_ax_or_grid="ax")
# pylint: disable=invalid-name
def seasonality_subplot(
    ax,
    col_name: str,
    text_fontsize: int,
    original_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    date_col: str,
):
    """Subplot function for seasonality visualisation"""
    # Plot decomposition
    ax.plot(original_df[date_col], original_df[col_name], alpha=0.9, label="Original values")
    ax.plot(original_df[date_col], seasonality_df[col_name + "_seas"], label="Seasonality")
    ax.plot(original_df[date_col], seasonality_df[col_name + "_resid"], label="Residuals")
    ax.plot(original_df[date_col], seasonality_df[col_name + "_trend"], label="Trend")
    ax.legend(loc="best", ncol=2, fontsize=text_fontsize - 2)


def visualize_stationarity(*args, **kwargs):
    raise NotImplementedError
