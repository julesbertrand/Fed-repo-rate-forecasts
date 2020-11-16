import numpy as np
import pandas as pd
from math import ceil  # to compute nrows

import matplotlib.pyplot as plt
import seaborn as sns

# decorator
def visualization_grid(give_grid_to_subplot_function=False):
    def inner_function(subplot_function):
        def wrapper(
            data,
            date_col=None,
            items=None,
            excl_items=[],
            ncols=3,
            height_per_ax=3,
            subplot_titles_suffix=None,
            fig_title=None,
            subplot_params={},
        ):
            if items is None:
                items = data.columns

            # remove excluded items
            items = filter(lambda x: x not in excl_items and x != date_col, items)
            items = list(items)

            # Create the grid
            n_items = len(items)
            ncols = min(ncols, n_items)
            nrows = ceil(n_items / ncols)
            plt.style.use("seaborn-darkgrid")
            text_font_size = 14 - ncols
            grid_w = ncols
            grid_h = nrows * (height_per_ax + 1) + 1
            fig = plt.figure(figsize=(16, grid_h))
            grid = plt.GridSpec(grid_h, grid_w, hspace=0.5, wspace=0.2)

            # Suptitle setup in the top box
            title_height = 2
            ax_title = fig.add_subplot(grid[:title_height, :], fc="w")
            ax_title.annotate(
                fig_title,
                fontsize=18,
                xy=(0.5, 0.7),
                xycoords="axes fraction",
                va="center",
                ha="center",
            )
            ax_title.tick_params(
                axis="both",
                which="both",
                left=False,
                labelleft=False,
                bottom=False,
                labelbottom=False,
            )  # remove ticks and axes for title box

            # x axis min and max
            if date_col is not None:  # same x_limits for all subplots: with dates
                min_x = data[date_col].iloc[0] - pd.offsets.MonthEnd(12)
                max_x = data[date_col].iloc[-1] + pd.offsets.MonthEnd(12)
            else:  # same x_limits for all subplots: without dates
                min_x = data.index[0]
                max_x = data.index[-1]

            # subplot titles
            if isinstance(subplot_titles_suffix, dict):
                subplot_titles = {
                    key: str(val).replace("_", " ")
                    for key, val in subplot_titles_suffix.items()
                }
            else:
                if subplot_titles_suffix is None:
                    if isinstance(items[0], str):
                        subplot_titles = {item: item.replace("_", " ") for item in items}
                    else:
                        subplot_titles = {}
                elif isinstance(subplot_titles_suffix, str):
                    subplot_titles = {
                        item: (item + ", " + subplot_titles_suffix).replace("_", " ")
                        for item in items
                    }
                elif isinstance(subplot_titles_suffix, list):
                    titles = [
                        (items[i] + ", " + str(subplot_titles_suffix[i])).replace(
                            "_", " "
                        )
                        for i in range(n_items)
                    ]
                    subplot_titles = dict(zip(items, titles))

            for i, item in enumerate(items):
                idx_row, idx_col = i // ncols, i % ncols
                top = idx_row * (height_per_ax + 1) + title_height
                bottom = (
                    idx_row * (height_per_ax + 1) + height_per_ax + title_height - 1
                )
                horiz = idx_col
                # plot on subplot: either give grid or give ax
                if give_grid_to_subplot_function:
                    grid_pos = (fig, grid, top, bottom, horiz)
                    ax_pos = len(fig.axes)
                    subplot_function(
                        grid_pos, item, text_font_size=text_font_size, **subplot_params
                    )
                    for ax in fig.axes[ax_pos:]:
                        # set x lim and ticks for all subplots created by the subplot function
                        ax.set_xlim(left=min_x, right=max_x)
                        ax.tick_params(labelsize=text_font_size - 2)
                    if not fig.axes[ax_pos].get_title():
                        # Set title for first ax created in given grid_pos if none has been given
                        fig.axes[ax_pos].set_title(
                            subplot_titles[items[i]],
                            fontsize=text_font_size + 4,
                        )
                else:
                    ax = fig.add_subplot(grid[top : bottom + 1, horiz])
                    ax.set_xlim(left=min_x, right=max_x)
                    subplot_function(
                        ax, item, text_font_size=text_font_size, **subplot_params
                    )
                    ax.tick_params(labelsize=text_font_size)
                    if len(ax.get_title()) == 0:  # Set title for ax if none exists
                        ax.set_title(
                            subplot_titles[items[i]],
                            fontsize=text_font_size + 4,
                        )
            fig.tight_layout()
            # grid.tight_layout(fig)
            plt.show()

        return wrapper

    return inner_function


def visualize_features(
    data,
    date_col=None,
    columns=[],
    excl_cols=[],
    ncols=3,
    height_per_ax=2,
    subplot_titles_suffix=None,
):
    @visualization_grid(give_grid_to_subplot_function=False)
    def visualize_features_subplot(ax, col_name, data, date_col, text_font_size=10):
        if date_col is not None:
            ax.plot(data[date_col], data[col_name])
        else:
            ax.plot(data[col_name])

    visualize_features_subplot(
        data=data,
        date_col=date_col,
        items=columns,
        excl_items=excl_cols,
        ncols=ncols,
        height_per_ax=height_per_ax,
        subplot_titles_suffix=subplot_titles_suffix,
        fig_title="Feature visualization",
        subplot_params={"data": data, "date_col": date_col},
    )


def visualize_stationarity(
    data,
    date_col=None,
    columns=[],
    excl_cols=[],
    ncols=3,
    height_per_ax=2,
    subplot_titles_suffix=None,
    addfuller_results=None,
    plot_test_results=True,
):
    if addfuller_results is None:
        plot_test_results = False

    @visualization_grid(give_grid_to_subplot_function=False)
    def stationarity_subplot(
        ax,
        col_name,
        data,
        date_col,
        plot_test_results,
        addfuller_results,
        txt_box_props={},
        num_format="{:.1f}",
        text_font_size=10,
    ):
        col = data[col_name].dropna()
        # Compute rolling statistics
        rol_mean = col.rolling(window=12, min_periods=1).mean()
        rol_std = col.rolling(window=12, min_periods=1).std()
        # Plot rolling statistics
        if date_col is not None:
            # adjusting dates because of drop_na step
            date_col_temp = data[date_col].loc[col.index]
            ax.plot(
                date_col_temp,
                col,
                color=sns.color_palette()[0],
                alpha=0.8,
                label="Original",
            )
            ax.plot(
                date_col_temp,
                rol_mean,
                color=sns.color_palette()[3],
                label="Rolling Mean",
            )
            ax.plot(
                date_col_temp,
                rol_std,
                color=sns.color_palette()[2],
                label="Rolling Std",
            )
        else:
            ax.plot(col, color=sns.color_palette()[0], alpha=0.9, label="Original")
            ax.plot(rol_mean, color=sns.color_palette()[3], label="Rolling Mean")
            ax.plot(rol_std, color=sns.color_palette()[2], label="Rolling Std")
        # plot text box with Dickey-fuller test results
        if plot_test_results:
            res_test = addfuller_results.loc[col_name]
            text_str = "DF test results".center(25, " ") + "\n"
            text_str += "\n".join(
                (
                    "{:20}".format(idx + ":") + "{: >5.2f}".format(res_test[idx])
                    for idx in res_test.index
                )
            )
            ax.text(
                0.03,
                0.5,
                text_str,
                fontsize=text_font_size - 2,
                bbox=txt_box_props,
                transform=ax.transAxes,
            )

    txt_box_props = dict(
        boxstyle="round", alpha=0.8, facecolor="#EAEAF2", edgecolor="#EAEAF2"
    )
    fig_title = "Feature stationarity: rolling mean and std"
    fig_title += " and Dickey-Fuller test results" * plot_test_results
    stationarity_subplot(
        data=data,
        date_col=date_col,
        items=columns,
        excl_items=excl_cols,
        ncols=ncols,
        height_per_ax=height_per_ax,
        subplot_titles_suffix=subplot_titles_suffix,
        fig_title=fig_title,
        subplot_params={
            "data": data,
            "date_col": date_col,
            "txt_box_props": txt_box_props,
            "addfuller_results": addfuller_results,
            "plot_test_results": plot_test_results,
        },
    )


def visualize_seasonality(
    data,
    df_trend,
    df_seas,
    df_resid,
    date_col=None,
    columns=[],
    excl_cols=[],
    ncols=3,
    height_per_ax=2,
    subplot_titles_suffix=None,
):
    @visualization_grid(give_grid_to_subplot_function=False)
    def seasonality_subplot(
        ax,
        col_name,
        df_original,
        date_col,
        df_trend,
        df_seas,
        df_resid,
        text_font_size=10,
    ):
        # Plot decomposition
        ax.plot(
            df_original[date_col],
            df_original[col_name],
            color=sns.color_palette()[0],
            alpha=0.9,
            label="Original values",
        )
        ax.plot(
            df_original[date_col],
            df_resid[col_name + "_residual"],
            color=sns.color_palette()[2],
            label="Residuals",
        )
        ax.plot(
            df_original[date_col],
            df_seas[col_name + "_seasonal"],
            color=sns.color_palette()[1],
            label="Seasonality",
        )
        ax.plot(
            df_original[date_col],
            df_trend[col_name + "_trend"],
            color=sns.color_palette()[3],
            label="Trend",
        )
        ax.legend(loc="best", ncol=2, fontsize=text_font_size - 2)

    fig_title = "Feature seasonality: Original, trend, seasonality and noise"
    seasonality_subplot(
        data=data,
        date_col=date_col,
        items=columns,
        excl_items=excl_cols,
        ncols=ncols,
        height_per_ax=height_per_ax,
        subplot_titles_suffix=subplot_titles_suffix,
        fig_title=fig_title,
        subplot_params={
            "df_original": data,
            "date_col": date_col,
            "df_trend": df_trend,
            "df_seas": df_seas,
            "df_resid": df_resid,
        },
    )
