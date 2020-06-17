import numpy as np
import pandas as pd
import datetime as dt
from math import ceil  # to compute nrows

import matplotlib.pyplot as plt
import seaborn as sns

# from utils import settings

units = {
    'cpi_all_seas': 'Index 1982-84=100',
    'cpi_all_unadj': 'Index 1982-84=100',
    'cpi_energy_seas': 'Index 1982-84=100',
    'cpi_energy_unadj': 'Index 1982-84=100',
    'cpi_less_energy_seas': 'Index 1982-84=100',
    'cpi_less_energy_unadj': 'Index 1982-84=100',
    'empl_pop_ratio_seas': '% of empl. over US pop.',
    'empl_pop_ratio_unadj': '% of empl. over US pop.',
    'unempl_level_seas': "1000' people",
    'unempl_level_unadj': "1000' people",
    'unempl_rate_seas': "%",
    'unempl_rate_unadj': "%",
    'EA19_GDP_gth_rate': "Growth rate %",
    'EU27_2020_GDP_gth_rate': "Growth rate %",
    'G-20_GDP_gth_rate': "Growth rate %",
    'G-7_GDP_gth_rate': "Growth rate %",
    'OECD_GDP_gth_rate': "Growth rate %",
    'USA_GDP_gth_rate': "Growth rate %",
    'Recession': 'Binary',
    'US_debt_share_in_GDP': "% of GDP",
    'US_debt_nominal': "M$",
    'USDCNY': 'Fx rate',
    'EURUSD': 'Fx rate',
    'VIX': 'Index',
    'GSPC(S&P500)': 'Index',
    'potus': '1 = Democrat, 0 = Republican',
    'houseOfRep': '1 = Democrat, 0 = Republican',
    'fedChair': '1 = Democrat, 0 = Republican',
    'trade_balance_All': 'M$',
    'trade_balance_Goods': 'M$',
    'trade_balance_Services': 'M$',
    'WTI_oil_price': '$ per barrel',
    'FF_month_avg': 'Interest rate',
    'FF_spot_EOM': 'Interest rate',
    'FF_growth_rate': 'Growth rate %',
    'FF_month_avg_diff': 'nominal change in rate',
    'FF_month_avg_pct_change': 'Growth rate %',
    'FF_trend_diff': 'categorical: Up, Down, Stable',
    'FF_trend_pct_change': 'categorical: Up, Down, Stable',
}

def visualization_basis(data,
                        subplot_function,
                        subplot_params={},
                        date_col=None,
                        items=None,
                        excl_items=[],
                        ncols=3,
                        height_per_ax=3,  # must be an integer
                        # width_per_ax=5,  # must be an integer
                        subplot_title_complements=None,
                        fig_title=None,
                        give_grid_to_subplot_function=False
                        ):
    """
    Give a frame for subplots for all features visualization functions
    """
    if items is None:
        items = data.columns
    n_excl = len(items) - len(set(items) - set(excl_items) - set([date_col]))  # actual number of excluded items
    ncols = min(ncols, len(items) - n_excl)   
    nrows = ceil((len(items) - n_excl) / ncols)
    plt.style.use("seaborn-darkgrid")
    # Create the grid
    grid_w = ncols #* width_per_ax
    grid_h = nrows * (height_per_ax + 1) + 1
    fig = plt.figure(figsize=(16, grid_h))
    grid = plt.GridSpec(
        grid_h,
        grid_w,
        hspace=0.5,
        wspace=0.2
    )
    # Suptitle in the top box of the grid (box unused by subplots)
    title_height = 2
    ax_title = fig.add_subplot(grid[:title_height, :], fc='w')
    ax_title.annotate(
        fig_title, fontsize = 18,
        xy=(0.5, 0.7), xycoords='axes fraction',
        va='center', ha='center'
    )
    ax_title.tick_params(
        axis='both', which='both',
        left=False, labelleft=False,
        bottom=False, labelbottom=False
    )
    # x axis min and max
    if date_col is not None:  # same x_limits for all subplots: with dates
        min_x = data[date_col].iloc[0] - pd.offsets.MonthEnd(12)
        max_x = data[date_col].iloc[-1] + pd.offsets.MonthEnd(12)
    else:  # same x_limits for all subplots: without dates
        min_x = data.index[0]
        max_x = data.index[-1]  
    # Titles for subplots
    items_names = list(map(lambda x: str(x), items))
    if isinstance(subplot_title_complements, dict):
        pass
    else:
        if subplot_title_complements == None:
            zip_title = zip(items_names, items_names)
        elif isinstance(titles, str):
            zip_title = zip(items_names, [", " + subplot_title_complements])
        elif isinstance(titles, list):
            zip_title = zip(items_names, [", " + t for t in subplot_title_complements])
        subplot_title_complements = dict(zip_title)
    j = 0  # number of excluded items count
    subplot_params['text_font_size'] = 12 - ncols + 2
    for i, item in enumerate(items):
        if item in excl_items or item == date_col:  # do not plot excluded items
            j += 1
            continue
        # choose subplot position
        if ncols == 1 and nrows == 1:
            top, bottom, horiz = title_height, height_per_ax + title_height - 1, 0
        else:
            idx_row, idx_col = (i - j) // ncols, (i - j) % ncols
            top = idx_row * (height_per_ax + 1) + title_height
            bottom = idx_row * (height_per_ax + 1) + height_per_ax + title_height - 1
            horiz = idx_col           
        # plot on subplot
        if give_grid_to_subplot_function:
            grid_pos = (fig, grid, top, bottom, horiz)
            ax_pos = len(fig.axes)
            subplot_function(grid_pos, item, **subplot_params)
            for ax in fig.axes[ax_pos:]:  # first ax created in subplot_function
                ax.set_xlim(left=min_x, right=max_x)
                ax.tick_params(labelsize=12 - ncols + 2)
            if len(fig.axes[ax_pos].get_title()) == 0:  # Set title for first ax created in given grid_pos if none has been given
                fig.axes[ax_pos].set_title(subplot_title_complements[items_names[i]].replace("_", " "), fontsize=16 - ncols + 2)
        else:
            ax = fig.add_subplot(grid[top:bottom + 1, horiz])
            subplot_function(ax, item, **subplot_params)
            ax.set_xlim(left=min_x, right=max_x)
            ax.tick_params(labelsize=12 - ncols + 2)
            if len(ax.get_title()) == 0:  # Set title for ax if none exists
                ax.set_title(subplot_title_complements[items_names[i]].replace("_", " "), fontsize=16 - ncols + 2)
    fig.tight_layout()
    # grid.tight_layout(fig)
    plt.show()


def visualize_features(data,
                        date_col=None, 
                        columns=None, 
                        excl_cols=[], 
                        ncols=3, 
                        height_per_ax=2, 
                        width_per_ax=5, 
                        subplot_title_complements=None
                        ):
    if columns is None:
        columns = data.columns
    def visualize_features_subplot(ax, col_name, data=data, date_col=date_col, text_font_size=10):
        if date_col is not None:
            ax.plot(data[date_col], data[col_name])
        else:
            ax.plot(data[col_name])
    visualization_basis(data=data,
                        subplot_function=visualize_features_subplot,
                        subplot_params={},
                        date_col=date_col,
                        items=columns,
                        excl_items=excl_cols,
                        ncols=ncols,
                        height_per_ax=height_per_ax,
                        # width_per_ax=width_per_ax,
                        subplot_title_complements=subplot_title_complements,
                        fig_title="Feature visualization"
                        )

if __name__ == "__main__":
    # from utils import settings
    # settings.init()
    data = pd.read_csv("Data/dataset_monthly.csv", sep=';')
    data_pct_change = pd.read_csv("Data/dataset_monthly_pct_change.csv", sep=';')
    data['Date'] = pd.to_datetime(data['Date'])
    data_pct_change['Date'] = pd.to_datetime(data_pct_change['Date'])

    NON_NUM_COLS = ['Date', 'Recession', 'potus', 'houseOfRep', 'fedChair']  # non numeric features (boolean, date, etc)
    NUM_COLS = [x for x in data.columns if x not in NON_NUM_COLS]

    data = pd.merge(data, data_pct_change, on='Date')
    sns.set()
    visualize_features(data=data,
                    columns=NUM_COLS[:12],
                   date_col='Date',
                   ncols=2,
                   height_per_ax=3)
    

