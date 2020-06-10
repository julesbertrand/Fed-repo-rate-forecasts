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
    'US debt share in GDP': "% of GDP",
    'US debt nominal': "M$",
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
    'WTI oil price': '$ per barrel',
    'FF month avg': 'Interest rate',
    'FF spot EOM': 'Interest rate',
    'FF growth rate': 'Growth rate %',
    'FF monthly avg diff': 'nominal change in rate',
    'FF monthly avg pct change': 'Growth rate %',
    'FF trend diff': 'categorical: Up, Down, Stable',
    'FF trend pct change': 'categorical: Up, Down, Stable',
}

def visualization_basis(data,
                        subplot_function,
                        subplot_params={},
                        date_col=None,
                        columns=None,
                        excl_cols=[],
                        ncols=3,
                        height_per_ax=3,
                        width_per_ax=5,
                        titles=None
                        ):
    """
    Give a frame for subplots for all features visualization functions
    """
    if columns is None:
        columns = data.columns
    n_excl = len(columns) - len(set(columns) - set(excl_cols) - set([date_col]))  # actual number of excluded columns
    ncols = min(ncols, len(columns))
    nrows = ceil((len(columns) - n_excl) / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * width_per_ax, nrows * height_per_ax))
    if date_col is not None:
        min_x, max_x = data[date_col].iloc[0], data[date_col].iloc[-1]  # same x_limits for all subplots: with dates
    else:
        min_x, max_x = data.index[0], data.index[-1]  # same x_limits for all subplots: without dates
    j = 0  # number of excluded columns count
    for i, col_name in zip(range(len(columns)), columns):
        if col_name in excl_cols or col_name == date_col:  # do not plot excluded columns
            j += 1
            continue
        # choose subplot position
        if ncols == 1 and nrows == 1:
            ax = axs 
        else:
            idx_row, idx_col = (i - j) // ncols, (i - j) % ncols
            if (ncols == 1) != (nrows == 1):  # XOR operator
                ax = axs[max(idx_row, idx_col)]
            else:
                ax = axs[idx_row, idx_col]
        # plot on subplot
        subplot_function(ax, col_name, **subplot_params)
        ax.set_xlim(left=min_x, right=max_x)
        # add title
        if titles == None:
            ax.set_title(col_name, size=14)
        elif isinstance(titles, str):
            ax.set_title(col_name + ", " + titles, size=14)
        elif isinstance(titles, dict):
            ax.set_title(col_name + ", " + titles[col_name], size=14)
        else:
            ax.set_title(col_name + ", " + titles[i], size=14)
    fig.tight_layout()
    plt.show()


def visualize_features(data,
                        date_col=None, 
                        columns=None, 
                        excl_cols=[], 
                        ncols=3, 
                        height_per_ax=2, 
                        width_per_ax=5, 
                        titles=None
                        ):
    if columns is None:
        columns = data.columns
    def visualize_features_subplot(ax, col_name, data=data, date_col=date_col):
        if date_col is not None:
            ax.plot(data[date_col], data[col_name])
        else:
            ax.plot(data[col_name])
    visualization_basis(data=data,
                        subplot_function=visualize_features_subplot,
                        subplot_params={},
                        date_col=date_col,
                        columns=columns,
                        excl_cols=excl_cols,
                        ncols=ncols,
                        height_per_ax=height_per_ax,
                        width_per_ax=width_per_ax,
                        titles=titles
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
                    columns=NUM_COLS[:6],
                   date_col='Date',
                   ncols=4)
    

