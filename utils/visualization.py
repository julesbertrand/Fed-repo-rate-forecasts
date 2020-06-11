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
                        columns=None,
                        excl_cols=[],
                        ncols=3,
                        height_per_ax=3,
                        width_per_ax=5,
                        subplot_title_complements=None,
                        fig_title=None
                        ):
    """
    Give a frame for subplots for all features visualization functions
    """
    if columns is None:
        columns = data.columns
    n_excl = len(columns) - len(set(columns) - set(excl_cols) - set([date_col]))  # actual number of excluded columns
    ncols = min(ncols, len(columns) - n_excl)   
    nrows = ceil((len(columns) - n_excl) / ncols)
    plt.style.use("seaborn-darkgrid")
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols, 
        figsize=(ncols * width_per_ax, nrows * height_per_ax)
    )
    if date_col is not None:
        min_x, max_x = data[date_col].iloc[0], data[date_col].iloc[-1]  # same x_limits for all subplots: with dates
    else:
        min_x, max_x = data.index[0], data.index[-1]  # same x_limits for all subplots: without dates
    if isinstance(subplot_title_complements, dict):
        pass
    elif subplot_title_complements == None:
        subplot_title_complements = dict(zip(columns, columns))
    elif isinstance(titles, str):
        subplot_title_complements = dict(zip(columns, [", " + subplot_title_complements]))
    elif isinstance(titles, list):
        subplot_title_complements = dict(zip(columns, [", " + t for t in subplot_title_complements]))
    j = 0  # number of excluded columns count
    for i, col_name in zip(range(len(columns)), columns):
        if col_name in excl_cols or col_name == date_col:  # do not plot excluded columns
            j += 1
            continue
        # choose subplot position
        if ncols == 1 and nrows == 1:
            ax = axes 
        else:
            idx_row, idx_col = (i - j) // ncols, (i - j) % ncols
            if (ncols == 1) != (nrows == 1):  # XOR operator
                ax = axes[max(idx_row, idx_col)]
            else:
                ax = axes[idx_row, idx_col]
        # plot on subplot
        subplot_function(ax, col_name, **subplot_params)
        ax.set_xlim(left=min_x, right=max_x)
        ax.set_title(subplot_title_complements[col_name].replace("_", " "), size=14)
    fig.suptitle(fig_title, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
                    columns=NUM_COLS[:6],
                   date_col='Date',
                   ncols=4)
    

