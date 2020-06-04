import numpy as np
import datetime as dt
import pandas as pd
from functools import reduce
from glob import glob
import ntpath
from math import ceil

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

from statsmodels.tsa.stattools import adfuller

from utils import settings

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
    'FF trend': 'categorical: Up, Down, Stable'
}

def visualize_features(data, columns=None, date_col=None, excl_cols=[], height_per_ax=2, width_per_ax=5, ncols=3, titles=None):
    if columns is None:
        columns = data.columns
    n_excl = len(columns) - len(set(columns) - set(excl_cols) - set([date_col]))  # actual number of excluded columns
    ncols = min(ncols, len(columns))
    nrows = ceil((len(columns) - n_excl) / ncols)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * width_per_ax, nrows * height_per_ax)
    )
    if date_col is not None:
        min_date, max_date = data[date_col].iloc[0], data[date_col].iloc[-1]  # same x_limits for all subplots: with dates
    else:
        min_x, max_x = data.index[0], data.index[-1]  # same x_limits for all subplots: without dates
    j = 0  # number of excluded columns count

    for i, col_name in zip(range(len(columns)), columns):
        if col_name in excl_cols or col_name == date_col:  # do not plot excluded columns
            j += 1
            continue
        idx_row, idx_col = (i - j) // ncols, (i - j) % ncols
        if ncols == 1 and nrows == 1:
            ax = axs 
        elif (ncols == 1) != (nrows == 1):  # XOR operator
            ax = axs[max(idx_row, idx_col)]
        else:
            ax = axs[idx_row, idx_col]
        if date_col is not None:
            ax.plot(data[date_col], data[col_name])
            ax.set_xlim(left=min_date, right=max_date)
        else:
            ax.plot(data[col_name])
            ax.set_xlim(left=min_x, right=max_x)
        if titles == None:
            ax.set_title(col_name)
        elif isinstance(titles, str):
            ax.set_title(col_name + ", " + titles)
        elif isinstance(titles, dict):
            ax.set_title(col_name + ", " + titles[col_name])
        else:
            ax.set_title(col_name + ", " + titles[i])
    fig.tight_layout()
    plt.show()



def test_stationarity(data, stat_conf_level='1%', columns=None, date_col=None, excl_cols=[], height_per_ax=2, width_per_ax=5, ncols=3, num_format='{:.3f}', print_graphs=True, print_test_results=True):
    if columns is None:
        columns = data.columns
    result_dict = {}
    if print_graphs:
        ncols = min(ncols, len(columns))
        n_excl = len(columns) - len(set(columns) - set(excl_cols) - set([date_col]))  # actual number of excluded columns
        nrows = ceil((len(columns) - n_excl) / ncols)
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * width_per_ax, nrows * height_per_ax)
        )
        if date_col is not None:
            min_date, max_date = data[date_col].iloc[0], data[date_col].iloc[-1]  # same x_limits for all subplots: with dates
        else:
            min_x, max_x = data.index[0], data.index[-1]  # same x_limits for all subplots: without dates
        j = 0  # number of excluded columns count
        if print_test_results:
            pad = 25  # for string padding in text_box
            txt_box_props = dict(boxstyle='round', alpha=0.8, facecolor='#EAEAF2')
    
        for i, col_name in zip(range(len(columns)), columns):
            if col_name in excl_cols or col_name == date_col:  # do not plot excluded columns
                j += 1
                continue
            idx_row, idx_col = (i - j) // ncols, (i - j) % ncols
            if ncols == 1 and nrows == 1:
                ax = axs 
            elif (ncols == 1) != (nrows == 1):  # XOR operator
                ax = axs[max(idx_row, idx_col)]
            else:
                ax = axs[idx_row, idx_col]
            col = data[col_name].dropna()  # no D-F test without dropna

            # Compute rolling statistics
            rol_mean = col.rolling(window=12, min_periods=1).mean()
            rol_std = col.rolling(window=12, min_periods=1).std()

            # Plot rolling statistics
            if date_col is not None:
                date_col_temp = data[date_col].loc[col.index]  # adjusting dates because of drop_na step
                ax.plot(date_col_temp, col, color='tab:blue', label='Original')
                ax.plot(date_col_temp, rol_mean, color='tab:red', label='Rolling Mean')
                ax.plot(date_col_temp, rol_std, color='darkgreen', label='Rolling Std')            
                ax.set_xlim(left=min_date, right=max_date)
            else:
                ax.plot(col, color='tab:blue', label='Original')
                ax.plot(rol_mean, color='tab:red', label='Rolling Mean')
                ax.plot(rol_std, color='darkgreen', label='Rolling Std')
                ax.set_xlim(left=min_x, right=max_x)

            # Perform Dickey-Fuller test
            df_test = adfuller(col, autolag='AIC')
            result_dict[col_name] = df_test[0] < df_test[4][stat_conf_level]
            if print_test_results:
                text_str = '\n'.join(('Dickey-Fuller test results'.center(pad + 6, ' '),
                                      'Test Statistic:'.ljust(pad) +  num_format.format(df_test[0]),
                                      'p-value:'.ljust(pad) +  num_format.format(df_test[1]),
                                      '# Lags used:'.ljust(pad) +  '{:d}'.format(df_test[2]),
                                      '# Observations used:'.ljust(pad) +  '{:d}'.format(df_test[3])
                                     ))
                for key,value in df_test[4].items():
                    text_str += '\nCritical Value {:s}'.format(key).ljust(pad) +  num_format.format(value)
                ax.text(0.03, 0.58, text_str, fontsize=12, bbox=txt_box_props, transform = ax.transAxes)
            ax.set_title(col_name)
            ax.legend(loc='lower left')
        fig.tight_layout()
        plt.show()
    else:
        for col_name in columns:
            if col_name in excl_cols or col_name == date_col:
                df_test = adfuller(data[col_name], autolag='AIC')
                result_dict[col_name] = df_test[0] < df_test[4][stat_conf_level]
    return result_dict

def remove_non_stationarity(data, stat_results):
    for col_name in settings.get('ORIGINAL_COLS'):
        if col_name in settings.get('NUM_COLS'):
            try:
                if col_name + '_pct_change' in stat_results.keys():
                    if not stat_results[col_name]:
                        data[col_name] = data[col_name + '_pct_change']
                        data.drop(columns=[col_name + '_pct_change'], inplace=True)
                        data.rename(columns={col_name: col_name + '_pct_change'}, inplace=True)
                        units[col_name + '_pct_change'] = '% change'
                        del units[col_name]
                    else:
                        data.drop(columns=[col_name + '_pct_change'], inplace=True)
                else:
                    if not stat_results[col_name]:
                        data.drop(columns=[col_name], inplace=True)
            except: pass