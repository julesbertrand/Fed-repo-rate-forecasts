import numpy as np
import pandas as pd
import datetime as dt
from math import ceil  # to compute nrows

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller  # Dickey-fuller test for stationarity of a series
from statsmodels.tsa.seasonal import seasonal_decompose  # seasonal decomposition of a signal (trend, seasonal, residuals)

from utils import settings

def test_stationarity(data, stat_conf_level='1%', columns=None, date_col=None, excl_cols=[], height_per_ax=2, width_per_ax=5, ncols=3, num_format='{:.3f}', print_graphs=True, print_test_results=True):
    if columns is None:
        columns = data.columns
    result_dict = {}
    if print_graphs:
        ncols = min(ncols, len(columns))
        n_excl = len(columns) - len(set(columns) - set(excl_cols) - set([date_col]))  # actual number of excluded columns
        nrows = ceil((len(columns) - n_excl) / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * width_per_ax, nrows * height_per_ax))
        if date_col is not None:
            min_x, max_x = data[date_col].iloc[0], data[date_col].iloc[-1]  # same x_limits for all subplots: with dates
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
                ax.plot(date_col_temp, col, color=sns.color_palette()[0], alpha=0.7, label='Original')
                ax.plot(date_col_temp, rol_mean, color=sns.color_palette()[3], label='Rolling Mean')
                ax.plot(date_col_temp, rol_std, color=sns.color_palette()[2], label='Rolling Std')            
                ax.set_xlim(left=min_x, right=max_x)
            else:
                ax.plot(col, color=sns.color_palette()[0], label='Original')
                ax.plot(rol_mean, color=sns.color_palette()[3], label='Rolling Mean')
                ax.plot(rol_std, color=sns.color_palette()[2], label='Rolling Std')
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
            ax.set_title(col_name, size=14)
            ax.legend(loc='lower left')
        fig.tight_layout()
        plt.show()
    else:
        for col_name in columns:
            if col_name in excl_cols or col_name == date_col:
                df_test = adfuller(data[col_name], autolag='AIC')
                result_dict[col_name] = df_test[0] < df_test[4][stat_conf_level]
    return result_dict

def remove_non_stationary_features(data, stat_results):
    try:
        units = settings.get('UNITS')
        units_present = True
    except KeyError:
        units_present = False
        return
    non_existing_cols = []
    for col_name in settings.get('ORIGINAL_COLS'):
        if col_name in settings.get('NUM_COLS') and col_name in stat_results.keys():
            try:
                if col_name + '_pct_change' in stat_results.keys():
                    if not stat_results[col_name]:
                        data[col_name] = data[col_name + '_pct_change']
                        data.drop(columns=[col_name + '_pct_change'], inplace=True)
                        data.rename(columns={col_name: col_name + '_pct_change'}, inplace=True)
                        if units_present:
                            units[col_name + '_pct_change'] = '% change'
                            del units[col_name]
                    else:
                        data.drop(columns=[col_name + '_pct_change'], inplace=True)
                else:
                    if not stat_results[col_name]:
                        data.drop(columns=[col_name], inplace=True)
            except KeyError:
                non_existing_cols.append(col_name)
    settings.replace('UNITS', units)
    if len(non_existing_cols) > 0:
        string = "\n".join(non_existing_cols)
        print("Following features not found in provided dataframe:")
        print(string)

def seasonal_decomposition(data, date_col, columns=None, excl_cols=[], height_per_ax=4, width_per_ax=5, ncols=3, print_graphs=False):
    if columns is None:
        columns = data.columns
    n_features = len(columns)
    df_trend = pd.DataFrame(index=data[date_col])
    df_seas = pd.DataFrame(index=data[date_col])
    df_resid = pd.DataFrame(index=data[date_col])
    df = data.set_index(date_col)
    if print_graphs:
        n_excl = n_features - len(set(columns) - set(excl_cols) - set([date_col]))  # actual number of excluded columns
        ncols = min(ncols, n_features)
        nrows = ceil((n_features - n_excl) / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * width_per_ax, nrows * height_per_ax))
        min_x, max_x = data[date_col].iloc[0], data[date_col].iloc[-1]  # same x_limits for all subplots: with dates
        j = 0  # number of excluded columns count

        for i, col_name in zip(range(n_features), columns):
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

            # Decomposition
            col = df[col_name].dropna()
            decomposition = seasonal_decompose(col)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

            # Plot decomposition
            ax.plot(col.index, col, color=sns.color_palette()[0], label="Original values", alpha=.7)
            ax.plot(col.index, residual, color=sns.color_palette()[2], label='Residuals')
            ax.plot(col.index, seasonal, color=sns.color_palette()[1], label='Seasonality')
            ax.plot(col.index, trend, color=sns.color_palette()[3], label='Trend')
            ax.set_xlim(left=min_x, right=max_x)
            ax.set_title(col_name, size=14)
            ax.legend(loc='best', ncol=2)
            df_trend[col_name +   '_trend'] = trend
            df_seas[col_name + '_seasonal'] = seasonal
            df_resid[col_name + '_residual'] = residual
        fig.tight_layout()
        plt.show()
    else:
        for col_name in columns:
            col = df[col_name].dropna()
            decomposition = seasonal_decompose(col)
            df_trend[col_name + '_trend'] = decomposition.trend
            df_seas[col_name + '_seasonal'] = decomposition.seasonal
            df_resid[col_name + '_residual'] = decomposition.resid
    df_trend.reset_index(level=['Date'], inplace=True)
    df_seas.reset_index(level=['Date'], inplace=True)
    df_resid.reset_index(level=['Date'], inplace=True)
    # df = pd.concat([df, df_trend, df_seas, df_resid], axis=1, keys=(['Original', 'Trend', 'Seasonality', 'Residuals']))
    return df_trend, df_seas, df_resid