import numpy as np
import pandas as pd
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller  # Dickey-fuller test for stationarity of a series
from statsmodels.tsa.seasonal import seasonal_decompose  # seasonal decomposition of a signal (trend, seasonal, residuals)

from utils import settings
from visualization import visualization_basis

def test_stationarity(data,
                    stat_conf_level='1%',  # at which level of confidence to retain stationarity for Dickey-Fuller test
                    date_col=None, 
                    columns=None, 
                    excl_cols=[],
                    plot_graphs=True,
                    ncols=3, 
                    height_per_ax=4, 
                    width_per_ax=5, 
                    titles=None,
                    plot_test_results=True,
                    num_format='{:.3f}'
                ):
    """
    Perfoms Dickey-Fuller test for feature stationarity
    Input: data
        stat_conf_level: confidence level to retaing H1 for Dickey-Fuller test
        columns, excl_col, date_col: columns to analyze, to exclude, date_col for graph (excluded from analysis)
        height_per_ax, width_per_ax, ncols: plot params
        plot_graphs: whether to plot graphs with original, rolling mean, rolling std 
        plot_test_results: whether to plot D-F test results on graph
        num_format: numeric format for plot_test_results, default {:.3f}
    Output: results_dict: dict with analyzed columns as keys and boolean for stationarity as value 
    """
    if columns is None:
        columns = data.columns
    result_dict = {}
    if not plot_graphs:
        for col_name in columns:
            if col_name in excl_cols or col_name == date_col:
                df_test = adfuller(data[col_name], autolag='AIC')
                result_dict[col_name] = df_test[0] < df_test[4][stat_conf_level]
    else:
        subplot_params = {'plot_test_results': plot_test_results}
        if plot_test_results:
            subplot_params['num_format'] = num_format
            subplot_params['pad'] = 25  # for string padding in text_box
            subplot_params['txt_box_props'] = dict(boxstyle='round', alpha=0.8, facecolor='#EAEAF2')
        def stationarity_subplot(ax,
                                col_name, 
                                data=data, 
                                date_col=date_col, 
                                plot_test_results=False, 
                                txt_box_props=None, 
                                pad=None, 
                                num_format='{:.3f}'
                                ):
            # Perform Dickey-Fuller test
            col = data[col_name].dropna()  # no D-F test without dropna
            df_test = adfuller(col, autolag='AIC')
            result_dict[col_name] = df_test[0] < df_test[4][stat_conf_level]
            # Compute rolling statistics
            rol_mean = col.rolling(window=12, min_periods=1).mean()
            rol_std = col.rolling(window=12, min_periods=1).std()
            # Plot rolling statistics
            if date_col is not None:
                date_col_temp = data[date_col].loc[col.index]  # adjusting dates because of drop_na step
                ax.plot(date_col_temp, col, color=sns.color_palette()[0], alpha=0.9, label='Original')
                ax.plot(date_col_temp, rol_mean, color=sns.color_palette()[3], label='Rolling Mean')
                ax.plot(date_col_temp, rol_std, color=sns.color_palette()[2], label='Rolling Std')              
            else:
                ax.plot(col, color=sns.color_palette()[0], alpha=0.9, label='Original')
                ax.plot(rol_mean, color=sns.color_palette()[3], label='Rolling Mean')
                ax.plot(rol_std, color=sns.color_palette()[2], label='Rolling Std')
            if plot_test_results:
                text_str = '\n'.join(('Dickey-Fuller test results'.center(pad + 6, ' '),
                                    'Test Statistic:'.ljust(pad) +  num_format.format(df_test[0]),
                                    'p-value:'.ljust(pad) +  num_format.format(df_test[1]),
                                    '# Lags used:'.ljust(pad) +  '{:d}'.format(df_test[2]),
                                    '# Observations used:'.ljust(pad) +  '{:d}'.format(df_test[3])
                                    ))
                for key,value in df_test[4].items():
                    text_str += '\nCritical Value {:s}'.format(key).ljust(pad) +  num_format.format(value)
                ax.text(0.03, 0.58, text_str, fontsize=12, bbox=txt_box_props, transform = ax.transAxes)
            ax.legend(loc='lower left')
        visualization_basis(data=data,
                            subplot_function=stationarity_subplot,
                            subplot_params=subplot_params,
                            date_col=date_col,
                            columns=columns,
                            excl_cols=excl_cols,
                            ncols=ncols,
                            height_per_ax=height_per_ax,
                            width_per_ax=width_per_ax,
                            titles=titles
                            )
    return result_dict

def remove_non_stationary_features(data, stat_results):
    """
    Remove non stationary columns and keep one among a feature and its % change
    Input: data, 
        stat_results: dict with boolean for each column giving stationarity
    Ouput: data with either col_name if feature was stationary, or col_name_pct_change otherwise 
    """
    try:
        units = settings.get('UNITS')
        units_present = True
    except KeyError:
        units_present = False
    non_existing_cols = []
    dropped_cols = []
    kept_cols = []
    for col_name in settings.get('ORIGINAL_COLS'):
        if col_name in settings.get('NUM_COLS') and col_name in stat_results.keys():
            try:
                if col_name + '_pct_change' in stat_results.keys():
                    if not stat_results[col_name]:
                        data[col_name] = data[col_name + '_pct_change']
                        data.drop(columns=[col_name + '_pct_change'], inplace=True)
                        data.rename(columns={col_name: col_name + '_pct_change'}, inplace=True)
                        kept_cols.append(col_name + '_pct_change')
                        dropped_cols.append(col_name)
                        if units_present:
                            units[col_name + '_pct_change'] = '% change'
                    else:
                        data.drop(columns=[col_name + '_pct_change'], inplace=True)
                        kept_cols.append(col_name)
                        dropped_cols.append(col_name + '_pct_change')
                else:
                    if not stat_results[col_name]:
                        data.drop(columns=[col_name], inplace=True)
            except KeyError:
                non_existing_cols.append(col_name)
    if units_present: 
        settings.update('UNITS', units)
    if len(non_existing_cols) > 0:
        string = "\n".join(non_existing_cols)
        print("Following features not found in provided dataframe:")
        print(string)
    return kept_cols, dropped_cols

def seasonal_decomposition(data,
                        date_col=None, 
                        columns=None, 
                        excl_cols=[],
                        ncols=3, 
                        height_per_ax=4, 
                        width_per_ax=5, 
                        titles=None,
                        plot_graphs=False
                        ):
    """
    Perfoms sklearn seasonal decomposition
    Input: data
    date_col: for graph + needed to perform seasonal decomposition
        columns, excl_col, date_col: columns to analyze, to exclude
        height_per_ax, width_per_ax, ncols: plot params
        plot_graphs: whether to plot graphs with original, rolling mean, rolling std 
    Output: results_dict: dict with analyzed columns as keys and boolean for stationarity as value 
    """
    if columns is None:
        columns = data.columns
    n_features = len(columns)
    df_trend = pd.DataFrame(index=data[date_col])
    df_seas = pd.DataFrame(index=data[date_col])
    df_resid = pd.DataFrame(index=data[date_col])
    df = data.set_index(date_col)
    # if not plot_graphs:
    for col_name in columns:
        if col_name in excl_cols or col_name == date_col:
            continue
        col = df[col_name].dropna()
        decomposition = seasonal_decompose(col)
        df_trend[col_name + '_trend'] = decomposition.trend
        df_seas[col_name + '_seasonal'] = decomposition.seasonal
        df_resid[col_name + '_residual'] = decomposition.resid
    if plot_graphs:
        def seasonality_subplot(ax, col_name, df_original=df, df_trend=df_trend, df_seas=df_seas, df_resid=df_resid):
            col = df[col_name]
            # Plot decomposition
            ax.plot(col.index, col, color=sns.color_palette()[0], alpha=0.9, label="Original values")
            ax.plot(col.index, df_resid[col_name + '_residual'], color=sns.color_palette()[2], label='Residuals')
            ax.plot(col.index, df_seas[col_name + '_seasonal'], color=sns.color_palette()[1], label='Seasonality')
            ax.plot(col.index, df_trend[col_name + '_trend'], color=sns.color_palette()[3], label='Trend')
            ax.legend(loc='best', ncol=2)
        visualization_basis(data=data,
                            subplot_function=seasonality_subplot,
                            subplot_params={},
                            date_col=date_col,
                            columns=columns,
                            excl_cols=excl_cols,
                            ncols=ncols,
                            height_per_ax=height_per_ax,
                            width_per_ax=width_per_ax,
                            titles=titles
                            )
    df_trend.reset_index(level=['Date'], inplace=True)
    df_seas.reset_index(level=['Date'], inplace=True)
    df_resid.reset_index(level=['Date'], inplace=True)
    # df = pd.concat([df, df_trend, df_seas, df_resid], axis=1, keys=(['Original', 'Trend', 'Seasonality', 'Residuals']))
    return df_trend, df_seas, df_resid

def remove_seasonality(data, data_seas, threshold = 0.2, columns=None, excl_cols=[]):
    if columns == None:
        columns = data.columns
    for col_name in columns:
        if col_name == 'Date' or col_name in excl_cols:
            continue
        if data_seas[col_name + '_seasonal'].max() > threshold * data[col_name].std():
            data[col_name] = data[col_name] - data_seas[col_name+'_seasonal']
            print(col_name)

def vif_analysis(data, columns, threshold=10):
    excl_columns_vif = {}
    data_vif = data[columns]
    data_vif = data_vif.dropna()
    print(" VIF analysis starting: {:d} features ".format(len(data_vif.columns)).center(120, "-"))
    # first iteration
    VIF_factor = [variance_inflation_factor(data_vif.values, i) for i in range(data_vif.shape[1])]
    vif = pd.DataFrame({'VIF_factor': VIF_factor, 'features': data_vif.columns})
    max_vif_idx = vif['VIF_factor'].idxmax
    max_vif = vif['VIF_factor'].loc[max_vif_idx]
    
    # Exclude features based on previous VIF analysis (iterative process)
    while vif['VIF_factor'].loc[max_vif_idx] > threshold:
        max_vif_feature = vif['features'].loc[max_vif_idx]
        excl_columns_vif[max_vif_feature] = max_vif
        data_vif.drop(columns=[max_vif_feature], inplace=True)
        if len(data_vif.columns) == 1:
            print(" Only one column remaining, stopping vif analysis ".center(120, "-"))
            return vif, excl_columns_vif
        # New vif dataframe with fresh vif values
        VIF_factor = [variance_inflation_factor(data_vif.values, i) for i in range(data_vif.shape[1])]
        vif = pd.DataFrame({'VIF_factor': VIF_factor, 'features': data_vif.columns})
        # New index and max vif
        max_vif_idx = vif['VIF_factor'].idxmax
        max_vif = vif['VIF_factor'].loc[max_vif_idx]
    print(" VIF analysis succesfully completed ".center(120, "-"))
    return vif, excl_columns_vif

def shift_features(data, row_shifts = (1), columns=None, date_col='Date', excl_cols=[]):
    if columns == None:
        columns = data.columns
    data_shifted = pd.DataFrame(index = data.index)
    for col_name in columns:
        if col_name == date_col or col_name in excl_cols:
            continue
        for t in shifts:
            data_shifted[col_name + '_t-{:d}'.format(t)] = data[col_name].shift(t)
    return data_shifted



if __name__ == "__main__":
    data = pd.read_csv("Data/dataset_monthly.csv", sep=';')
    data_pct_change = pd.read_csv("Data/dataset_monthly_pct_change.csv", sep=';')
    data['Date'] = pd.to_datetime(data['Date'])
    data_pct_change['Date'] = pd.to_datetime(data_pct_change['Date'])

    NON_NUM_COLS = ['Date', 'Recession', 'potus', 'houseOfRep', 'fedChair']  # non numeric features (boolean, date, etc)
    NUM_COLS = [x for x in data.columns if x not in NON_NUM_COLS]

    data = pd.merge(data, data_pct_change, on='Date')
    sns.set()
    # test_stationarity(data=data,
    #                 columns=NUM_COLS[:6],
    #                date_col='Date',
    #                ncols=4)
    seasonal_decomposition(data=data,
                    columns=NUM_COLS[:6],
                   date_col='Date',
                   ncols=4,
                   plot_graphs=True)


def test_stationarity_bis(data,
                    stat_conf_level='1%',  # at which level of confidence to retain stationarity for Dickey-Fuller test
                    columns=None, 
                    date_col=None, 
                    excl_cols=[],
                    height_per_ax=2,
                    width_per_ax=5,
                    ncols=3,
                    plot_graphs=True,
                    plot_test_results=True,
                    num_format='{:.3f}'
                ):
    """
    Perfoms Dickey-Fuller test for feature stationarity
    Input: data
        stat_conf_level: confidence level to retaing H1 for Dickey-Fuller test
        columns, excl_col, date_col: columns to analyze, to exclude, date_col for graph (excluded from analysis)
        height_per_ax, width_per_ax, ncols: plot params
        plot_graphs: whether to plot graphs with original, rolling mean, rolling std 
        plot_test_results: whether to plot D-F test results on graph
        num_format: numeric format for plot_test_results, default {:.3f}
    Output: results_dict: dict with analyzed columns as keys and boolean for stationarity as value 
    """
    if columns is None:
        columns = data.columns
    result_dict = {}
    if plot_graphs:
        ncols = min(ncols, len(columns))
        n_excl = len(columns) - len(set(columns) - set(excl_cols) - set([date_col]))  # actual number of excluded columns
        nrows = ceil((len(columns) - n_excl) / ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * width_per_ax, nrows * height_per_ax))
        if date_col is not None:
            min_x, max_x = data[date_col].iloc[0], data[date_col].iloc[-1]  # same x_limits for all subplots: with dates
        else:
            min_x, max_x = data.index[0], data.index[-1]  # same x_limits for all subplots: without dates
        j = 0  # number of excluded columns count
        if plot_test_results:
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
                ax.plot(date_col_temp, col, color=sns.color_palette()[0], alpha=0.9, label='Original')
                ax.plot(date_col_temp, rol_mean, color=sns.color_palette()[3], label='Rolling Mean')
                ax.plot(date_col_temp, rol_std, color=sns.color_palette()[2], label='Rolling Std')            
                ax.set_xlim(left=min_x, right=max_x)
            else:
                ax.plot(col, color=sns.color_palette()[0], alpha=0.9, label='Original')
                ax.plot(rol_mean, color=sns.color_palette()[3], label='Rolling Mean')
                ax.plot(rol_std, color=sns.color_palette()[2], label='Rolling Std')
                ax.set_xlim(left=min_x, right=max_x)

            # Perform Dickey-Fuller test
            df_test = adfuller(col, autolag='AIC')
            result_dict[col_name] = df_test[0] < df_test[4][stat_conf_level]
            if plot_test_results:
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


def seasonal_decomposition_bis(data, date_col, columns=None, excl_cols=None, height_per_ax=4, width_per_ax=5, ncols=3, plot_graphs=False):
    """
    Perfoms sklearn seasonal decomposition
    Input: data
    date_col: for graph + needed to perform seasonal decomposition
        columns, excl_col, date_col: columns to analyze, to exclude
        height_per_ax, width_per_ax, ncols: plot params
        plot_graphs: whether to plot graphs with original, rolling mean, rolling std 
    Output: results_dict: dict with analyzed columns as keys and boolean for stationarity as value 
    """
    if columns is None:
        columns = data.columns
    n_features = len(columns)
    df_trend = pd.DataFrame(index=data[date_col])
    df_seas = pd.DataFrame(index=data[date_col])
    df_resid = pd.DataFrame(index=data[date_col])
    df = data.set_index(date_col)
    if plot_graphs:
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
            ax.plot(col.index, col, color=sns.color_palette()[0], alpha=0.9, label="Original values")
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
            if col_name in excl_cols or col_name == date_col:
                continue
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