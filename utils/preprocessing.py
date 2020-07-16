import numpy as np
import pandas as pd
import datetime as dt
from functools import reduce

from utils import settings
from utils.visualization import visualize_features, units

def data_processing_end(data, columns=None, visualize=False, date_col=None, date_format="%Y-%m-%d", month_offset=0):
    """
    End of preprocessing for each data file: put Date column as index, change type of given columns to numeric type
    input: dataframe with date column
            date format of date column
            colums to convert to float64
            visualize to call utils.visualization.visualize_features
            month_offset: as all data will be dated on the end of month, month_offset is here to get as close as possible to the actual date of the data
    output: dataframe cleaned with column date as index and all given columns to float64 type
    """
    data.rename(columns=dict(zip(data.columns, [col_name.replace(" ", "_") for col_name in data.columns])), inplace=True)
    if columns == None:
        columns = data.columns
    if date_col in columns:
        columns = list(columns)
        columns.remove(date_col)
    for col_name in columns:
        # if col_name != date_col:
        data[col_name] = pd.to_numeric(data[col_name])
    if date_col is not None:
        data['Date'] = pd.to_datetime(data[date_col], format=date_format) - pd.offsets.MonthEnd(month_offset)
        if date_col != 'Date':
            data.drop(columns=[date_col], inplace=True)
        data = data[data['Date'] < settings.get('LAST_AVLBLE_DATE')]
    if visualize and len(columns) > 0:
        visualize_features(
            data=data,
            columns=columns,
            date_col='Date',
            ncols=3,
        )
    data.set_index('Date', inplace=True)
    data.sort_index(inplace=True)
    return data

def data_processing_us_bls(file_names, path):
    """
    process all data files from US Bureau of Labor Statistics and lerge them into one dataframe
    input: list of file names, path to the folder
    ouput: dataframe
    """
    data_list = []
    for file_name in file_names:
        data = pd.read_csv(path + file_name)
        data = data[10:]
        data.columns = data.iloc[0]
        data = data[1:]
        value_name = file_name.replace("economics_", "").replace(".csv", "")
        data = pd.melt(data,  # pivot table to column table
                       id_vars = 'Year',
                       value_vars = list(data.columns[1:13]),
                       var_name = 'Month',
                       value_name = value_name
                      )
        data[value_name] = pd.to_numeric(data[value_name])
        data['Date'] = pd.to_datetime(data['Year'] + '-' + data['Month'], format = '%Y-%b')  # create datetime formated column
        data['Date'] += pd.tseries.offsets.MonthEnd()  # end of month offset
        data.drop(columns=['Year', 'Month'], inplace = True)
        data_list.append(data)
    df = reduce(lambda left,right: pd.merge(left, right, on='Date', how='outer'), data_list)
    df = df[df['Date'] < settings.get('LAST_AVLBLE_DATE')]  # removing too recent rows (no data)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

if __name__ == "__main__":
    data = pd.read_csv(settings.get('PATH_TO_DATA') + "WTISPLC.csv")
    data.rename(columns={
        'DATE':'Date',
        'WTISPLC':'WTI oil price'
        }, inplace=True)
    data_oil = data_processing_end(data, month_offset=1, visualize=True)