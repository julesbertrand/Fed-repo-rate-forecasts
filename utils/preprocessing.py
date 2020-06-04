import numpy as np
import datetime as dt
import pandas as pd
from functools import reduce
from glob import glob
import ntpath

from utils import settings

def data_processing_end(df, columns=None, date_format="%Y-%m-%d", month_offset=0):
    """
    End of preprocessing for each data file: put Date column as index, change type of given columns to numeric type
    input: dataframe with date column
            date format of date column
            colums to convert to float64
            month_offset: as all data will be dated on the end of month, month_offset is here to get as close as possible to the actual date of the data
    output: dataframe cleaned with column date as index and all given columns to float64 type
    """
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format=date_format) - pd.offsets.MonthEnd(month_offset)
        df = df[df['Date'] < settings.get('LAST_AVLBLE_DATE')]
    columns = columns if columns is not None else df.columns[1:]
    for col in columns:
        df[col] = pd.to_numeric(df[col])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def get_file_names(path = "Data/"):
    """
    Retrieve all data files names in a directory 
    input: path
    output: list of file names
    """
    temp = glob(path + "/*.csv")
    file_names = []
    for file_name in temp:
        _, file_name = ntpath.split(file_name) 
        file_names.append(file_name)
    return file_names

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
        data = pd.melt(data, 
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