from functools import reduce
from typing import List

import pandas as pd


def merge_df_list_on(df_list: List[pd.DataFrame], on: str, sort: bool = True) -> pd.DataFrame:
    """Merge (outer) a list of dataframes with at least one common column together

    Parameters
    ----------
    df_list: List[pd.DataFrame]

    Returns
    -------
    pd.DataFrame
    """
    merged_data = reduce(lambda left, right: pd.merge(left, right, on=on, how="outer"), df_list)
    if sort:
        merged_data = merged_data.sort_values(by=on).reset_index(drop=True)
    return merged_data


def check_columns(data: pd.DataFrame, columns: list = None, excl_columns: list = None) -> list:
    """
    Check that columns is a list
    Filters excl columns and columns not in dataset
    Check it is not empty

    Raises
    ------
    TypeError
        If columns is not a list
    """
    if excl_columns is None:
        excl_columns = []
    if columns is None:
        columns = list(data.columns)

    if not isinstance(columns, list) or not isinstance(excl_columns, list):
        raise TypeError("Invalid input: columns and excl_columns must be lists")
    columns = [col for col in columns if (col in data.columns and col not in excl_columns)]
    if not columns:
        msg = "Invalid input: columns does not overlap with data.columns \
or you put too many names in excl_columns"
        raise ValueError(msg)
    return columns


def filter_df(df: pd.DataFrame, filters: dict = None):
    """Filter pd.DataFrame in columns filters.keys() based on filters.values()

    Parameters
    ----------
    df: pd.DataFrame
    filters: dict
        dict with columns of df to be used to filter as keys and \
values to find in these columns as values. filters.values() can be:
        - None, 'all', 'dropna' => will only dropna on the specified column
        - list or set of authorized values in the specified column
        - another value (float, int, str, ...) to filter the column on

    Returns
    -------
    pd.DataFrame
        filtered copy of parameter df
    """
    if filters is None:
        return df

    data = df.copy()
    for col, val in filters.items():
        if val in [None, "all", "dropna"]:
            data = data.dropna(subset=[col])
        elif isinstance(val, (list, set)):
            data = data[data[col].str.contains("|".join(val), na=False)]
        else:
            data = data[data[col] == val]
    return data
