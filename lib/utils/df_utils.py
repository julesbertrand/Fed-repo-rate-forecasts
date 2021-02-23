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
    merged_data = reduce(
        lambda left, right: pd.merge(left, right, on=on, how="outer"),
        df_list,
    )
    if sort:
        merged_data = merged_data.sort_values(by=on).reset_index(drop=True)
    return merged_data
