from functools import reduce
from typing import List
import pandas as pd


def merge_df_list_on(df_list: List[pd.DataFrame], on: str) -> pd.DataFrame:
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
    return merged_data
