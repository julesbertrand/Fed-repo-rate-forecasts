import pytest
import numpy as np
import pandas as pd

from lib.utils.df_utils import merge_df_list_on


def test_merge_df_on_list():
    df_1 = pd.DataFrame(
        {"Date": [1, 2, 3, 4, 5, 7, 10], "Value_1": ["a", "b", "c", "d", "e", "g", "j"]}
    )
    df_2 = pd.DataFrame(
        {"Date": [1, 2, 3, 4, 5, 6, 8], "Value_2": ["a", "b", "c", "d", "e", "f", "h"]}
    )
    df_3 = pd.DataFrame({"Date": [1, 3, 4, 5, 9], "Value_3": ["a", "c", "d", "e", "i"]})
    df_result = pd.DataFrame(
        {
            "Date": list(range(1, 11)),
            "Value_1": ["a", "b", "c", "d", "e", np.nan, "g", np.nan, np.nan, "j"],
            "Value_2": ["a", "b", "c", "d", "e", "f", np.nan, "h", np.nan, np.nan],
            "Value_3": [
                "a",
                np.nan,
                "c",
                "d",
                "e",
                np.nan,
                np.nan,
                np.nan,
                "i",
                np.nan,
            ],
        }
    )
    print(merge_df_list_on([df_1, df_2, df_3], on="Date"))
    print(df_result)
    assert merge_df_list_on([df_1, df_2, df_3], on="Date").equals(df_result)
