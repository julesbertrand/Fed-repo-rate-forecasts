from functools import reduce
import datetime as dt

import pandas as pd

from lib.data_retrieval.data_getters import FREDGetter, USBLSGetter, OECDGetter
from lib.utils import save_file, merge_df_list_on

GETTERS = {"FRED": FREDGetter, "USBLS": USBLSGetter, "OECD": OECDGetter}


def get_data_from_apis(
    api_keys: dict,
    api_params: dict,
    data_start_date: dt.date,
    data_end_date: dt.date = None,
    metadata_filepath: str = "./data/metadata.yaml",
    providers: list = ["FRED", "USBLS", "OECD"],
) -> pd.DataFrame:
    """Fetch, clean and merge data from Fred, USBLS and OECD in one DataFrame
    Save related metadata in one yaml file
    """
    getters = {k: v for k, v in GETTERS.items() if k in providers}

    for provider in getters.keys():
        if provider not in api_keys.keys():
            raise KeyError(f"No API Key was provided for {provider}.")
        if provider not in api_params.keys():
            raise KeyError(f"No API requests parameters were provided for {provider}.")

    metadata_list = []
    obs_df_list = []
    # pylint: disable=invalid-name
    for provider, Getter in getters.items():
        getter = Getter(api_key=api_keys[provider])
        data, info = getter.get_data(
            series_params=api_params[provider]["series_params"][:2],
            start_date=data_start_date,
            end_date=data_end_date,
        )
        metadata_list.append(info)
        obs_df_list.append(data)

    merged_info = reduce(lambda left, right: left + right, metadata_list)
    save_file(filepath=metadata_filepath, data=merged_info)

    merged_data = merge_df_list_on(obs_df_list, on="date")
    return merged_data
