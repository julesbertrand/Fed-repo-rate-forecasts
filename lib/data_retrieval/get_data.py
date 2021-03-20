from functools import reduce
import datetime as dt

import pandas as pd

from lib.data_retrieval.data_getters import FREDGetter, USBLSGetter, OECDGetter
from lib.utils.df_utils import merge_df_list_on
from lib.utils.files import save_yaml
from lib.utils.path import get_projet_root, create_dir_if_missing
from lib.utils.errors import InvalidAPIKey, InvalidAPIRequestsParams

GETTERS = {"FRED": FREDGetter, "USBLS": USBLSGetter, "OECD": OECDGetter}


def get_data_from_apis(
    api_keys: dict,
    api_params: dict,
    data_start_date: dt.date,
    data_end_date: dt.date = None,
    providers: list = None,
    save_dirpath: str = None,
) -> pd.DataFrame:
    """Fetch, clean and merge data from Fred, USBLS and OECD in one DataFrame
    Save related metadata in one yaml file
    """
    if providers is None:
        getters = GETTERS
    else:
        getters = {k: v for k, v in GETTERS.items() if k in providers}

    for provider in getters.keys():
        if provider not in api_keys.keys():
            raise InvalidAPIKey(f"No API Key was provided for {provider}.")
        if provider not in api_params.keys():
            msg = f"No API requests parameters were provided for {provider}."
            raise InvalidAPIRequestsParams(msg)

    metadata_list = []
    obs_df_list = []
    # pylint: disable=invalid-name
    for provider, Getter in getters.items():
        getter = Getter(api_key=api_keys[provider])
        data, metadata = getter.get_data(
            series_params=api_params[provider]["series_params"],
            start_date=data_start_date,
            end_date=data_end_date,
        )
        metadata_list.append(metadata)
        obs_df_list.append(data)

    merged_metadata = reduce(lambda left, right: left + right, metadata_list)
    merged_data = merge_df_list_on(obs_df_list, on="date")
    return merged_data, merged_metadata
