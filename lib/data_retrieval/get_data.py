import datetime as dt
from functools import reduce
from pathlib import Path

import pandas as pd
from loguru import logger

from lib.config import ROOT_PATH
from lib.data_retrieval.data_getters import FREDGetter, OECDGetter, USBLSGetter
from lib.utils.df_utils import merge_df_list_on
from lib.utils.errors import InvalidAPIKey, InvalidAPIRequestsParams
from lib.utils.files import save_yaml
from lib.utils.path import create_dir_if_missing

_GETTERS = {"FRED": FREDGetter, "USBLS": USBLSGetter, "OECD": OECDGetter}


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
        getters = _GETTERS
    else:
        getters = {k: v for k, v in _GETTERS.items() if k in providers}

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

    if save_dirpath is not None:
        date = dt.date.today().strftime("%Y%m%d")
        dirpath = Path(ROOT_PATH) / save_dirpath / date
        create_dir_if_missing(dirpath)
        data_path = dirpath / "raw_data.csv"
        data.to_csv(data_path, sep=";", index=False, encoding="utf-8")
        metadata_path = dirpath / "metadata.yaml"
        save_yaml(metadata, metadata_path)
        logger.success(f"All data retrieved, cleaned and saved to {dirpath}.")

    return merged_data, merged_metadata
