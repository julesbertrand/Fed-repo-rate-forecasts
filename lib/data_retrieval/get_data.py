from functools import reduce
import datetime as dt

import pandas as pd

from lib.data_retrieval.data_getters import FREDGetter, USBLSGetter
from lib.utils import save_file


def get_data_from_apis(
    api_keys: dict, api_params: dict, data_start_date: dt.date, data_end_date: dt.date = None
) -> pd.DataFrame:
    getters = {"FRED": FREDGetter, "USBLS": USBLSGetter}
    metadata_list = []
    data_list = []

    # pylint: disable=invalid-name
    for provider, Getter in getters.items():
        getter = Getter(api_key=api_keys[provider])
        data, info = getter.get_data(
            series_params=api_params[provider]["series_params"][:2],
            start_date=data_start_date,
            end_date=data_end_date
        )
        metadata_list.append(info)
        data_list.append(data)

    merged_info = reduce(lambda left, right: left + right, metadata_list)
    save_file(filepath="./data/metadata.yaml", data=merged_info)

    merged_data = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        data_list,
    )
    return merged_data
