from functools import reduce
from typing import Callable
import pandas as pd


def clean_received_data(
    obs_data_list: list, info_data_list: list, series_cleaner: Callable
) -> pd.DataFrame:
    """Clean output from get_[insert provider]_data to get an aggregated dataframe

    Parameters
    ----------
    obs_data_list: list
        List of all fred responses bodies in json format
    info_data_list
        List of dicts containing metadata about every retrieved series

    Raises
    ------
    KeyError
        If 'series_id' not in info_data keys

    Returns
    -------
    pd.DataFrame
        Aggregated data with names, in the right format
    """
    cleaned_data_list = []
    for i, obs_data in enumerate(obs_data_list):
        if info_data_list[i].get("series_id") is None:
            raise KeyError("No 'series_id' in info_data: this key is mandatory.")
        if len(obs_data) > 0:
            cleaned_series = series_cleaner(obs_data=obs_data)
        else:  # no obs
            cleaned_series = pd.DataFrame(columns=["date", "value"])
        series_name = give_name_to_series(info_data_list[i])
        cleaned_series.rename(columns={"value": series_name}, inplace=True)
        cleaned_data_list.append(cleaned_series)

    merged_data = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        cleaned_data_list,
    )

    merged_data["date"] = pd.to_datetime(merged_data["date"], format="%Y-%m-%d")
    data_types = {c: float for c in merged_data.columns.difference(["date"])}
    merged_data = merged_data.astype(data_types)
    return merged_data


def give_name_to_series(info_data: dict) -> str:
    """Give name to series using metadata about the series"""
    name_components = []
    if "series_id" not in info_data.keys():
        raise KeyError("No 'series_id' in info_data: this key is mandatory.")
    fields_list = [
        "series_id",
        "frequency",
        "units",
        "aggregation_method",
        "seasonal_adjustment",
    ]
    for field in fields_list:
        if info_data.get(field):
            name_components.append(info_data[field])
    series_name = "_".join(name_components)
    return series_name


def clean_fred_series(obs_data: list) -> pd.DataFrame:
    """Clean a series json response from fred api

    Parameters
    ----------
    obs_data: json or dict
        body of the respone of the api

    Returns
    -------
    pd.DataFrame
        df with columns 'date' and 'value' cleaned
    """
    series = pd.json_normalize(obs_data["observations"])
    series.drop(columns=["realtime_start", "realtime_end"], inplace=True)
    return series


def clean_usbls_series(obs_data: list) -> pd.DataFrame:
    """Clean a series json response from fred api

    Parameters
    ----------
    obs_data: json or dict
        body of the respone of the api

    Returns
    -------
    pd.DataFrame
        df with columns 'date' and a new name based on info_data for cleaned series
    """
    series = pd.DataFrame(obs_data)
    series["date"] = pd.to_datetime(
        series["year"] + "-" + series["periodName"] + "-01", format="%Y-%B-%d"
    )
    series.drop(columns=series.columns.difference(["value", "date"]), inplace=True)
    return series
