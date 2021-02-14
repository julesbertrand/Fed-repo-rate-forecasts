from functools import reduce
import requests
import pandas as pd
from config.config import API_URLS


def get_fred_data(api_key: str, series_params: list, start_date: str = None, end_date: str = None):
    """Get multiple series data and info from fred api
    and group in two lists (one info, one data)

    Parameters
    ----------
    api_key: str
        Your api key to access fred api
        see
    series_params: list
        List of dicts with series_params, including at least 'series_id'
    start_date: str
        The start of the observation period. YYYY-MM-DD
        https://fred.stlouisfed.org/docs/api/fred/series_observations.html#observation_start
    end_date: str
        The start of the observation period. YYYY-MM-DD
        https://fred.stlouisfed.org/docs/api/fred/series_observations.html#observation_end


    Returns
    -------
    info_list: list
        List of dictionaries with metadata for every series retrieved
    data_list: list
        List of dictionaries with data reponse from fred api in json format
    """
    format_params = {
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    info_list = []
    obs_list = []
    for params in series_params:
        params.update(format_params)

        info_resp = requests.get(API_URLS["FRED_API_URL_SER"], params=params)
        info_resp.raise_for_status()
        info_series = info_resp.json()["seriess"][0]

        obs_resp = requests.get(
            API_URLS["FRED_API_URL_OBS"],
            params=params,
        )
        obs_resp.raise_for_status()
        obs_content = obs_resp.json()

        info_dict = {
            "name": info_series.get("title"),
            "series_id": params["series_id"],
            "frequency": params.get("frequency"),
            "units": params.get("units"),
            "aggregation_method": params.get("aggregation_method"),
            "seasonal_adjustment": params.get("seasonal_adjustment"),
        }

        info_list.append(info_dict)
        obs_list.append(obs_content)
    return obs_list, info_list


def clean_fred_data(obs_data_list: list, info_data_list: list) -> pd.DataFrame:
    """Clean output from get_fred_data to get an aggregated dataframe

    Parameters
    ----------
    obs_data_list: list
        List of all fred responses bodies in json format
    info_data_list
        List of dicts containing metadata about every retrieved serie

    Returns
    -------
    pd.DataFrame
        Aggregated data with names, in the right format
    """
    cleaned_data_list = []
    for i, obs_data in enumerate(obs_data_list):
        cleaned_data = clean_fred_serie(obs_data=obs_data, info_data=info_data_list[i])
        cleaned_data_list.append(cleaned_data)

    merged_data = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"),
        cleaned_data_list,
    )

    merged_data["date"] = pd.to_datetime(merged_data["date"], format="%Y-%m-%d")
    data_types = {c: float for c in merged_data.columns.difference(["date"])}
    merged_data = merged_data.astype(data_types)
    return merged_data


def clean_fred_serie(obs_data, info_data: dict) -> pd.DataFrame:
    """Clean a series json response from fred api

    Parameters
    ----------
    obs_data: json or dict
        body of the respone of the api
    info_data: dict
        Information on teh serie, including at least 'series_id'

    Raises
    ------
    KeyError
        If 'series_id' not in info_data keys

    Returns
    -------
    pd.DataFrame
        df with columns 'date' and a new name based on info_data for cleaned serie
    """
    if info_data.get("series_id") is None:
        raise KeyError("No 'series_id' in info_data: this key is mandatory.")

    data = pd.json_normalize(obs_data["observations"])
    data.drop(columns=["realtime_start", "realtime_end"], inplace=True)
    name_components = []
    for field in [
        "series_id",
        "frequency",
        "units",
        "aggregation_method",
        "seasonal_adjustment",
    ]:
        if info_data.get(field):
            name_components.append(info_data[field])
    series_name = "_".join(name_components)
    data.rename(columns={"value": series_name}, inplace=True)
    return data
